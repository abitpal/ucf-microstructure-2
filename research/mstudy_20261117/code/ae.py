"""Part 1c: sequence autoencoder on the last 64 book events -> 16-d microstructure embedding.
Probe: can a classifier recover market type from the embedding alone? Also a supervised GRU classifier whose
penultimate activations feed the trade model."""
import sys, os, json, glob, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from common import *
from features import FEATS
from classify import split_mask, FINE_KEEP, plot_cm

torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
STRICT_AE = bool(os.environ.get('STRICT_AE'))
SFX = '_strict' if STRICT_AE else ''
REP = f'{ROOT}/report/ae{SFX}'
os.makedirs(REP, exist_ok=True)
ZD = 16


class SeqAE(nn.Module):
    def __init__(self, d_in=16, h=128, z=ZD, L=64):
        super().__init__()
        self.L = L
        self.inp = nn.Linear(d_in, h)
        self.enc = nn.GRU(h, h, num_layers=2, batch_first=True)
        self.to_z = nn.Linear(h, z)
        self.from_z = nn.Linear(z, h)
        self.pos = nn.Parameter(torch.randn(L, h) * .1)
        self.dec = nn.GRU(h, h, num_layers=1, batch_first=True)
        self.out = nn.Linear(h, d_in)

    def encode(self, x):
        _, hN = self.enc(torch.relu(self.inp(x)))
        return self.to_z(hN[-1])

    def forward(self, x):
        z = self.encode(x)
        h0 = torch.tanh(self.from_z(z))
        dec_in = self.pos.unsqueeze(0).expand(x.shape[0], -1, -1) + h0.unsqueeze(1)
        y, _ = self.dec(dec_in, h0.unsqueeze(0))
        return self.out(y), z


class SeqClf(nn.Module):
    def __init__(self, d_in=16, h=96, ncls=13):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        self.enc = nn.GRU(h, h, num_layers=2, batch_first=True)
        self.head1 = nn.Linear(h, 32)
        self.head2 = nn.Linear(32, ncls)

    def act(self, x):
        _, hN = self.enc(torch.relu(self.inp(x)))
        return torch.relu(self.head1(hN[-1]))

    def forward(self, x):
        return self.head2(self.act(x))


def load():
    fs = sorted(glob.glob(f'{DATA}/feat/win/*.parquet'))
    metas, seqs = [], []
    for f in fs:
        tag = os.path.basename(f)[:-8]
        m = pd.read_parquet(f, columns=['ticker', 'day', 'coarse', 'fine'] + [f'N64__{x}' for x in FEATS] + ['fw_1.0'])
        s = np.load(f'{DATA}/feat/seq/{tag}.npy')
        metas.append(m); seqs.append(s)
    M = pd.concat(metas, ignore_index=True); S = np.concatenate(seqs)
    keep = M.fine.isin(FINE_KEEP).values
    M = M[keep].reset_index(drop=True); S = S[keep]
    M['test'] = split_mask(M)
    return M, S


def balanced_idx(M, per=25000, test=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for f, g in M[M.test == test].groupby('fine'):
        ix = g.index.values
        out.append(rng.choice(ix, min(per, len(ix)), replace=False))
    return np.sort(np.concatenate(out))


TICK_DIMS = [5, 8, 10]  # level distance (ticks), spread (ticks), mid change (ticks)


def to_t(S):
    x = torch.tensor(np.nan_to_num(S.astype(np.float32)))
    if STRICT_AE:
        x[..., TICK_DIMS] = 0.0
    return torch.clamp(x, -8, 8)


def train_ae(S, tr_idx, va_idx, run, budget_s=900):
    model = SeqAE()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr = to_t(S[tr_idx]); Xva = to_t(S[va_idx[:8000]])
    t0 = time.time(); step = 0; ep = 0
    while time.time() - t0 < budget_s:
        perm = torch.randperm(len(Xtr))
        model.train()
        for i in range(0, len(Xtr), 512):
            b = Xtr[perm[i:i + 512]]
            rec, z = model(b)
            loss = ((rec - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 50 == 0:
                run.log({'ae/train_mse': float(loss), 'ae/step': step})
            if time.time() - t0 > budget_s:
                break
        model.eval()
        with torch.no_grad():
            rv, _ = model(Xva)
            vl = float(((rv - Xva) ** 2).mean())
            base = float(((Xva - Xtr.mean((0, 1), keepdim=True)) ** 2).mean())
        run.log({'ae/val_mse': vl, 'ae/val_mse_ratio_vs_mean': vl / base, 'ae/epoch': ep})
        log(f'  AE ep{ep} val_mse={vl:.4f} (mean-baseline {base:.4f}) {time.time()-t0:.0f}s', 'analysis.log')
        ep += 1
    return model, vl, base


def train_clf(S, y, tr_idx, va_idx, ncls, run, budget_s=600):
    model = SeqClf(ncls=ncls)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr = to_t(S[tr_idx]); ytr = torch.tensor(y[tr_idx]); Xva = to_t(S[va_idx]); yva = y[va_idx]
    lossf = nn.CrossEntropyLoss()
    t0 = time.time(); best = (0, None); ep = 0
    while time.time() - t0 < budget_s:
        perm = torch.randperm(len(Xtr)); model.train()
        for i in range(0, len(Xtr), 512):
            b = perm[i:i + 512]
            loss = lossf(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if time.time() - t0 > budget_s:
                break
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xva[i:i + 4096]).argmax(1) for i in range(0, len(Xva), 4096)]).numpy()
        ba = balanced_accuracy_score(yva, pv)
        run.log({'seqclf/val_bal_acc': ba, 'seqclf/epoch': ep})
        log(f'  SeqClf ep{ep} val bal_acc={ba:.3f}', 'analysis.log')
        if ba > best[0]:
            best = (ba, {k: v.clone() for k, v in model.state_dict().items()})
        ep += 1
    model.load_state_dict(best[1])
    return model


@torch.no_grad()
def embed(model_fn, S, bs=4096):
    out = []
    for i in range(0, len(S), bs):
        out.append(model_fn(to_t(S[i:i + bs])).numpy())
    return np.concatenate(out)


def main():
    run = wandb_init('seq-autoencoder' + SFX, 'autoencoder', job_type='train',
                     config=dict(L=64, d_in=16, z=ZD, hidden=128, arch='GRU2-enc/GRU1-dec'))
    import wandb
    M, S = load()
    classes = sorted(M.fine.unique()); cmap = {c: i for i, c in enumerate(classes)}
    y = M.fine.map(cmap).values
    per_tr, per_va = (2000, 500) if os.environ.get('SMOKE') else (25000, 6000)
    tr = balanced_idx(M, per_tr, False); va = balanced_idx(M, per_va, True, seed=1)
    log(f'AE: {len(M)} seqs, train {len(tr)} test {len(va)}', 'analysis.log')
    ae, vl, base = train_ae(S, tr, va, run, budget_s=int(os.environ.get('AE_BUDGET', 900)))
    torch.save(ae.state_dict(), f'{ROOT}/models_ae{SFX}.pt')
    Ztr = embed(ae.encode, S[tr]); Zva = embed(ae.encode, S[va])
    res = {'ae_val_mse': vl, 'ae_mean_baseline_mse': base}
    # probes
    for lab in ('coarse', 'fine'):
        cl = sorted(M[lab].unique()); yy = M[lab].map({c: i for i, c in enumerate(cl)}).values
        mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
        lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit((Ztr - mu) / sd, yy[tr])
        p = lr.predict((Zva - mu) / sd)
        r = dict(tag=f'ae_linear_probe_{lab}', classes=cl, bal_acc=float(balanced_accuracy_score(yy[va], p)),
                 macro_f1=float(f1_score(yy[va], p, average='macro')), chance=1 / len(cl),
                 cm=confusion_matrix(yy[va], p, labels=range(len(cl)), normalize='true').round(3).tolist())
        res[r['tag']] = r
        plot_cm(r, f'{REP}/cm_linear_probe_{lab}.png', f'Linear probe on 16-d AE embedding ({lab})')
        import lightgbm as lgb
        m = lgb.train(dict(objective='multiclass', num_class=len(cl), learning_rate=.05, num_leaves=31, verbose=-1,
                           num_threads=8, class_weight=None), lgb.Dataset(Ztr, yy[tr]), 300)
        p = m.predict(Zva).argmax(1)
        r = dict(tag=f'ae_gbm_probe_{lab}', classes=cl, bal_acc=float(balanced_accuracy_score(yy[va], p)),
                 macro_f1=float(f1_score(yy[va], p, average='macro')), chance=1 / len(cl),
                 cm=confusion_matrix(yy[va], p, labels=range(len(cl)), normalize='true').round(3).tolist())
        res[r['tag']] = r
        plot_cm(r, f'{REP}/cm_gbm_probe_{lab}.png', f'GBM probe on 16-d AE embedding ({lab})')
        run.log({f'probe/{lab}_linear_bal_acc': res[f'ae_linear_probe_{lab}']['bal_acc'],
                 f'probe/{lab}_gbm_bal_acc': r['bal_acc']})
        log(f"  probe {lab}: linear {res[f'ae_linear_probe_{lab}']['bal_acc']:.3f} gbm {r['bal_acc']:.3f}", 'analysis.log')
    # PCA scatter
    Zc = Zva - Ztr.mean(0)
    U, s_, Vt = np.linalg.svd((Ztr - Ztr.mean(0)) / (Ztr.std(0) + 1e-6), full_matrices=False)
    P2 = ((Zva - Ztr.mean(0)) / (Ztr.std(0) + 1e-6)) @ Vt[:2].T
    col = {'crypto': '#2a6fdb', 'sports': '#e07b39', 'perp': '#3aa17e'}
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    co = M.coarse.values[va]
    for c in col:
        k = co == c
        axs[0].scatter(P2[k, 0], P2[k, 1], s=2, alpha=.25, color=col[c], label=c)
    axs[0].legend(markerscale=6); axs[0].set_title('AE embedding (PCA of 16-d latent), coloured by coarse class')
    fn = M.fine.values[va]
    cm_ = plt.get_cmap('tab20')
    for i, c in enumerate(classes):
        k = fn == c
        axs[1].scatter(P2[k, 0], P2[k, 1], s=2, alpha=.25, color=cm_(i), label=c)
    axs[1].legend(markerscale=6, fontsize=6, ncol=2); axs[1].set_title('...coloured by fine class')
    fig.tight_layout(); fig.savefig(f'{REP}/embedding_pca.png'); plt.close(fig)
    # interpret latent dims via correlation with handcrafted N64 features
    Fva = M.loc[va, [f'N64__{x}' for x in FEATS]].values
    interp = {}
    for d in range(ZD):
        cors = []
        for j, f in enumerate(FEATS):
            a = Zva[:, d]; b = Fva[:, j]; ok = np.isfinite(b)
            cors.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 and b[ok].std() > 0 else 0)
        cors = np.nan_to_num(np.array(cors))
        top = np.argsort(-np.abs(cors))[:4]
        interp[f'z{d}'] = [(FEATS[j], round(float(cors[j]), 3)) for j in top]
    res['latent_interpretation'] = interp
    # class means of standardized latent dims
    Zs = (Zva - Ztr.mean(0)) / (Ztr.std(0) + 1e-6)
    cm_lat = pd.DataFrame(Zs).groupby(fn).mean()
    cm_lat.columns = [f'z{d}' for d in range(ZD)]
    res['latent_class_means'] = cm_lat.round(3).to_dict(orient='index')
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(cm_lat.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    ax.set_yticks(range(len(cm_lat))); ax.set_yticklabels(cm_lat.index)
    ax.set_xticks(range(ZD)); ax.set_xticklabels([f"z{d}\n{interp[f'z{d}'][0][0]}" for d in range(ZD)], fontsize=6, rotation=90)
    ax.grid(False); plt.colorbar(im, ax=ax, fraction=.02)
    ax.set_title('Mean latent value per market (dims labelled by most-correlated handcrafted feature)')
    fig.tight_layout(); fig.savefig(f'{REP}/latent_class_means.png'); plt.close(fig)
    if STRICT_AE:
        json.dump(res, open(f'{REP}/summary.json', 'w'), indent=1, default=float)
        for pth in glob.glob(f'{REP}/*.png'):
            wandb.log({'ae_strict/' + os.path.basename(pth)[:-4]: wandb.Image(pth)})
        run.finish(); log('AE strict done', 'analysis.log'); return
    # supervised sequence classifier (provides activations for trade model)
    clf = train_clf(S, y, tr, va, len(classes), run, budget_s=int(os.environ.get('CLF_BUDGET', 600)))
    with torch.no_grad():
        p = embed(lambda x: clf(x).argmax(1, keepdim=True).float(), S[va])[:, 0].astype(int)
    r = dict(tag='seq_gru_classifier_fine', classes=classes, bal_acc=float(balanced_accuracy_score(y[va], p)),
             macro_f1=float(f1_score(y[va], p, average='macro')), chance=1 / len(classes),
             cm=confusion_matrix(y[va], p, labels=range(len(classes)), normalize='true').round(3).tolist())
    res[r['tag']] = r
    plot_cm(r, f'{REP}/cm_seq_gru_classifier.png', 'Supervised GRU on raw 64-event sequences (fine)')
    torch.save({'state': clf.state_dict(), 'classes': classes}, f'{ROOT}/models_seqclf.pt')
    run.log({'seqclf/test_bal_acc': r['bal_acc']})
    # embeddings + activations for trade sequences
    fs = sorted(glob.glob(f'{DATA}/feat/trade/*_seq.npy'))
    for f in fs:
        fine = pd.read_parquet(f.replace('_seq.npy', '.parquet'), columns=['fine']).fine.values
        keep = np.nonzero(trade_keep(fine))[0]
        TS = np.load(f, mmap_mode='r')[keep]
        Z = embed(ae.encode, TS); A = embed(clf.act, TS)
        P = embed(lambda x: torch.softmax(clf(x), 1), TS)
        np.savez(f.replace('_seq.npy', '_emb.npz'), z=Z.astype(np.float32), act=A.astype(np.float32),
                 prob=P.astype(np.float32), keep=keep)
        log(f'  embedded {len(keep)} trades from {os.path.basename(f)}', 'analysis.log')
    json.dump(res, open(f'{REP}/summary.json', 'w'), indent=1, default=float)
    for pth in glob.glob(f'{REP}/*.png'):
        wandb.log({'ae/' + os.path.basename(pth)[:-4]: wandb.Image(pth)})
    run.summary.update({k: v['bal_acc'] for k, v in res.items() if isinstance(v, dict) and 'bal_acc' in v})
    run.finish()
    log('AE done', 'analysis.log')


if __name__ == '__main__':
    main()
