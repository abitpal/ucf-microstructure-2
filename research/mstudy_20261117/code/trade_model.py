"""Part 1d: can microstructure state + the incoming taker order predict that order's PnL / toxicity?
Multi-task MLP over 4 horizons, feature-set ablations, LightGBM baseline, confusion matrices."""
import sys, os, json, glob, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, average_precision_score
from common import *
from features import FEATS
from classify import split_mask, FINE_KEEP

torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
REP = f'{ROOT}/report/trade'
os.makedirs(REP, exist_ok=True)
HS = ['0p1', '0p5', '1p0', '10p0']
NONTRIV = [f for f in FEATS if not f.startswith('tv_')]


def load():
    fs = sorted(f for f in glob.glob(f'{DATA}/feat/trade/*.parquet'))
    T, Z, A, P = [], [], [], []
    for f in fs:
        e = f.replace('.parquet', '_emb.npz')
        if not os.path.exists(e):
            continue
        d = pd.read_parquet(f); em = np.load(e)
        d = d.iloc[em['keep']].reset_index(drop=True)
        T.append(d); Z.append(em['z']); A.append(em['act']); P.append(em['prob'])
    T = pd.concat(T, ignore_index=True)
    Z = np.concatenate(Z); A = np.concatenate(A); P = np.concatenate(P)
    keep = T.fine.isin(FINE_KEEP).values
    T = T[keep].reset_index(drop=True); Z, A, P = Z[keep], A[keep], P[keep]
    if os.environ.get('SMOKE'):
        k = np.random.default_rng(0).random(len(T)) < 0.05
        T, Z, A, P = T[k].reset_index(drop=True), Z[k], A[k], P[k]
    T['test'] = split_mask(T)
    T['val'] = split_mask(T, 0.15, salt='val') & ~T.test
    return T, Z, A, P


def build_sets(T, Z, A, P):
    trade = [c for c in T.columns if c.startswith('trf_')]
    win = [f'w__{f}' for f in NONTRIV]
    onehot = pd.get_dummies(T.coarse).astype(np.float32)
    base = T[trade].values.astype(np.float32)
    W = T[win].values.astype(np.float32)
    C = onehot.values
    sets = {
        'trade_only': np.c_[base, C],
        'trade+window': np.c_[base, W, C],
        'trade+window+AE': np.c_[base, W, Z, C],
        'trade+window+AE+clf': np.c_[base, W, Z, A, P, C],
        'trade+AE+clf (no handcrafted window)': np.c_[base, Z, A, P, C],
    }
    names = {
        'trade+window+AE+clf': trade + win + [f'ae_z{i}' for i in range(Z.shape[1])] + [f'clf_a{i}' for i in range(A.shape[1])]
                               + [f'clf_p{i}' for i in range(P.shape[1])] + list(onehot.columns)}
    return sets, names


def targets(T):
    # markouts in spread units (per fine class median spread), clipped
    spr = T.groupby('fine').trf_spread.transform('median').clip(lower=1).values
    R = np.stack([np.clip(T[f'y_mk_{h}'].values / spr, -10, 10) for h in HS], 1).astype(np.float32)
    TOX = np.stack([(T[f'y_mk_{h}'].values > 0) for h in HS], 1).astype(np.float32)
    NET = np.stack([(T[f'y_net_{h}'].values > 0) for h in HS], 1).astype(np.float32)
    return R, TOX, NET


def prep_X(X, tr):
    X = X.copy()
    med = np.nanmedian(X[tr], 0)
    X = np.where(np.isfinite(X), X, med)
    lo, hi = np.percentile(X[tr], 0.5, 0), np.percentile(X[tr], 99.5, 0)
    X = np.clip(X, lo, hi)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


class MT(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(.15), nn.Linear(h, h), nn.GELU(),
                                   nn.Dropout(.15), nn.Linear(h, 128), nn.GELU())
        self.reg = nn.Linear(128, 4); self.tox = nn.Linear(128, 4); self.net = nn.Linear(128, 4)

    def forward(self, x):
        z = self.trunk(x)
        return self.reg(z), self.tox(z), self.net(z)


def train_mt(X, R, TOX, NET, tr, va, run, tag, budget_s=420):
    m = MT(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    pw = torch.tensor(np.clip((1 - NET[tr].mean(0)) / (NET[tr].mean(0) + 1e-4), 1, 50), dtype=torch.float32)
    bce_t = nn.BCEWithLogitsLoss(); bce_n = nn.BCEWithLogitsLoss(pos_weight=pw); hub = nn.HuberLoss(delta=1.0)
    Xt = torch.tensor(X[tr]); Rt = torch.tensor(R[tr]); Tt = torch.tensor(TOX[tr]); Nt = torch.tensor(NET[tr])
    Xv = torch.tensor(X[va])
    t0 = time.time(); best = (-1, None); ep = 0; bad = 0
    while time.time() - t0 < budget_s and bad < 6:
        perm = torch.randperm(len(Xt)); m.train()
        for i in range(0, len(Xt), 2048):
            b = perm[i:i + 2048]
            r, t, n = m(Xt[b])
            loss = hub(r, Rt[b]) + bce_t(t, Tt[b]) + bce_n(n, Nt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            r, t, n = m(Xv)
        aucs = [roc_auc_score(TOX[va][:, k], t[:, k].numpy()) for k in range(4)]
        score = float(np.mean(aucs))
        run.log({f'{tag}/val_tox_auc_mean': score, f'{tag}/loss': float(loss), f'{tag}/epoch': ep})
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in m.state_dict().items()}); bad = 0
        else:
            bad += 1
        ep += 1
    m.load_state_dict(best[1]); m.eval()
    return m


def evaluate(pred_r, pred_t, pred_n, R, TOX, NET, idx, T):
    out = {}
    for k, h in enumerate(HS):
        y = TOX[idx, k]; yn = NET[idx, k]
        out[f'tox_auc_{h}'] = float(roc_auc_score(y, pred_t[:, k])) if 0 < y.mean() < 1 else np.nan
        out[f'net_auc_{h}'] = float(roc_auc_score(yn, pred_n[:, k])) if 0 < yn.mean() < 1 else np.nan
        out[f'net_ap_{h}'] = float(average_precision_score(yn, pred_n[:, k])) if 0 < yn.mean() < 1 else np.nan
        out[f'net_base_{h}'] = float(yn.mean()); out[f'tox_base_{h}'] = float(y.mean())
        c = np.corrcoef(pred_r[:, k], R[idx, k])[0, 1]
        out[f'mk_corr_{h}'] = float(c)
        ss = ((R[idx, k] - pred_r[:, k]) ** 2).sum(); st = ((R[idx, k] - R[idx, k].mean()) ** 2).sum()
        out[f'mk_r2_{h}'] = float(1 - ss / st)
        # economic: top decile by predicted net prob
        q = np.quantile(pred_n[:, k], 0.9)
        top = pred_n[:, k] >= q
        net_ticks = T[f'y_net_{h}'].values[idx]
        out[f'top10_net_hit_{h}'] = float(yn[top].mean())
        out[f'top10_mean_net_ticks_{h}'] = float(np.mean(net_ticks[top]))
        out[f'all_mean_net_ticks_{h}'] = float(np.mean(net_ticks))
        q99 = np.quantile(pred_n[:, k], 0.99); t1 = pred_n[:, k] >= q99
        out[f'top1_net_hit_{h}'] = float(yn[t1].mean())
        out[f'top1_mean_net_ticks_{h}'] = float(np.mean(net_ticks[t1]))
    return out


def cm_plot(y, p, thr, path, title):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    lab = ['benign', 'toxic'] if 'oxic' in title else ['loses', 'profits']
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cmn[i, j]:.2f}\n(n={cm[i, j]:,})', ha='center', va='center', fontsize=8,
                    color='w' if cmn[i, j] > .5 else 'k')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['pred ' + l for l in lab]); ax.set_yticks([0, 1]); ax.set_yticklabels(['true ' + l for l in lab])
    ax.grid(False); ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return cm.tolist()


def best_thr(y, p):
    qs = np.quantile(p, np.linspace(0.02, 0.98, 97))
    ba = [balanced_accuracy_score(y, (p >= q).astype(int)) for q in qs]
    return float(qs[int(np.argmax(ba))])


def main():
    run = wandb_init('trade-toxicity-pnl', 'trade-model', job_type='train')
    import wandb
    T, Z, A, P = load()
    R, TOX, NET = targets(T)
    te = T.test.values; va = T.val.values; tr = ~te & ~va
    log(f'trade model: {len(T)} orders train {tr.sum()} val {va.sum()} test {te.sum()}', 'analysis.log')
    sets, names = build_sets(T, Z, A, P)
    results = {}; preds = {}
    for name, X in sets.items():
        Xs = prep_X(X, tr)
        m = train_mt(Xs, R, TOX, NET, tr, va, run, name.replace(' ', '_'), budget_s=int(os.environ.get('MT_BUDGET', 360)))
        with torch.no_grad():
            r, t, n = m(torch.tensor(Xs[te])); rv, tv, nv = m(torch.tensor(Xs[va]))
        pr, pt, pn = r.numpy(), torch.sigmoid(t).numpy(), torch.sigmoid(n).numpy()
        res = evaluate(pr, pt, pn, R, TOX, NET, te, T)
        results[f'MLP | {name}'] = res
        preds[name] = (pr, pt, pn, torch.sigmoid(tv).numpy(), torch.sigmoid(nv).numpy())
        run.log({f'test/{name}/{k}': v for k, v in res.items()})
        log(f"  {name}: tox AUC {[round(res[f'tox_auc_{h}'],3) for h in HS]} net AUC {[round(res[f'net_auc_{h}'],3) for h in HS]}", 'analysis.log')
    # LightGBM baseline on full set, per target
    full = 'trade+window+AE+clf'
    X = sets[full]
    gb = {}
    imps = {}
    for k, h in enumerate(HS):
        for kind, Y in (('tox', TOX), ('net', NET)):
            d = lgb.Dataset(X[tr], Y[tr, k]); dv = lgb.Dataset(X[va], Y[va, k])
            m = lgb.train(dict(objective='binary', learning_rate=.05, num_leaves=63, min_data_in_leaf=100,
                               feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, num_threads=10, verbose=-1,
                               scale_pos_weight=1.0), d, 30 if os.environ.get('SMOKE') else 600, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            p = m.predict(X[te])
            gb[f'{kind}_{h}'] = p
            imps[f'{kind}_{h}'] = pd.Series(m.feature_importance('gain'), index=names[full])
    resg = {}
    for k, h in enumerate(HS):
        y = TOX[te, k]; yn = NET[te, k]
        resg[f'tox_auc_{h}'] = float(roc_auc_score(y, gb[f'tox_{h}']))
        resg[f'net_auc_{h}'] = float(roc_auc_score(yn, gb[f'net_{h}'])) if 0 < yn.mean() < 1 else np.nan
        q = np.quantile(gb[f'net_{h}'], .9); top = gb[f'net_{h}'] >= q
        resg[f'top10_net_hit_{h}'] = float(yn[top].mean()); resg[f'top10_mean_net_ticks_{h}'] = float(T[f'y_net_{h}'].values[te][top].mean())
        resg[f'net_base_{h}'] = float(yn.mean()); resg[f'tox_base_{h}'] = float(y.mean())
    results['LightGBM | trade+window+AE+clf'] = resg
    log(f"  LGBM: tox AUC {[round(resg[f'tox_auc_{h}'],3) for h in HS]}", 'analysis.log')
    # per-coarse AUC for best MLP
    pr, pt, pn, tv, nv = preds[full]
    per = {}
    for co in sorted(T.coarse.unique()):
        k_ = (T.coarse.values[te] == co)
        per[co] = {}
        for k, h in enumerate(HS):
            y = TOX[te][k_, k]; yn = NET[te][k_, k]
            per[co][f'tox_auc_{h}'] = float(roc_auc_score(y, pt[k_, k])) if 0 < y.mean() < 1 else None
            per[co][f'net_auc_{h}'] = float(roc_auc_score(yn, pn[k_, k])) if 0 < yn.mean() < 1 else None
            per[co][f'tox_base_{h}'] = float(y.mean()); per[co][f'net_base_{h}'] = float(yn.mean())
            per[co][f'n'] = int(k_.sum())
    # per fine
    perf = {}
    for fi in sorted(T.fine.unique()):
        k_ = (T.fine.values[te] == fi)
        if k_.sum() < 300:
            continue
        perf[fi] = {f'tox_auc_{h}': (float(roc_auc_score(TOX[te][k_, k], pt[k_, k])) if 0 < TOX[te][k_, k].mean() < 1 else None)
                    for k, h in enumerate(HS)}
        perf[fi]['n'] = int(k_.sum())
    # confusion matrices (threshold chosen on validation for balanced accuracy)
    cms = {}
    for k, h in enumerate(HS):
        thr = best_thr(TOX[va, k], tv[:, k])
        cms[f'toxic_{h}'] = dict(thr=thr, cm=cm_plot(TOX[te, k], pt[:, k], thr, f'{REP}/cm_toxic_{h}.png',
                                                      f'Toxic flow at {h.replace("p", ".")}s (MLP full)'),
                                 bal_acc=float(balanced_accuracy_score(TOX[te, k], (pt[:, k] >= thr).astype(int))))
        if 0 < NET[va, k].mean() < 1:
            thr = best_thr(NET[va, k], nv[:, k])
            cms[f'net_{h}'] = dict(thr=thr, cm=cm_plot(NET[te, k], pn[:, k], thr, f'{REP}/cm_net_{h}.png',
                                                        f'Profitable after 2 taker fees, {h.replace("p", ".")}s'),
                                   bal_acc=float(balanced_accuracy_score(NET[te, k], (pn[:, k] >= thr).astype(int))))
    # AUC comparison chart
    Rdf = pd.DataFrame(results).T
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, kind in zip(axs, ('tox', 'net')):
        for name, row in Rdf.iterrows():
            ax.plot([0.1, 0.5, 1, 10], [row[f'{kind}_auc_{h}'] for h in HS], marker='o', label=name)
        ax.set_xscale('log'); ax.axhline(.5, color='k', lw=.6); ax.set_xlabel('horizon (s)'); ax.set_ylabel('test ROC-AUC')
        ax.set_title('Toxic-flow AUC (mid moves past fill)' if kind == 'tox' else 'Profitable-after-fees AUC')
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{REP}/auc_by_featureset.png'); plt.close(fig)
    # importance (toxic 1s)
    imp = imps['tox_1p0'].sort_values(ascending=False)
    grp = imp.groupby(lambda c: 'trade' if c.startswith('trf_') else 'window' if c.startswith('w__') else
                      'AE' if c.startswith('ae_') else 'clf' if c.startswith('clf_') else 'class').sum()
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    imp.head(25)[::-1].plot.barh(ax=axs[0]); axs[0].set_title('LightGBM gain, toxic@1s: top features')
    grp.sort_values().plot.barh(ax=axs[1]); axs[1].set_title('...by input block')
    fig.tight_layout(); fig.savefig(f'{REP}/importance_toxic_1s.png'); plt.close(fig)
    # empirical interaction: size x prior move -> toxic rate
    sb = pd.qcut(T.trf_logsize_vs_hit, 5, duplicates='drop'); mb = pd.cut(T.trf_mom10, [-1e9, -2, -0.01, 0.01, 2, 1e9])
    inter = T.assign(tox=TOX[:, 2], sb=sb, mb=mb).groupby(['sb', 'mb'], observed=True).tox.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(inter.values, cmap='RdYlBu_r', aspect='auto', vmin=.2, vmax=.7)
    for i in range(inter.shape[0]):
        for j in range(inter.shape[1]):
            ax.text(j, i, f'{inter.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_yticks(range(len(inter))); ax.set_yticklabels([f'{iv.left:.1f}..{iv.right:.1f}' for iv in inter.index], fontsize=7)
    ax.set_xticks(range(inter.shape[1])); ax.set_xticklabels(['down >2t', 'down', 'flat', 'up', 'up >2t'], fontsize=7)
    ax.set_ylabel('log(order size / touch depth)'); ax.set_xlabel('mid move over prior 10s, in the order\'s direction')
    ax.grid(False); plt.colorbar(im, ax=ax, label='P(toxic @1s)')
    ax.set_title('Large orders chasing a move: toxicity (all markets)')
    fig.tight_layout(); fig.savefig(f'{REP}/size_x_momentum.png'); plt.close(fig)
    out = dict(results=results, per_coarse=per, per_fine=perf, confusion=cms,
               importance_tox1s_top=imp.head(30).round(1).to_dict(), importance_by_block=grp.round(1).to_dict(),
               size_x_momentum={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in inter.round(3).to_dict(orient='index').items()},
               n=dict(total=int(len(T)), train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
               by_fine=T.fine.value_counts().to_dict())
    json.dump(out, open(f'{REP}/summary.json', 'w'), indent=1, default=float)
    for pth in glob.glob(f'{REP}/*.png'):
        wandb.log({'trade/' + os.path.basename(pth)[:-4]: wandb.Image(pth)})
    wandb.log({'trade/results': wandb.Table(dataframe=Rdf.reset_index())})
    run.finish()
    log('trade model done', 'analysis.log')


if __name__ == '__main__':
    main()
