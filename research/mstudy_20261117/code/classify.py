"""Part 1b: can scale-free microstructure features identify the market type?
LightGBM sweeps over history windows + feature-group ablations; MLP classifier (activations reused later)."""
import sys, os, json, glob, time, hashlib
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
from common import *
from features import FEATS, GROUPS

REP = f'{ROOT}/report/classify'
os.makedirs(REP, exist_ok=True)
NONTRIV = [f for f in FEATS if not f.startswith('tv_')]
# features whose value depends on the exchange price grid (tick size) rather than on behaviour
TICKGRID = ['sp_mean', 'sp_one', 'sp_std', 'd_gap1', 'pm_jump', 'ev_deep', 'ev_inside']
STRICT = [f for f in NONTRIV if f not in TICKGRID]
FINE_KEEP = ['BTC15M', 'alt15M', 'cryptoHourly', 'cryptoRange', 'NFL', 'NCAAF', 'MLB', 'SOCCER', 'WNBA', 'TENNIS',
             'BTCPERP', 'ETHPERP', 'altPERP']


def split_mask(df, frac=0.2, salt='v1'):
    key = (df.ticker.astype(str) + '|' + df.day.astype(str) + salt).values
    h = np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for k in key])
    return h < frac


def load():
    fs = sorted(glob.glob(f'{DATA}/feat/win/*.parquet'))
    W = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    W = W[W.fine.isin(FINE_KEEP)].reset_index(drop=True)
    W['test'] = split_mask(W)
    # balance: cap anchors per fine class (keeps fits tractable on ~12 free cores)
    cap = int(os.environ.get('CLS_CAP', 20000))
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(ix, min(len(ix), cap), replace=False) for ix in W.groupby('fine').indices.values()])
    W = W.iloc[np.sort(pick)].reset_index(drop=True)
    if os.environ.get('SMOKE'):
        W = W.sample(min(len(W), 20000), random_state=0).reset_index(drop=True)
    return W


def fit_eval(W, cols, label, tag, rows=None, rounds=400, ret_model=False, threads=10):
    if os.environ.get('SMOKE'):
        rounds = 20
    D = W if rows is None else W[rows]
    D = D[D[cols].notna().mean(1) > 0.5]
    classes = sorted(D[label].unique())
    if len(classes) < 2 or len(D) < 200 or D.test.sum() < 20 or (~D.test).sum() < 50:
        stub = dict(tag=tag, label=label, n_train=0, n_test=0, classes=classes, bal_acc=float('nan'),
                    macro_f1=float('nan'), chance=1.0 / max(len(classes), 1), cm=[], skipped=True)
        return (stub, None) if ret_model else stub
    y = D[label].map({c: i for i, c in enumerate(classes)}).values
    X = D[cols].values.astype(np.float32)
    te = D.test.values
    cw = pd.Series(y[~te]).value_counts()
    w = (1.0 / cw.reindex(y[~te]).values)
    w = w / w.mean()
    params = dict(objective='multiclass', num_class=len(classes), learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=threads, verbose=-1)
    dtr = lgb.Dataset(X[~te], y[~te], weight=w)
    m = lgb.train(params, dtr, num_boost_round=rounds)
    p = m.predict(X[te])
    pred = p.argmax(1)
    res = dict(tag=tag, label=label, n_train=int((~te).sum()), n_test=int(te.sum()), classes=classes,
               bal_acc=float(balanced_accuracy_score(y[te], pred)), macro_f1=float(f1_score(y[te], pred, average='macro')),
               chance=1.0 / len(classes),
               cm=confusion_matrix(y[te], pred, labels=range(len(classes)), normalize='true').round(3).tolist())
    if ret_model:
        return res, m
    return res


def plot_cm(res, path, title):
    cm = np.array(res['cm']); cl = res['classes']
    fig, ax = plt.subplots(figsize=(1 + .55 * len(cl), .8 + .5 * len(cl)))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(cl)):
        for j in range(len(cl)):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if cm[i, j] > .5 else 'k')
    ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, rotation=60, fontsize=7)
    ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl, fontsize=7); ax.grid(False)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(f"{title}\nbal.acc={res['bal_acc']:.3f} (chance {res['chance']:.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def mlp_classifier(W, run):
    """MLP on N256 non-trivial features -> fine label. Saves model for trade-model activations."""
    import torch, torch.nn as nn
    torch.set_num_threads(8)
    cols = [f'N256__{f}' for f in NONTRIV]
    D = W[W[cols].notna().mean(1) > 0.5].copy()
    classes = sorted(D.fine.unique())
    y = D.fine.map({c: i for i, c in enumerate(classes)}).values
    X = D[cols].values.astype(np.float32)
    te = D.test.values
    med = np.nanmedian(X[~te], 0)
    X = np.where(np.isnan(X), med, X)
    lo, hi = np.percentile(X[~te], 0.5, 0), np.percentile(X[~te], 99.5, 0)
    X = np.clip(X, lo, hi)
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    X = (X - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 128), nn.GELU(),
                        nn.Dropout(.1), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, len(classes)))
    cw = np.bincount(y[~te], minlength=len(classes)).astype(np.float32)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw.sum() / (cw + 1) / len(classes)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(X[~te]); yt = torch.tensor(y[~te]); Xv = torch.tensor(X[te]); yv = y[te]
    best = (0, None)
    for ep in range(3 if os.environ.get('SMOKE') else 40):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad(); l = lossf(net(Xt[b]), yt[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv).argmax(1).numpy()
        ba = balanced_accuracy_score(yv, pv)
        run.log({'mlp/epoch': ep, 'mlp/val_bal_acc': ba, 'mlp/train_loss': float(l)})
        if ba > best[0]:
            best = (ba, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        pv = net(Xv).argmax(1).numpy()
    res = dict(tag='mlp_N256_fine', label='fine', classes=classes, bal_acc=float(balanced_accuracy_score(yv, pv)),
               macro_f1=float(f1_score(yv, pv, average='macro')), chance=1 / len(classes),
               cm=confusion_matrix(yv, pv, labels=range(len(classes)), normalize='true').round(3).tolist())
    torch.save({'state': net.state_dict(), 'mu': mu, 'sd': sd, 'lo': lo, 'hi': hi, 'med': med, 'cols': NONTRIV,
                'classes': classes}, f'{ROOT}/models_clf_mlp.pt')
    return res


def main():
    run = wandb_init('market-type-classifier', 'classify', job_type='analysis')
    import wandb
    W = load()
    log(f'classify: {len(W)} anchors; test frac {W.test.mean():.3f}', 'analysis.log')
    results = []
    blocks = ['N16', 'N64', 'N256', 'N1024', 'W1', 'W10', 'W60', 'W600']
    # E1: window sweep, coarse + fine
    for fsname, fset in (('nontrivial', NONTRIV), ('strict', STRICT)):
        for label in ('coarse', 'fine'):
            for b in blocks:
                r = fit_eval(W, [f'{b}__{f}' for f in fset], label, f'window_{b}' if fsname == 'nontrivial' else f'strict_{b}')
                r['window'] = b; r['features'] = fsname
                r['coverage'] = float(W[f'{b}__tm_cv'].notna().mean())
                results.append(r); run.log({f'sweep/{fsname}/{label}/{b}_bal_acc': r['bal_acc']})
                log(f"  {fsname} {label} {b}: bal_acc={r['bal_acc']:.3f} cov={r['coverage']:.2f}", 'analysis.log')
    # with trivial (to show what we are excluding)
    for label in ('coarse', 'fine'):
        r = fit_eval(W, [f'N256__{f}' for f in FEATS], label, 'N256_with_trivial'); r['window'] = 'N256'
        r['features'] = 'all_incl_trivial'; results.append(r)
        r = fit_eval(W, [f'N256__{f}' for f in GROUPS['trivial']], label, 'N256_trivial_only'); r['window'] = 'N256'
        r['features'] = 'trivial_only'; results.append(r)
    # E2: ablations at N256 and N1024 (fine label)
    for b in ('N256', 'N1024'):
        for g, fs in GROUPS.items():
            if g == 'trivial':
                continue
            keep = [f for f in NONTRIV if f not in fs]
            r = fit_eval(W, [f'{b}__{f}' for f in keep], 'fine', f'{b}_drop_{g}'); r['window'] = b; r['features'] = f'drop_{g}'
            results.append(r)
            r = fit_eval(W, [f'{b}__{f}' for f in fs], 'fine', f'{b}_only_{g}'); r['window'] = b; r['features'] = f'only_{g}'
            results.append(r)
        # drop imbalance + spread (the "obvious" ones)
        r = fit_eval(W, [f'{b}__{f}' for f in STRICT], 'fine', f'{b}_strict'); r['window'] = b; r['features'] = 'strict (drop tick-grid)'
        results.append(r)
        keep = [f for f in NONTRIV if f not in GROUPS['imbalance'] + GROUPS['spread']]
        r = fit_eval(W, [f'{b}__{f}' for f in keep], 'fine', f'{b}_drop_imb_spread'); r['window'] = b
        r['features'] = 'drop_imbalance+spread'; results.append(r)
    # E3: multi-window
    cols = [f'{b}__{f}' for b in ('N16', 'N64', 'N256', 'N1024') for f in NONTRIV]
    rC, mC = fit_eval(W, cols, 'coarse', 'multiwindow', ret_model=True); rC['window'] = 'N16-1024'; rC['features'] = 'nontrivial'
    rF, mF = fit_eval(W, cols, 'fine', 'multiwindow', ret_model=True); rF['window'] = 'N16-1024'; rF['features'] = 'nontrivial'
    results += [rC, rF]
    cols_s = [f'{b}__{f}' for b in ('N16', 'N64', 'N256', 'N1024') for f in STRICT]
    rCs = fit_eval(W, cols_s, 'coarse', 'multiwindow_strict'); rCs['window'] = 'N16-1024'; rCs['features'] = 'strict'
    rFs, mFs = fit_eval(W, cols_s, 'fine', 'multiwindow_strict', ret_model=True); rFs['window'] = 'N16-1024'; rFs['features'] = 'strict'
    results += [rCs, rFs]
    if not rCs.get('skipped'):
        plot_cm(rCs, f'{REP}/cm_multiwindow_coarse_strict.png', 'Coarse class, STRICT features (no tick-grid, no scale)')
    if not rFs.get('skipped'):
        plot_cm(rFs, f'{REP}/cm_multiwindow_fine_strict.png', 'Fine class, STRICT features (no tick-grid, no scale)')
    imp = pd.Series(mF.feature_importance('gain'), index=cols)
    imp_f = imp.groupby(lambda c: c.split('__')[1]).sum().sort_values(ascending=False)
    imp_g = imp.groupby(lambda c: [g for g, fs in GROUPS.items() if c.split('__')[1] in fs][0]).sum()
    imp_w = imp.groupby(lambda c: c.split('__')[0]).sum()
    # E4: within-coarse discrimination
    for co in ('sports', 'crypto', 'perp'):
        r = fit_eval(W, [f'N1024__{f}' for f in NONTRIV], 'fine', f'within_{co}', rows=(W.coarse == co).values)
        r['window'] = 'N1024'; r['features'] = f'within_{co}'; results.append(r)
        if not r.get('skipped'):
            plot_cm(r, f'{REP}/cm_within_{co}.png', f'Within {co}: which market? (N=1024 events)')
    # crypto 15M BTC vs sports only: binary
    sub = W.fine.isin(['BTC15M']) | (W.coarse == 'sports')
    W['btc_vs_sport'] = np.where(W.fine == 'BTC15M', 'BTC15M', 'sports')
    for b in ('N16', 'N64', 'N256', 'N1024'):
        r = fit_eval(W, [f'{b}__{f}' for f in NONTRIV], 'btc_vs_sport', f'btc15m_vs_sports_{b}', rows=sub.values)
        r['window'] = b; r['features'] = 'nontrivial'; results.append(r)
    # MLP
    r = mlp_classifier(W, run); r['window'] = 'N256'; r['features'] = 'nontrivial'; results.append(r)
    plot_cm(r, f'{REP}/cm_mlp_N256_fine.png', 'MLP classifier, fine market type (N=256)')
    # plots
    plot_cm(rC, f'{REP}/cm_multiwindow_coarse.png', 'Crypto-binary vs sports vs perp (multi-window LightGBM)')
    plot_cm(rF, f'{REP}/cm_multiwindow_fine.png', 'Fine market type (multi-window LightGBM)')
    R = pd.DataFrame([{k: v for k, v in r.items() if k not in ('cm', 'classes')} for r in results])
    R.to_csv(f'{REP}/results.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label in zip(axs, ('coarse', 'fine')):
        s = R[(R.label == label) & R.tag.str.startswith('window_')]
        s2 = R[(R.label == label) & R.tag.str.startswith('strict_')]
        x = np.arange(len(s))
        ax.bar(x - .2, s.bal_acc.values, .4, color='#2a6fdb', label='non-trivial features')
        ax.bar(x + .2, s2.bal_acc.values, .4, color='#e07b39', label='strict (no tick-grid)')
        ax.set_xticks(x); ax.set_xticklabels(s.window.values)
        ax.axhline(s.chance.iloc[0], color='k', ls='--', lw=.8, label='chance')
        for i, (v, c) in enumerate(zip(s.bal_acc, s.coverage)):
            ax.text(i - .2, v + .01, f'{v:.2f}', ha='center', fontsize=6)
        for i, v in enumerate(s2.bal_acc):
            ax.text(i + .2, v + .01, f'{v:.2f}', ha='center', fontsize=6)
        ax.set_ylim(0, 1.08); ax.set_title(f'{label} label: balanced accuracy vs history window'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{REP}/window_sweep.png'); plt.close(fig)
    abl = R[R.tag.str.startswith('N256_drop') | R.tag.str.startswith('N256_only')]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(abl.features, abl.bal_acc, color=['#c44' if 'drop' in f else '#36c' for f in abl.features])
    base = R[(R.tag == 'window_N256') & (R.label == 'fine')].bal_acc.iloc[0]
    ax.axvline(base, color='k', ls='--', lw=.8, label=f'all non-trivial ({base:.2f})'); ax.legend()
    ax.set_title('Fine-label ablations at N=256 (red: drop group, blue: only group)')
    fig.tight_layout(); fig.savefig(f'{REP}/ablations.png'); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    imp_f.head(25)[::-1].plot.barh(ax=axs[0]); axs[0].set_title('Top features (gain, fine multi-window)')
    imp_g.sort_values().plot.barh(ax=axs[1]); axs[1].set_title('Importance by feature group')
    imp_w.reindex(['N16', 'N64', 'N256', 'N1024']).plot.bar(ax=axs[2]); axs[2].set_title('Importance by window')
    fig.tight_layout(); fig.savefig(f'{REP}/importance.png'); plt.close(fig)
    out = dict(results=results, importance_feature=imp_f.round(1).to_dict(), importance_group=imp_g.round(1).to_dict(),
               importance_window=imp_w.round(1).to_dict(),
               n_anchors=int(len(W)), by_fine=W.fine.value_counts().to_dict())
    json.dump(out, open(f'{REP}/summary.json', 'w'), indent=1, default=float)
    for p in glob.glob(f'{REP}/*.png'):
        wandb.log({'classify/' + os.path.basename(p)[:-4]: wandb.Image(p)})
    wandb.log({'classify/results': wandb.Table(dataframe=R)})
    run.finish()
    log('classify done', 'analysis.log')


if __name__ == '__main__':
    main()
