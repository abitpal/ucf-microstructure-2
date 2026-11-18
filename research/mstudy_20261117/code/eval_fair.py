"""Out-of-sample evaluation of the trained fair-value models on days never used in training/val/test.

usage: python eval_fair.py MARKET DAY [DAY ...]      (MARKET in BTC15M|BTCPERP)
Normalisation stats are recomputed exactly as in training (same files, same seed, train days 09-18..09-21).
LightGBM baseline is refit once with the training recipe and cached. Results -> report/fair_oos/{market}_{day}.json"""
import sys, os, json, glob, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *
import fair_models as fm

TRAIN_DAYS = ['2026-09-18', '2026-09-19', '2026-09-20', '2026-09-21', '2026-09-22', '2026-09-23']
VAL_DAY, TEST_DAY = '2026-09-22', '2026-09-23'
OUT = f'{ROOT}/report/fair_oos'
os.makedirs(OUT, exist_ok=True)
MODELS = {'BTC15M': ['transformer_cont', 'transformer_long', 'gru_long', 'gru'], 'BTCPERP': ['gru', 'transformer']}
EVAL_PER_MKT = int(os.environ.get('EVAL_PER_MKT', 500)); EVAL_PERP = int(os.environ.get('EVAL_PERP', 15000))


def load_files(files, per_mkt, perp_per_day, market, seed=0):
    """identical sampling to fair_models.load but over an explicit file list"""
    rng = np.random.default_rng(seed)
    S, B, Y, X, D = [], [], [], [], []
    for f in files:
        day = f.split('_')[-1][:-4]
        z = np.load(f, allow_pickle=True); mk = z['mkt']
        if market == 'BTC15M':
            idx = np.sort(np.concatenate([rng.choice(ix, min(per_mkt, len(ix)), replace=False)
                                          for ix in (np.nonzero(mk == m)[0] for m in np.unique(mk))]))
        else:
            idx = np.sort(rng.choice(len(mk), min(perp_per_day, len(mk)), replace=False))
        S.append(z['S'][idx].astype(np.float16)); B.append(z['B'][idx]); Y.append(z['Y'][idx]); X.append(z['X'][idx])
        D.append(np.full(len(idx), day))
    return np.concatenate(S), np.concatenate(B), np.concatenate(Y), np.concatenate(X), np.concatenate(D)


def train_stats(market):
    cache = f'{ROOT}/models_fair_{market}_trainstats.npz'
    if os.path.exists(cache):
        return dict(np.load(cache))
    files = [f'{DATA}/feat/fair/{market}_{d}.npz' for d in TRAIN_DAYS]
    S, B, Y, X, D = load_files(files, 1500, 40000, market)
    tr = (D != TEST_DAY) & (D != VAL_DAY); va = D == VAL_DAY
    st = dict(ysd=Y[tr].std(0) + 1e-6, mu=np.float32(S[tr][..., 6].astype(np.float32).mean()),
              sd=np.float32(S[tr][..., 6].astype(np.float32).std() + 1e-6), Bmu=B[tr].mean(0), Bsd=B[tr].std(0) + 1e-6)
    # LightGBM baseline, same recipe as fair_models.baselines
    import lightgbm as lgb
    F = np.c_[X, B].astype(np.float32)
    med = np.nanmedian(F[tr], 0); F = np.where(np.isfinite(F), F, med)
    lo, hi = np.percentile(F[tr], .5, 0), np.percentile(F[tr], 99.5, 0); F = np.clip(F, lo, hi)
    for k in range(Y.shape[1]):
        m = lgb.train(dict(objective='regression', learning_rate=.05, num_leaves=63, min_data_in_leaf=200,
                           feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, num_threads=16, verbose=-1),
                      lgb.Dataset(F[tr], Y[tr, k]), 1500, valid_sets=[lgb.Dataset(F[va], Y[va, k])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        m.save_model(f'{ROOT}/models_fair_{market}_lgbm_h{k}.txt')
    st.update(fmed=med, flo=lo, fhi=hi)
    np.savez(cache, **st)
    log(f'{market} train stats + LightGBM refit cached (n_train={tr.sum()})', 'fair_oos.log')
    return st


def predict_seq(model, S, Bn, mu, sd, ysd):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(S), 4096):
            out.append(model(torch.tensor(S[i:i + 4096].astype(np.float32)), torch.tensor(Bn[i:i + 4096]), mu, sd).numpy())
    return np.concatenate(out) * ysd


def metrics(y, p):
    r = fm.r2(y, p)
    da = [float((np.sign(p[y[:, k] != 0, k]) == np.sign(y[y[:, k] != 0, k])).mean()) for k in range(y.shape[1])]
    return dict(r2=[float(v) for v in r], dir_acc=da)


def main(market, days):
    torch.set_num_threads(int(os.environ.get('NTHREADS', 16)))
    st = train_stats(market)
    import lightgbm as lgb
    boosters = [lgb.Booster(model_file=f'{ROOT}/models_fair_{market}_lgbm_h{k}.txt') for k in range(4)]
    nets = {}
    for kind in MODELS[market]:
        net = fm.FairTransformer() if kind.startswith('transformer') else fm.FairGRU()
        net.load_state_dict(torch.load(f'{ROOT}/models_fair_{market}_{kind}.pt', map_location='cpu'))
        nets[kind] = net
    for day in days:
        f = f'{DATA}/feat/fair/{market}_{day}.npz'
        if not os.path.exists(f):
            log(f'{market} {day}: no fair features, skipped', 'fair_oos.log'); continue
        t0 = time.time()
        S, B, Y, X, D = load_files([f], EVAL_PER_MKT, EVAL_PERP, market, seed=1)
        Bn = ((B - st['Bmu']) / st['Bsd']).astype(np.float32)
        F = np.c_[X, B].astype(np.float32); F = np.where(np.isfinite(F), F, st['fmed']); F = np.clip(F, st['flo'], st['fhi'])
        res = {'zero': metrics(Y, np.zeros_like(Y)),
               'LightGBM (handcrafted)': metrics(Y, np.stack([b.predict(F, num_threads=16) for b in boosters], 1))}
        for kind, net in nets.items():
            res[kind] = metrics(Y, predict_seq(net, S, Bn, float(st['mu']), float(st['sd']), st['ysd']))
        nm = len(np.unique(np.load(f, allow_pickle=True)['mkt']))
        out = dict(market=market, day=day, n=int(len(Y)), n_markets=int(nm), target_std=Y.std(0).tolist(),
                   frac_zero=(Y == 0).mean(0).tolist(), in_training=day in TRAIN_DAYS, res=res)
        json.dump(out, open(f'{OUT}/{market}_{day}.json', 'w'), indent=1)
        log(f'{market} {day} n={len(Y)} ' + ' | '.join(f"{k} {res[k]['r2'][0]:.4f}/{res[k]['r2'][2]:.4f}" for k in res)
            + f' {time.time()-t0:.0f}s', 'fair_oos.log')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
