"""Part 2 (bitter lesson): raw-event Transformer and GRU predicting multi-horizon mid moves, vs handcrafted baselines.

usage: python fair_models.py MARKET MODEL   (MARKET in BTC15M|BTCPERP ; MODEL in baselines|transformer|gru)
Tokens = raw book events (type, side, add/cancel, price offset from mid in ticks, log size) with a continuous-time
encoding of each event's AGE (seconds before the prediction time) instead of token position."""
import sys, os, json, glob, time, math
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from common import *

torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
REP = f'{ROOT}/report/fair'
os.makedirs(REP, exist_ok=True)
HZ = ['0.1s', '0.5s', '1s', '10s']
TEST_DAY = os.environ.get('TEST_DAY', '2026-09-23'); VAL_DAY = os.environ.get('VAL_DAY', '2026-09-22')


def load(market, per_mkt=1500, perp_per_day=40000, seed=0):
    rng = np.random.default_rng(seed)
    S, B, Y, X, D, MK = [], [], [], [], [], []
    for f in sorted(glob.glob(f'{DATA}/feat/fair/{market}_*.npz')):
        day = f.split('_')[-1][:-4]
        z = np.load(f, allow_pickle=True)
        mk = z['mkt']
        if market == 'BTC15M':
            idx = []
            for m in np.unique(mk):
                ix = np.nonzero(mk == m)[0]
                idx.append(rng.choice(ix, min(per_mkt, len(ix)), replace=False))
            idx = np.sort(np.concatenate(idx))
        else:
            idx = np.sort(rng.choice(len(mk), min(perp_per_day, len(mk)), replace=False))
        S.append(z['S'][idx].astype(np.float16)); B.append(z['B'][idx]); Y.append(z['Y'][idx]); X.append(z['X'][idx])
        D.append(np.full(len(idx), day)); MK.append(mk[idx])
    return (np.concatenate(S), np.concatenate(B), np.concatenate(Y), np.concatenate(X), np.concatenate(D),
            np.concatenate(MK))


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


class TimeEnc(nn.Module):
    """continuous-time encoding of event age (seconds): fixed log-spaced sinusoids + learned Time2Vec part."""
    def __init__(self, d, k=16):
        super().__init__()
        w = 2 * math.pi / torch.logspace(-3, 3, k)  # periods 1ms .. 1000s
        self.register_buffer('w', w)
        self.lin = nn.Linear(2 * k + 2, d)
        self.t2v = nn.Linear(1, 8)
        self.out = nn.Linear(d + 8, d)

    def forward(self, age):  # age (B,L)
        a = age.unsqueeze(-1)
        f = torch.cat([torch.sin(a * self.w), torch.cos(a * self.w), torch.log1p(a * 1000) / 10, (a < 1e-3).float()], -1)
        t2 = self.t2v(torch.log1p(a * 1000))
        t2 = torch.cat([t2[..., :1], torch.sin(t2[..., 1:])], -1)
        return self.out(torch.cat([self.lin(f), t2], -1))


class TokEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cont = nn.Linear(5, d)            # side, add/cancel, price rel (scaled), log size (scaled), |rel|
        self.typ = nn.Embedding(3, d)
        self.prc = nn.Embedding(101, d)        # rounded price offset -50..50 ticks
        self.time = TimeEnc(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, S, size_mu, size_sd):
        typ = S[..., :3].argmax(-1)
        rel = S[..., 5]
        cont = torch.stack([S[..., 3], S[..., 4], rel / 10, (S[..., 6] - size_mu) / size_sd, rel.abs().clamp(max=20) / 10], -1)
        pidx = (rel.round().clamp(-50, 50) + 50).long()
        e = self.cont(cont) + self.typ(typ) + self.prc(pidx) + self.time(S[..., 7])
        return self.norm(e)


class FairTransformer(nn.Module):
    def __init__(self, d=96, layers=4, heads=4, nout=4):
        super().__init__()
        self.emb = TokEmb(d)
        self.book = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(lay, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, nout))

    def forward(self, S, Bk, mu, sd):
        e = self.emb(S, mu, sd)
        b = self.book(Bk).unsqueeze(1)
        x = torch.cat([self.cls.expand(len(S), -1, -1), b, e], 1)
        h = self.enc(x)
        return self.head(h[:, 0])


class FairGRU(nn.Module):
    def __init__(self, d=96, h=160, nout=4):
        super().__init__()
        self.emb = TokEmb(d)
        self.dt = nn.Linear(1, d)
        self.rnn = nn.GRU(d, h, num_layers=2, batch_first=True, dropout=0.1)
        self.book = nn.Sequential(nn.Linear(20, d), nn.GELU())
        self.head = nn.Sequential(nn.Linear(h + d, h), nn.GELU(), nn.Linear(h, nout))

    def forward(self, S, Bk, mu, sd):
        # chronological order: tokens are stored oldest->newest already (age decreasing)
        e = self.emb(S, mu, sd)
        age = S[..., 7]
        dt = torch.zeros_like(age); dt[:, 1:] = (age[:, :-1] - age[:, 1:]).clamp(min=0)
        e = e + self.dt(torch.log1p(dt * 1000).unsqueeze(-1) / 5)
        o, hN = self.rnn(e)
        return self.head(torch.cat([hN[-1], self.book(Bk)], -1))


def train_seq(kind, S, B, Y, D, run, budget_s):
    tr = (D != TEST_DAY) & (D != VAL_DAY); va = D == VAL_DAY; te = D == TEST_DAY
    ysd = Y[tr].std(0) + 1e-6
    clipv = 8
    Yn = np.clip(Y / ysd, -clipv, clipv).astype(np.float32)
    mu = float(S[tr][..., 6].astype(np.float32).mean()); sd = float(S[tr][..., 6].astype(np.float32).std() + 1e-6)
    Bmu, Bsd = B[tr].mean(0), B[tr].std(0) + 1e-6
    Bn = ((B - Bmu) / Bsd).astype(np.float32)
    model = FairTransformer() if kind.startswith('transformer') else FairGRU()
    nparam = sum(p.numel() for p in model.parameters())
    base_lr = float(os.environ.get('LR', 6e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    init = os.environ.get('INIT_FROM')
    step0 = 0
    if init:
        ck = torch.load(init, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and 'model' in ck:
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step0 = ck.get('step', 0)
        else:
            model.load_state_dict(ck)
        log(f'  {kind}: resumed weights from {init} (prior steps {step0})', 'fair.log')
    ckpt_path = f'{ROOT}/models_fair_ckpt_{kind}.pt'
    sched_total = 10 ** 9
    tri = np.nonzero(tr)[0]; vai = np.nonzero(va)[0]; tei = np.nonzero(te)[0]
    vsub = vai[np.random.default_rng(0).choice(len(vai), min(20000, len(vai)), replace=False)]
    bs = int(os.environ.get('BS', 256))
    t0 = time.time(); step = 0; best = (-9, None); ep = 0

    def predict(ix):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ix), 2048):
                j = ix[i:i + 2048]
                out.append(model(torch.tensor(S[j].astype(np.float32)), torch.tensor(Bn[j]), mu, sd).numpy())
        model.train()
        return np.concatenate(out) * ysd

    warm = int(os.environ.get('WARM', 300))
    if init:
        pv = predict(vsub); rv = r2(Y[vsub], pv)
        best = (float(rv.mean()), {k: v.clone() for k, v in model.state_dict().items()})
        log(f'  {kind} resumed val R2 {np.round(rv, 4)}', 'fair.log')
        run.log({**{f'{kind}/val_r2_{h}': float(v) for h, v in zip(HZ, rv)}, f'{kind}/val_r2_mean': float(rv.mean()), 'step': 0})
    while time.time() - t0 < budget_s:
        perm = np.random.permutation(tri)
        for i in range(0, len(perm), bs):
            j = np.sort(perm[i:i + bs])
            lr = base_lr * min(1, (step + 1) / warm) * max(0.1, 1 - (time.time() - t0) / budget_s)
            for g in opt.param_groups:
                g['lr'] = lr
            p = model(torch.tensor(S[j].astype(np.float32)), torch.tensor(Bn[j]), mu, sd)
            loss = nn.functional.mse_loss(p, torch.tensor(Yn[j]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 100 == 0:
                run.log({f'{kind}/train_loss': float(loss), f'{kind}/lr': lr, 'step': step})
            if step % 1500 == 0 or time.time() - t0 > budget_s:
                pv = predict(vsub)
                rv = r2(Y[vsub], pv)
                run.log({**{f'{kind}/val_r2_{h}': float(v) for h, v in zip(HZ, rv)}, f'{kind}/val_r2_mean': float(rv.mean()), 'step': step})
                log(f'  {kind} step {step} ep {ep} val R2 {np.round(rv, 4)} {time.time()-t0:.0f}s', 'fair.log')
                if rv.mean() > best[0]:
                    best = (float(rv.mean()), {k: v.clone() for k, v in model.state_dict().items()})
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step0 + step}, ckpt_path)
            if time.time() - t0 > budget_s:
                break
        ep += 1
    model.load_state_dict(best[1])
    pt = predict(tei)
    return model, r2(Y[tei], pt), nparam, step, pt, tei


def baselines(X, B, Y, D, run):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    tr = (D != TEST_DAY) & (D != VAL_DAY); va = D == VAL_DAY; te = D == TEST_DAY
    F = np.c_[X, B].astype(np.float32)
    med = np.nanmedian(F[tr], 0); F = np.where(np.isfinite(F), F, med)
    lo, hi = np.percentile(F[tr], .5, 0), np.percentile(F[tr], 99.5, 0); F = np.clip(F, lo, hi)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; Fn = (F - mu) / sd
    out = {}
    rid = Ridge(alpha=10).fit(Fn[tr], Y[tr]); out['ridge (handcrafted)'] = r2(Y[te], rid.predict(Fn[te]))
    P = np.zeros_like(Y[te])
    for k in range(Y.shape[1]):
        m = lgb.train(dict(objective='regression', learning_rate=.05, num_leaves=63, min_data_in_leaf=200,
                           feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, num_threads=8, verbose=-1),
                      lgb.Dataset(F[tr], Y[tr, k]), 1500, valid_sets=[lgb.Dataset(F[va], Y[va, k])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        P[:, k] = m.predict(F[te])
    out['LightGBM (handcrafted)'] = r2(Y[te], P)
    # MLP on handcrafted (the original project's model family, small)
    ysd = Y[tr].std(0) + 1e-6
    net = nn.Sequential(nn.Linear(F.shape[1], 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 256), nn.GELU(),
                        nn.Dropout(.1), nn.Linear(256, 4))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    Xt = torch.tensor(Fn[tr].astype(np.float32)); Yt = torch.tensor(np.clip(Y[tr] / ysd, -8, 8).astype(np.float32))
    Xv = torch.tensor(Fn[va].astype(np.float32)); best = (-9, None)
    for ep in range(25):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            l = nn.functional.mse_loss(net(Xt[b]), Yt[b]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            rv = r2(Y[va], net(Xv).numpy() * ysd).mean()
        if rv > best[0]:
            best = (rv, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        out['MLP (handcrafted)'] = r2(Y[te], net(torch.tensor(Fn[te].astype(np.float32))).numpy() * ysd)
    out['zero'] = r2(Y[te], np.zeros_like(Y[te]))
    return out


def main(market, kind):
    budget = int(os.environ.get('BUDGET', 2400))
    S, B, Y, X, D, MK = load(market)
    days = sorted(set(D))
    log(f'{market} {kind}: n={len(Y)} days={days} test={TEST_DAY}', 'fair.log')
    run = wandb_init(f'fair-{market}-{kind}', f'fair-{market}', job_type='train',
                     config=dict(market=market, model=kind, n=len(Y), days=days, test_day=TEST_DAY, val_day=VAL_DAY,
                                 seq_len=S.shape[1], horizons=HZ, budget_s=budget))
    import wandb
    stats = dict(target_std_ticks=Y.std(0).tolist(), frac_zero=(Y == 0).mean(0).tolist())
    if kind == 'baselines':
        res = baselines(X, B, Y, D, run)
        out = {k: v.tolist() for k, v in res.items()}
    else:
        model, rt, nparam, steps, pt, tei = train_seq(kind, S, B, Y, D, run, budget)
        out = {f'{kind} (raw events)': rt.tolist()}
        stats.update(nparam=nparam, steps=steps)
        torch.save(model.state_dict(), f'{ROOT}/models_fair_{market}_{kind}.pt')
        # directional accuracy on non-zero moves
        yt = Y[tei]
        out_dir = {}
        for k, h in enumerate(HZ):
            nz = yt[:, k] != 0
            out_dir[h] = float((np.sign(pt[nz, k]) == np.sign(yt[nz, k])).mean())
        stats['directional_acc_nonzero'] = out_dir
    for name, v in out.items():
        run.summary.update({f'test_r2/{name}/{h}': x for h, x in zip(HZ, v)})
    json.dump(dict(market=market, kind=kind, r2=out, stats=stats), open(f'{REP}/{market}_{kind}.json', 'w'), indent=1)
    run.finish()
    log(f'DONE {market} {kind}: {out}', 'fair.log')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
