"""Stage B: per-market feature construction.

Produces (under DATA/feat):
  win/{regime}_{day}.parquet  : window features at anchors, for event windows N in NS and clock windows W in WS
  seq/{regime}_{day}.npz      : AE token sequences (L=64) at the same anchors
  trade/{regime}_{day}.parquet + trade_seq npz : taker orders with features + labels
  fair/{day}.npz              : BTC15M raw-event sequences + multi-horizon targets (+ handcrafted feats)

usage: python features.py REGIME DAY [workers]
"""
import sys, os, json, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import numba as nb
import multiprocessing as mp
from common import *

NS = (16, 64, 256, 1024)
WS = (1.0, 10.0, 60.0, 600.0)
SEQ_L = 64
FAIR_L = 128
HZ = (0.1, 0.5, 1.0, 10.0)
KAL_FEE = 0.07
PERP_FEE = 0.0002   # assumed 2 bps/side taker for Kalshi perps (documented assumption)

FEATS = [
    # timing (scale-free)
    'tm_cv', 'tm_logstd', 'tm_batch', 'tm_ac1', 'tm_halfratio', 'tm_fano',
    # event mix
    'ev_trade', 'ev_add', 'ev_cxl', 'ev_touch', 'ev_inside', 'ev_deep', 'ev_cxl_add', 'ev_bidshare',
    # spread
    'sp_mean', 'sp_one', 'sp_std', 'sp_volnorm',
    # imbalance
    'im1_mean', 'im1_std', 'im1_abs', 'im5_mean', 'im5_std', 'im_pred',
    # depth shape
    'dsh1', 'dsh2', 'dsh3', 'dsh4', 'dsh5', 'd_hhi', 'd_gap1', 'd_l1_vs_trade', 'd_l1_cv',
    # price dynamics
    'pm_change', 'pm_ac1', 'pm_jump', 'pm_eff', 'pm_trade_move',
    # trades
    'tr_sign_abs', 'tr_sign_ac1', 'tr_size_cv', 'tr_size_vs_l1', 'tr_impact', 'tr_walk',
    # 'trivial' / scale-revealing group, excluded from the main classifier
    'tv_rate', 'tv_bound', 'tv_logdepth', 'tv_reltick',
]
GROUPS = {
    'timing': [f for f in FEATS if f.startswith('tm_')],
    'eventmix': [f for f in FEATS if f.startswith('ev_')],
    'spread': [f for f in FEATS if f.startswith('sp_')],
    'imbalance': [f for f in FEATS if f.startswith('im')],
    'depth': [f for f in FEATS if f.startswith('d')],
    'price': [f for f in FEATS if f.startswith('pm_')],
    'trades': [f for f in FEATS if f.startswith('tr_')],
    'trivial': [f for f in FEATS if f.startswith('tv_')],
}
NF = len(FEATS)


# ------------------------------------------------------------------ per-market arrays
def prep(df, meta):
    tick = meta['tick']
    n = len(df)
    t = df.rn.values.astype(np.float64) / 1e9
    te = np.maximum.accumulate(df.ts.values.astype(np.float64) / 1e3)
    ty = df.ty.values.astype(np.int64)
    sd = df.sd.values.astype(np.int64)
    qx = df.qx.values.astype(np.float64)
    valid = df.valid.values.astype(np.int64)
    bp = df[[f'bp{k}' for k in range(5)]].values; ap = df[[f'ap{k}' for k in range(5)]].values
    bs = df[[f'bs{k}' for k in range(5)]].values.astype(np.float64); as_ = df[[f'as{k}' for k in range(5)]].values.astype(np.float64)
    mid = np.where(valid == 1, (bp[:, 0] + ap[:, 0]) / 2, np.nan)
    mid = pd.Series(mid).ffill().bfill().values
    bid = pd.Series(np.where(valid == 1, bp[:, 0], np.nan)).ffill().bfill().values
    ask = pd.Series(np.where(valid == 1, ap[:, 0], np.nan)).ffill().bfill().values
    spr = np.where(valid == 1, (ap[:, 0] - bp[:, 0]) / tick, np.nan)
    spr = pd.Series(spr).ffill().bfill().values
    b1 = np.nan_to_num(bs[:, 0]); a1 = np.nan_to_num(as_[:, 0])
    im1 = (b1 - a1) / np.maximum(b1 + a1, 1e-9)
    b5 = np.nan_to_num(bs).sum(1); a5 = np.nan_to_num(as_).sum(1)
    im5 = (b5 - a5) / np.maximum(b5 + a5, 1e-9)
    shb = np.nan_to_num(bs) / np.maximum(b5[:, None], 1e-9)
    sha = np.nan_to_num(as_) / np.maximum(a5[:, None], 1e-9)
    sh = (shb + sha) / 2
    gap1 = np.nan_to_num(((bp[:, 0] - bp[:, 1]) + (ap[:, 1] - ap[:, 0])) / 2 / tick, nan=1.0)
    lvlt = df.lvl.values.astype(np.float64) * meta['grid'] / tick
    dmid = np.zeros(n); dmid[1:] = np.diff(mid) / tick
    dmid = np.nan_to_num(dmid)
    is_binary = meta['coarse'] != 'perp'
    bound = np.minimum(mid, 1 - mid) if is_binary else np.full(n, 0.5)
    pre = pre_index(ty, te)
    return dict(t=t, te=te, ty=ty, sd=sd, qx=qx, valid=valid, mid=mid, bid=bid, ask=ask, spr=spr, im1=im1, im5=im5,
                sh=np.ascontiguousarray(sh), gap1=gap1, lvlt=lvlt, dmid=dmid, b1=b1, a1=a1, bound=bound,
                pre=pre, tick=tick, px=df.px.values.astype(np.float64), bp=bp, ap=ap, bs=bs, as_=as_)


@nb.njit(cache=True)
def pre_index(ty, te):
    """for trade rows: last row index whose exchange time < trade time (book before the fill's deltas)."""
    n = len(ty)
    pre = np.full(n, -1, np.int64)
    for i in range(n):
        if ty[i] == 1:
            j = i - 1
            c = 0
            while j >= 0 and te[j] >= te[i] and c < 400:
                j -= 1; c += 1
            pre[i] = j
    return pre


@nb.njit(cache=True)
def _corr(x, y, m):
    if m < 3:
        return np.nan
    mx = 0.0; my = 0.0
    for k in range(m):
        mx += x[k]; my += y[k]
    mx /= m; my /= m
    sxy = 0.0; sxx = 0.0; syy = 0.0
    for k in range(m):
        sxy += (x[k] - mx) * (y[k] - my); sxx += (x[k] - mx) ** 2; syy += (y[k] - my) ** 2
    if sxx <= 0 or syy <= 0:
        return np.nan
    return sxy / np.sqrt(sxx * syy)


@nb.njit(cache=True)
def window_feats(s, e, t, ty, sd, qx, spr, im1, im5, sh, gap1, lvlt, dmid, b1, a1, bound, mid, pre, tick, out):
    """features over rows [s, e) ; writes NF values into out."""
    m = e - s
    for k in range(out.shape[0]):
        out[k] = np.nan
    if m < 8:
        return
    # ---------- timing
    dts = np.empty(m - 1)
    for k in range(m - 1):
        dts[k] = max(t[s + k + 1] - t[s + k], 0.0)
    mu = dts.mean(); sdv = dts.std()
    out[0] = sdv / mu if mu > 0 else np.nan
    ld = np.log10(dts + 1e-4)
    out[1] = ld.std()
    out[2] = (dts < 1e-3).mean()
    out[3] = _corr(ld[:-1], ld[1:], m - 2)
    h = (m - 1) // 2
    if h > 1:
        a = dts[:h].mean(); b = dts[h:].mean()
        out[4] = np.log((b + 1e-6) / (a + 1e-6))
    dur = t[e - 1] - t[s]
    if dur > 0:
        cnt = np.zeros(16)
        for k in range(s, e):
            bi = int((t[k] - t[s]) / dur * 15.999)
            cnt[bi] += 1
        out[5] = cnt.var() / max(cnt.mean(), 1e-9)
    # ---------- event mix
    ntr = 0; nadd = 0; ncx = 0; ntouch = 0; nin = 0; ndeep = 0; nd = 0; nbid = 0
    for k in range(s, e):
        if ty[k] == 1:
            ntr += 1
        elif ty[k] == 0:
            nd += 1
            if qx[k] > 0:
                nadd += 1
            else:
                ncx += 1
            if lvlt[k] <= 0:
                ntouch += 1
            if lvlt[k] < 0 and qx[k] > 0:
                nin += 1
            if lvlt[k] >= 3:
                ndeep += 1
            if sd[k] == 0:
                nbid += 1
    out[6] = ntr / m
    if nd > 0:
        out[7] = nadd / nd; out[8] = ncx / nd; out[9] = ntouch / nd; out[10] = nin / nd; out[11] = ndeep / nd
        out[12] = ncx / (nadd + 1.0); out[13] = nbid / nd
    # ---------- spread
    ls = np.log1p(spr[s:e])
    out[14] = ls.mean()
    out[15] = (spr[s:e] <= 1.0001).mean()
    out[16] = ls.std()
    rv = np.sqrt((dmid[s + 1:e] ** 2).sum())
    out[17] = np.log((spr[s:e].mean() + 1e-3) / (rv + 1e-3))
    # ---------- imbalance
    out[18] = im1[s:e].mean(); out[19] = im1[s:e].std(); out[20] = np.abs(im1[s:e]).mean()
    out[21] = im5[s:e].mean(); out[22] = im5[s:e].std()
    out[23] = _corr(im1[s:e - 1], dmid[s + 1:e], m - 1)
    # ---------- depth
    hh = 0.0
    for L in range(5):
        v = sh[s:e, L].mean()
        out[24 + L] = v
        hh += v * v
    out[29] = hh
    out[30] = np.log1p(np.abs(gap1[s:e]).mean())
    # trade sizes
    sizes = np.empty(max(ntr, 1)); signs = np.empty(max(ntr, 1)); rel = np.empty(max(ntr, 1))
    walk = 0; c = 0; mv = 0; imp = 0.0; nimp = 0
    for k in range(s, e):
        if ty[k] == 1:
            sizes[c] = np.log(qx[k] + 1e-9)
            signs[c] = sd[k]
            p = pre[k]
            dep = 0.0
            if p >= 0:
                dep = a1[p] if sd[k] > 0 else b1[p]
            rel[c] = np.log((qx[k] + 1e-9) / (dep + 1e-9)) if dep > 0 else np.nan
            if dep > 0 and qx[k] > dep:
                walk += 1
            if dmid[k] != 0 or (k + 1 < e and dmid[k + 1] != 0):
                mv += 1
            if k + 5 < e:
                acc = 0.0
                for j in range(1, 6):
                    acc += dmid[k + j]
                imp += sd[k] * acc; nimp += 1
            c += 1
    l1 = 0.5 * (b1[s:e] + a1[s:e])
    if ntr >= 2:
        med = np.median(np.exp(sizes[:ntr]))
        out[31] = np.log((l1.mean() + 1e-9) / (med + 1e-9))
    out[32] = l1.std() / (l1.mean() + 1e-9)
    # ---------- price dynamics
    nch = 0
    for k in range(s + 1, e):
        if dmid[k] != 0:
            nch += 1
    out[33] = nch / (m - 1)
    if nch >= 3:
        ch = np.empty(nch); c2 = 0
        for k in range(s + 1, e):
            if dmid[k] != 0:
                ch[c2] = dmid[k]; c2 += 1
        sg = np.sign(ch)
        out[34] = _corr(sg[:-1], sg[1:], nch - 1)
        out[35] = np.log1p(np.abs(ch).mean())
        tot = (mid[e - 1] - mid[s]) / tick
        out[36] = np.log((tot * tot + 1e-3) / ((ch * ch).sum() + 1e-3))
    if ntr >= 1:
        out[37] = mv / ntr
        out[38] = np.abs(signs[:ntr].mean())
        if ntr >= 4:
            out[39] = _corr(signs[:ntr - 1], signs[1:ntr], ntr - 1)
            out[40] = sizes[:ntr].std()
        r = rel[:ntr]
        r = r[~np.isnan(r)]
        if len(r) > 0:
            out[41] = r.mean()
        if nimp > 0:
            out[42] = imp / nimp
        out[43] = walk / ntr
    # ---------- trivial group
    out[44] = np.log(m / max(dur, 1e-3))
    out[45] = bound[e - 1]
    out[46] = np.log(l1.mean() + 1.0)
    out[47] = np.log(tick / max(abs(mid[e - 1]), 1e-6))


@nb.njit(cache=True)
def batch_feats(anchors, starts, t, ty, sd, qx, spr, im1, im5, sh, gap1, lvlt, dmid, b1, a1, bound, mid, pre, tick, nf):
    X = np.full((len(anchors), nf), np.nan, np.float32)
    out = np.empty(nf)
    for i in range(len(anchors)):
        e = anchors[i]; s = starts[i]
        if s < 0:
            continue
        window_feats(s, e, t, ty, sd, qx, spr, im1, im5, sh, gap1, lvlt, dmid, b1, a1, bound, mid, pre, tick, out)
        for k in range(nf):
            X[i, k] = out[k]
    return X


SEQ_D = 16


@nb.njit(cache=True)
def seq_tokens(ends, L, t, ty, sd, qx, spr, im1, sh, lvlt, dmid, b1, a1):
    """AE tokens for rows [e-L, e). Per-sequence normalisation of size and time so no absolute scale leaks."""
    n = len(ends)
    S = np.zeros((n, L, SEQ_D), np.float32)
    for i in range(n):
        e = ends[i]; s = e - L
        if s < 1:
            continue
        # per-seq refs
        l1 = 0.0
        for k in range(s, e):
            l1 += 0.5 * (b1[k] + a1[k])
        l1 = np.log1p(l1 / L)
        ld = np.empty(L)
        for k in range(L):
            ld[k] = np.log10(max(t[s + k] - t[s + k - 1], 0.0) + 1e-4)
        mu = ld.mean(); sdv = ld.std() + 1e-6
        for k in range(L):
            r = s + k
            tok = S[i, k]
            if ty[r] == 0:
                tok[0] = 1.0
                tok[3] = 1.0 if sd[r] == 0 else -1.0
                tok[4] = 1.0 if qx[r] > 0 else -1.0
                tok[5] = min(max(lvlt[r], -5.0), 20.0) / 5.0
            elif ty[r] == 1:
                tok[1] = 1.0
                tok[3] = sd[r]
            else:
                tok[2] = 1.0
            tok[6] = np.log1p(abs(qx[r])) - l1 if ty[r] != 2 else 0.0
            tok[7] = (ld[k] - mu) / sdv
            tok[8] = np.log1p(spr[r])
            tok[9] = im1[r]
            tok[10] = min(max(dmid[r], -5.0), 5.0) / 2.0
            tok[11] = sh[r, 0]; tok[12] = sh[r, 1]; tok[13] = sh[r, 2]
            tok[14] = np.log1p(b1[r]) - l1
            tok[15] = np.log1p(a1[r]) - l1
    return S


def sample_anchors(n, valid, minstart, cap, rng):
    if n <= minstart + 10:
        return np.zeros(0, np.int64)
    step = max(64, (n - minstart) // cap)
    a = np.arange(minstart, n, step)
    a = a + rng.integers(0, step, len(a))
    a = a[a < n]
    a = a[valid[a - 1] == 1]
    return a.astype(np.int64)


def fwd_index(te, targets):
    return np.searchsorted(te, targets, side='right') - 1


def kalshi_fee(p):
    return KAL_FEE * p * (1 - p)


def trade_rows(P, meta, rng, cap=4000):
    cap = {'cryptoHourly': 400, 'cryptoRange': 300, 'alt15M': 800}.get(meta['fine'], 300 if meta['coarse'] == 'sports' else cap)
    """taker orders (consecutive fills w/ same exchange ms & side) -> features + labels."""
    ty, te, sd, qx, px = P['ty'], P['te'], P['sd'], P['qx'], P['px']
    tr = np.nonzero(ty == 1)[0]
    if len(tr) < 20:
        return None
    # aggregate fills
    key_t = te[tr]; key_s = sd[tr]
    new = np.ones(len(tr), bool)
    new[1:] = (key_t[1:] != key_t[:-1]) | (key_s[1:] != key_s[:-1])
    gid = np.cumsum(new) - 1
    ng = gid[-1] + 1
    first = tr[new]
    size = np.bincount(gid, weights=qx[tr], minlength=ng)
    notional = np.bincount(gid, weights=qx[tr] * px[tr], minlength=ng)
    nfill = np.bincount(gid, minlength=ng)
    pmax = pd.Series(px[tr]).groupby(gid).max().values; pmin = pd.Series(px[tr]).groupby(gid).min().values
    vwap = notional / np.maximum(size, 1e-12)
    side = sd[first].astype(np.float64)
    pre = P['pre'][first]
    ok = pre >= 1100
    tick = meta['tick']
    t_end = P['te'][-1]
    ok &= (te[first] + 10.0) <= t_end
    ok &= P['valid'][np.maximum(pre, 0)] == 1
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return None
    if len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    fi = first[idx]; pi = pre[idx]; sg = side[idx]; vw = vwap[idx]; sz = size[idx]
    mid0 = P['mid'][pi]; bid0 = P['bid'][pi]; ask0 = P['ask'][pi]
    touch = np.where(sg > 0, ask0, bid0)
    hitdepth = np.where(sg > 0, P['a1'][pi], P['b1'][pi])
    oppdepth = np.where(sg > 0, P['b1'][pi], P['a1'][pi])
    # previous trade info
    order_te = te[first]
    prev_dt = np.full(len(idx), np.nan); prev_same = np.zeros(len(idx)); run = np.zeros(len(idx))
    gi = idx
    has_prev = gi > 0
    prev_dt[has_prev] = order_te[gi[has_prev]] - order_te[gi[has_prev] - 1]
    prev_same[has_prev] = (side[gi[has_prev] - 1] == side[gi[has_prev]]).astype(float)
    # run length of same-side orders before
    same = np.zeros(ng); r = 0
    for k in range(1, ng):
        r = r + 1 if side[k] == side[k - 1] else 0
        same[k] = r
    run = same[gi]
    # recent typical size (median of previous 50 orders)
    med_size = pd.Series(size).rolling(50, min_periods=5).median().shift(1).values[gi]
    med_gap = pd.Series(np.r_[np.nan, np.diff(order_te)]).rolling(50, min_periods=5).median().shift(1).values[gi]
    # past mid moves (signed in trade direction) over 1s / 10s
    feats = {}
    for lag in (1.0, 10.0, 60.0):
        j = fwd_index(P['te'], order_te[gi] - lag)
        j = np.clip(j, 0, None)
        feats[f'trf_mom{int(lag)}'] = sg * (mid0 - P['mid'][j]) / tick
    feats.update({
        'trf_side': sg,
        'trf_logsize_vs_hit': np.log((sz + 1e-9) / (hitdepth + 1e-9)),
        'trf_logsize_vs_med': np.log((sz + 1e-9) / (med_size + 1e-9)),
        'trf_walk_ticks': sg * (vw - touch) / tick,
        'trf_levels': np.log1p(np.abs(pmax[gi] - pmin[gi]) / tick),
        'trf_nfill': np.log1p(nfill[gi]),
        'trf_log_prevdt_rel': np.log((prev_dt + 1e-4) / (med_gap + 1e-4)),
        'trf_prev_same': prev_same,
        'trf_run': np.log1p(run),
        'trf_spread': P['spr'][pi],
        'trf_imb1_signed': sg * P['im1'][pi],
        'trf_imb5_signed': sg * P['im5'][pi],
        'trf_opp_vs_hit': np.log((oppdepth + 1) / (hitdepth + 1)),
        'trf_price_vs_mid': sg * (vw - mid0) / tick,
        'trf_bound': P['bound'][pi],
        'trf_fee_ticks': (kalshi_fee(vw) / tick) if meta['coarse'] != 'perp' else (PERP_FEE * vw / tick),
    })
    lab = {}
    for h in HZ:
        j = fwd_index(P['te'], order_te[gi] + h)
        midh = P['mid'][j]; bidh = P['bid'][j]; askh = P['ask'][j]
        mk = sg * (midh - vw) / tick
        exitp = np.where(sg > 0, bidh, askh)
        gross_rt = sg * (exitp - vw)
        if meta['coarse'] != 'perp':
            fees = kalshi_fee(vw) + kalshi_fee(exitp)
        else:
            fees = PERP_FEE * (vw + exitp)
        net = (gross_rt - fees) / tick
        hs = str(h).replace('.', 'p')
        lab[f'y_mk_{hs}'] = mk                     # markout vs mid, ticks (taker view)
        lab[f'y_net_{hs}'] = net                   # round-trip taker pnl after 2 taker fees + exit spread, ticks
        lab[f'y_mkfee_{hs}'] = mk - 2 * fees / tick / 2 if False else (sg * (midh - vw) - fees) / tick  # exit at mid, 2 fees
    out = pd.DataFrame({**feats, **lab})
    out['row'] = fi; out['pre'] = pi; out['te'] = order_te[gi]
    out['notional_usd'] = sz * vw if meta['coarse'] != 'perp' else sz * vw
    return out


def fair_tokens(anchors, P, L):
    """Raw event tokens for the bitter-lesson models. token fields:
    [is_delta, is_trade, is_snap, side(+1 bid/yes, -1 ask/no or trade sign), add/cancel, price_rel_mid_ticks,
     log1p|qty|, age_seconds]"""
    ty, sd, qx, px, t = P['ty'], P['sd'], P['qx'], P['px'], P['t']
    tick = P['tick']; binary = P['binary']
    n = len(anchors)
    S = np.zeros((n, L, 8), np.float32)
    for i, e in enumerate(anchors):
        s = e - L
        r = np.arange(s, e)
        m0 = P['mid'][e - 1]
        tt = ty[r]
        S[i, :, 0] = tt == 0; S[i, :, 1] = tt == 1; S[i, :, 2] = tt == 2
        side = np.where(tt == 1, sd[r], np.where(sd[r] == 0, 1, -1))
        S[i, :, 3] = side
        S[i, :, 4] = np.where(tt == 0, np.sign(qx[r]), 0)
        p = px[r]
        if binary:  # NO-side delta prices -> YES-equivalent ask price
            p = np.where((tt == 0) & (sd[r] == 1), 1 - p, p)
        rel = np.where(tt == 2, 0.0, (p - m0) / tick)
        S[i, :, 5] = np.clip(rel, -50, 50)
        S[i, :, 6] = np.log1p(np.abs(qx[r]))
        S[i, :, 7] = t[e - 1] - t[r]
    # book state at anchor (raw L1-5 sizes & offsets) as separate array
    B = np.zeros((n, 20), np.float32)
    a = anchors - 1
    for k in range(5):
        B[:, k] = np.log1p(np.nan_to_num(P['bs'][a, k])); B[:, 5 + k] = np.log1p(np.nan_to_num(P['as_'][a, k]))
        B[:, 10 + k] = np.clip(np.nan_to_num((P['mid'][a] - P['bp'][a, k]) / tick, nan=50), 0, 50)
        B[:, 15 + k] = np.clip(np.nan_to_num((P['ap'][a, k] - P['mid'][a]) / tick, nan=50), 0, 50)
    return S, B


def process_market(args):
    path, meta, day, mode = args
    try:
        rng = np.random.default_rng(abs(hash(meta['ticker'] + day)) % (2 ** 32))
        df = pd.read_parquet(path)
        P = prep(df, meta)
        P['binary'] = meta['coarse'] != 'perp'
        n = len(df)
        res = {}
        if mode in ('all', 'win'):
            cap = {'BTC15M': 500, 'alt15M': 150, 'cryptoHourly': 60, 'cryptoRange': 60, 'BTCPERP': 3000,
                   'ETHPERP': 3000, 'altPERP': 1200}.get(meta['fine'], 40)
            anc = sample_anchors(n, P['valid'], 1100, cap, rng)
            if len(anc) > 0:
                blocks = {}
                args_ = (P['t'], P['ty'], P['sd'], P['qx'], P['spr'], P['im1'], P['im5'], P['sh'], P['gap1'], P['lvlt'],
                         P['dmid'], P['b1'], P['a1'], P['bound'], P['mid'], P['pre'], P['tick'], NF)
                for N in NS:
                    blocks[f'N{N}'] = batch_feats(anc, anc - N, *args_)
                for W in WS:
                    st = np.searchsorted(P['t'], P['t'][anc - 1] - W, side='left').astype(np.int64)
                    st = np.where(anc - st >= 8, st, -1)
                    blocks[f'W{int(W)}'] = batch_feats(anc, st, *args_)
                seq = seq_tokens(anc, SEQ_L, P['t'], P['ty'], P['sd'], P['qx'], P['spr'], P['im1'], P['sh'],
                                 P['lvlt'], P['dmid'], P['b1'], P['a1'])
                # forward mid moves for anchors (for analysis)
                fw = {}
                for h in HZ:
                    j = fwd_index(P['te'], P['te'][anc - 1] + h)
                    fw[f'fw_{h}'] = (P['mid'][j] - P['mid'][anc - 1]) / P['tick']
                res['win'] = (anc, blocks, seq, fw)
        if mode in ('all', 'trade'):
            tr = trade_rows(P, meta, rng)
            if tr is not None:
                pre = tr['pre'].values.astype(np.int64)
                args_ = (P['t'], P['ty'], P['sd'], P['qx'], P['spr'], P['im1'], P['im5'], P['sh'], P['gap1'], P['lvlt'],
                         P['dmid'], P['b1'], P['a1'], P['bound'], P['mid'], P['pre'], P['tick'], NF)
                Xw = batch_feats(pre + 1, pre + 1 - 256, *args_)
                sq = seq_tokens(pre + 1, SEQ_L, P['t'], P['ty'], P['sd'], P['qx'], P['spr'], P['im1'], P['sh'],
                                P['lvlt'], P['dmid'], P['b1'], P['a1'])
                res['trade'] = (tr, Xw, sq)
        if mode in ('all', 'fair') and meta['fine'] in ('BTC15M', 'BTCPERP'):
            anc = np.arange(FAIR_L + 1100, n, 6, dtype=np.int64)
            anc = anc[(P['valid'][anc - 1] == 1)]
            anc = anc[P['te'][anc - 1] + 10.0 <= P['te'][-1]]
            capf = 6000 if meta['fine'] == 'BTC15M' else 60000
            if len(anc) > capf:
                anc = np.sort(rng.choice(anc, capf, replace=False))
            if len(anc) > 0:
                S, B = fair_tokens(anc, P, FAIR_L)
                Y = np.zeros((len(anc), len(HZ)), np.float32)
                for k, h in enumerate(HZ):
                    j = fwd_index(P['te'], P['te'][anc - 1] + h)
                    Y[:, k] = (P['mid'][j] - P['mid'][anc - 1]) / P['tick']
                args_ = (P['t'], P['ty'], P['sd'], P['qx'], P['spr'], P['im1'], P['im5'], P['sh'], P['gap1'], P['lvlt'],
                         P['dmid'], P['b1'], P['a1'], P['bound'], P['mid'], P['pre'], P['tick'], NF)
                Xh = batch_feats(anc, anc - 256, *args_)
                Xh2 = batch_feats(anc, anc - 64, *args_)
                res['fair'] = (S, B, Y, np.concatenate([Xh, Xh2], 1), P['te'][anc - 1])
        return meta, res
    except Exception:
        import traceback
        return meta, {'error': traceback.format_exc()[-800:]}


def main(regime, day, workers=10, mode='all'):
    src = f'{DATA}/events/{regime}/{day}'
    metas = json.load(open(f'{src}/_meta.json'))
    out = f'{DATA}/feat'
    for d in ('win', 'seq', 'trade', 'fair'):
        os.makedirs(f'{out}/{d}', exist_ok=True)
    tag = f'{regime}_{day}'
    if os.path.exists(f'{out}/done_{tag}_{mode}'):
        log(f'skip feat {tag}', 'features.log'); return
    t0 = time.time()
    jobs = [(f"{src}/{m['ticker'].replace('/', '_')}.parquet", m, day, mode) for m in metas]
    jobs.sort(key=lambda j: -j[1]['n'])
    W_rows, SEQ, TR_rows, TR_X, TR_S, FAIR = [], [], [], [], [], []
    nerr = 0
    with mp.Pool(workers) as pool:
        for meta, res in pool.imap_unordered(process_market, jobs, chunksize=1):
            if 'error' in res:
                nerr += 1
                if nerr <= 3:
                    log(f"  ERR {meta['ticker']}: {res['error']}", 'features.log')
                continue
            base = dict(ticker=meta['ticker'], day=day, regime=regime, coarse=meta['coarse'], fine=meta['fine'])
            if 'win' in res:
                anc, blocks, seq, fw = res['win']
                cols = {'anchor': anc}
                for k, v in base.items():
                    cols[k] = np.full(len(anc), v, dtype=object)
                for bn, X in blocks.items():
                    for j, f in enumerate(FEATS):
                        cols[f'{bn}__{f}'] = X[:, j]
                cols.update(fw)
                d = pd.DataFrame(cols)
                W_rows.append(d); SEQ.append(seq)
            if 'trade' in res:
                tr, Xw, sq = res['trade']
                extra = {k: np.full(len(tr), v, dtype=object) for k, v in base.items()}
                extra.update({f'w__{f}': Xw[:, j] for j, f in enumerate(FEATS)})
                tr = pd.concat([tr.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
                TR_rows.append(tr); TR_S.append(sq)
            if 'fair' in res:
                S, B, Y, Xh, te = res['fair']
                FAIR.append((meta['ticker'], meta['fine'], S, B, Y, Xh, te))
    if W_rows:
        W = pd.concat(W_rows, ignore_index=True)
        W.to_parquet(f'{out}/win/{tag}.parquet', index=False)
        np.save(f'{out}/seq/{tag}.npy', np.concatenate(SEQ).astype(np.float16))
    if TR_rows:
        T = pd.concat(TR_rows, ignore_index=True)
        T.to_parquet(f'{out}/trade/{tag}.parquet', index=False)
        np.save(f'{out}/trade/{tag}_seq.npy', np.concatenate(TR_S).astype(np.float16))
    if FAIR:
        for fine in ('BTC15M', 'BTCPERP'):
            F = [f for f in FAIR if f[1] == fine]
            if not F:
                continue
            np.savez(f'{out}/fair/{fine}_{day}.npz', S=np.concatenate([f[2] for f in F]),
                     B=np.concatenate([f[3] for f in F]), Y=np.concatenate([f[4] for f in F]),
                     X=np.concatenate([f[5] for f in F]), te=np.concatenate([f[6] for f in F]),
                     mkt=np.concatenate([np.full(len(f[4]), f[0]) for f in F]))
    open(f'{out}/done_{tag}_{mode}', 'w').write('1')
    log(f'DONE feat {tag}: win={sum(len(w) for w in W_rows)} trades={sum(len(t) for t in TR_rows)} '
        f'fair={sum(len(f[4]) for f in FAIR)} errs={nerr} {time.time()-t0:.0f}s', 'features.log')


if __name__ == '__main__':
    regime, day = sys.argv[1], sys.argv[2]
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    mode = sys.argv[4] if len(sys.argv) > 4 else 'all'
    main(regime, day, w, mode)
