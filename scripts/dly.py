"""LATENCY SWEEP: fake PnL vs execution delay, with and without the no-chase guard.

Off the cached 0.1s-target predictions -- identical trades to the fake-PnL table,
nothing retrained.

For each delay L:
  delayed   entry at t+L, always.                     The pure cost of latency.
  guarded   entry at t+L, ONLY if the touch has not moved against us:
            ask(t+L) <= ask(t) for a buy, bid(t+L) >= bid(t) for a sell.

fake PnL = side*(mid(t_e+h) - mid(t_e)) - half_spread(t_e) - 0.07*m(t_e)*(1-m(t_e))
one cost, at entry, no exit. L=0 reproduces the baseline.

The guard changes WHICH trades happen, so its pass rate is reported per delay --
a guard that fires on 5% of trades is a different strategy from one that fires
on 60%, and the PnL alone cannot tell them apart.
"""
import sys, os, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
COINS=os.environ.get("DLY_COINS","BTC,ETH,XRP,NEAR").split(",")
CACHE="/mnt/data/fccache"; TT="/mnt/data/touchtape"
SUFFIX=os.environ.get("DLY_SUFFIX","")
OUT=f"/var/lib/kalshi-study/DELAY{SUFFIX.replace('.','p')}.json"
TAKER=0.07; SCALE=10000.0
DELAYS=[0.0,0.005,0.010,0.020,0.050,0.100,0.200,0.500]
HOR=[0.1,1.0,5.0,30.0]
res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT))
    except Exception: res={}
def save():
    t=OUT+".tmp"; json.dump(res,open(t,"w"),indent=1,default=str); os.replace(t,OUT)
def boot(v,g,B=2000,seed=0):
    u,inv=np.unique(g,return_inverse=True)
    sm=np.bincount(inv,weights=v,minlength=len(u))
    ct=np.bincount(inv,minlength=len(u)).astype("float64")
    r=np.random.default_rng(seed)
    pk=r.integers(0,len(u),size=(B,len(u)))
    d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
    return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
            float(min((d<=0).mean(),(d>=0).mean())*2))

for coin in COINS:
    cf=f"{CACHE}/{coin}{SUFFIX}.npz"
    if not os.path.exists(cf): print(f"{coin}: no cache",flush=True); continue
    z=np.load(cf,allow_pickle=True)
    p=z["p"];m0=z["m0"];s0=z["s0"];t0=z["t0"];mc_te=z["mc_te"]
    munq=z["munq"];c_te=z["c_te"]
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0)
    take=(np.abs(p)*m0)>(half+fee0); side=np.sign(p)
    n=len(p)
    print(f"\n=== {coin}: {int(take.sum()):,} gated trades",flush=True)
    # entry state at every delay, plus forward mids from each entry anchor
    EM={L:np.full(n,np.nan) for L in DELAYS}     # mid at t+L
    ES={L:np.full(n,np.nan) for L in DELAYS}     # spread at t+L
    EA={L:np.full(n,np.nan) for L in DELAYS}     # ask at t+L
    EB={L:np.full(n,np.nan) for L in DELAYS}     # bid at t+L
    FW={(L,h):np.full(n,np.nan) for L in DELAYS for h in HOR}
    for ci in c_te:
        f=f"{TT}/tt_{str(munq[ci])}.parquet"
        if not os.path.exists(f): continue
        tt=pq.read_table(f,columns=["ts","best_bid","best_ask"])
        tts=tt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        bb=tt.column("best_bid").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        ba=tt.column("best_ask").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        o=np.argsort(tts); tts,bb,ba=tts[o],bb[o],ba[o]
        tm=np.where(np.isfinite(bb)&np.isfinite(ba),(bb+ba)/2.0,np.nan)
        rows=np.flatnonzero(mc_te==ci)
        if len(rows)==0: continue
        tr=t0[rows]
        def at(tgt,arr):
            j=np.searchsorted(tts,tgt,side="right")-1
            ok=j>=0; j=np.clip(j,0,len(tts)-1)
            return np.where(ok,arr[j],np.nan)
        for L in DELAYS:
            EM[L][rows]=at(tr+L,tm); EA[L][rows]=at(tr+L,ba); EB[L][rows]=at(tr+L,bb)
            ES[L][rows]=at(tr+L,ba)-at(tr+L,bb)
            for h in HOR:
                j=np.searchsorted(tts,tr+L+h,side="right")-1
                ok=j>=0; jc=np.clip(j,0,len(tts)-1)
                v=np.where(ok,tm[jc],np.nan); age=np.where(ok,(tr+L+h)-tts[jc],np.nan)
                FW[(L,h)][rows]=np.where(age<=h*0.25+0.005,v,np.nan)
    print(f"{'delay':>7s} {'h':>5s} {'arm':8s} {'trades':>8s} {'pass%':>6s} "
          f"{'PnL/tr':>9s} {'p':>7s} {'sig':>4s} {'95% CI':>22s}",flush=True)
    for L in DELAYS:
        nw=np.where(side>0, EA[L]<=EA[0.0], EB[L]>=EB[0.0])
        pr=float(nw[take&np.isfinite(EA[L])].mean()) if (take&np.isfinite(EA[L])).any() else float("nan")
        res[f"{coin}|pass|{L}"]={"guard_pass":pr}
        for h in HOR:
            hm=ES[L]/2.0; fm=TAKER*EM[L]*(1.0-EM[L])
            pnl=side*(FW[(L,h)]-EM[L])-hm-fm
            base=take&np.isfinite(pnl)&np.isfinite(ES[L])&(ES[L]>=0)
            for arm,msk in (("delayed",base),("guarded",base&nw)):
                if msk.sum()<100: continue
                v=pnl[msk]; g=mc_te[msk]
                lo,hi,pv=boot(v,g); sig=(lo>0)==(hi>0)
                print(f"{1000*L:>6.0f}ms {h:>5.1f} {arm:8s} {int(msk.sum()):>8,} "
                      f"{100*pr:>5.1f}% {100*v.mean():>+8.4f}c {pv:>7.4f} "
                      f"{('SIG' if sig else '-'):>4s} [{100*lo:+.4f},{100*hi:+.4f}]c",flush=True)
                res[f"{coin}|{arm}|{L}|{h}"]={"trades":int(msk.sum()),
                    "pnl_cents":float(100*v.mean()),"guard_pass":pr,
                    "ci_cents":[float(100*lo),float(100*hi)],"p":pv,"sig":bool(sig)}
        save()
    gc.collect()
save(); print(f"\nwrote {OUT}")
