"""HORIZON-SCALED GATE: edge > k*cost, with k swept separately in the tapered
and mid price bands.

WHY. In the tapered tick regions ([0,0.10) and [0.90,1.00], tick 0.0010) the
taker fee 0.07*m*(1-m) and the minimum spread both collapse, so cost falls to
~0.14c. `edge > cost` is then nearly free to clear. A model trained on a LONG
target predicts a correspondingly LARGE move, so it clears that tiny bar almost
always -- the 1.0s model's take rate is 17.3% at 98-100c against 1.18% for the
0.1s model -- and fires on mostly noise, diluting itself to zero. The gate
threshold was only ever calibrated for the mid-range.

So sweep k. k=1 is the current rule. Larger k in the tapered band should restore
selectivity without changing the model. Reported per (model, band, k):
trades, latency-adjusted PnL per contract, TOTAL dollars, significance.

Band split is on the entry mid -- observable before trading, a venue-structure
split, not conditioning the forecast on price.
"""
import sys, os, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
CACHE="/mnt/data/fccache"; SZ="/mnt/data/sizetape"
OUT="/var/lib/kalshi-study/GATE.json"
TAKER=0.07; SCALE=10000.0; DELAY=0.010
COIN=os.environ.get("GT_COIN","BTC")
TAGS=os.environ.get("GT_TAGS","_t0.1,_t0.5,_t1.0").split(",")
HOLDS=[5.0,30.0]
KS=[1.0,1.5,2.0,3.0,5.0,8.0]
res={}
def boot(v,g,B=1500,seed=0):
    u,inv=np.unique(g,return_inverse=True)
    sm=np.bincount(inv,weights=v,minlength=len(u))
    ct=np.bincount(inv,minlength=len(u)).astype("float64")
    r=np.random.default_rng(seed)
    pk=r.integers(0,len(u),size=(B,len(u)))
    d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
    return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
            float(min((d<=0).mean(),(d>=0).mean())*2))

for TAG in TAGS:
    cf=f"{CACHE}/{COIN}{TAG}.npz"
    if not os.path.exists(cf): print(f"no cache {cf}",flush=True); continue
    z=np.load(cf,allow_pickle=True)
    p=z["p"];m0=z["m0"];s0=z["s0"];t0=z["t0"];mc_te=z["mc_te"]
    munq=z["munq"];c_te=z["c_te"];r2=float(z["r2"])
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0); cost0=half+fee0
    edge=np.abs(p)*m0; side=np.sign(p)
    n=len(p)
    A0=np.full(n,np.nan);B0=np.full(n,np.nan);AD=np.full(n,np.nan);BD=np.full(n,np.nan)
    MD=np.full(n,np.nan);SDv=np.full(n,np.nan);AV=np.full(n,np.nan)
    FW={h:np.full(n,np.nan) for h in HOLDS}
    nm=0
    for ci in c_te:
        f=f"{SZ}/sz_{str(munq[ci])}.parquet"
        if not os.path.exists(f): continue
        nm+=1
        tt=pq.read_table(f,columns=["ts","best_bid","best_ask","bid_size_1","ask_size_1"])
        tts=tt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        bb=tt.column("best_bid").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        ba=tt.column("best_ask").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        bs=tt.column("bid_size_1").to_numpy(zero_copy_only=False).astype("float64")/100.0
        as_=tt.column("ask_size_1").to_numpy(zero_copy_only=False).astype("float64")/100.0
        o=np.argsort(tts); tts,bb,ba,bs,as_=tts[o],bb[o],ba[o],bs[o],as_[o]
        tm=np.where(np.isfinite(bb)&np.isfinite(ba),(bb+ba)/2.0,np.nan)
        rows=np.flatnonzero(mc_te==ci)
        if len(rows)==0: continue
        tr=t0[rows]
        def at(tg,arr):
            j=np.searchsorted(tts,tg,side="right")-1
            ok=j>=0; j=np.clip(j,0,len(tts)-1)
            return np.where(ok,arr[j],np.nan)
        A0[rows]=at(tr,ba); B0[rows]=at(tr,bb)
        AD[rows]=at(tr+DELAY,ba); BD[rows]=at(tr+DELAY,bb)
        MD[rows]=at(tr+DELAY,tm); SDv[rows]=at(tr+DELAY,ba)-at(tr+DELAY,bb)
        sd=side[rows]
        AV[rows]=np.where(sd>0,at(tr+DELAY,as_),at(tr+DELAY,bs))
        for h in HOLDS:
            j=np.searchsorted(tts,tr+DELAY+h,side="right")-1
            ok=j>=0; jc=np.clip(j,0,len(tts)-1)
            v=np.where(ok,tm[jc],np.nan); age=np.where(ok,(tr+DELAY+h)-tts[jc],np.nan)
            FW[h][rows]=np.where(age<=h*0.25+0.005,v,np.nan)
    nw=np.where(side>0,AD<=A0,BD>=B0)
    hm=SDv/2.0; fm=TAKER*MD*(1.0-MD)
    tap=(m0<0.10)|(m0>=0.90)
    print(f"\n{'='*104}")
    print(f"{COIN}{TAG}  r2={r2:+.6f}  delay={1000*DELAY:.0f}ms guarded  {nm} markets")
    print('='*104)
    print(f"{'band':9s} {'hold':>5s} {'k':>4s} {'trades':>8s} {'take%':>6s} "
          f"{'c/contract':>11s} {'sig':>4s} {'total $':>9s} {'depth':>7s} {'$/d@500':>9s}")
    for h in HOLDS:
        pnl=side*(FW[h]-MD)-hm-fm
        okr=np.isfinite(pnl)&np.isfinite(SDv)&(SDv>=0)&np.isfinite(AV)&nw
        for bname,bm in (("tapered",tap),("mid",~tap)):
            for k in KS:
                g=okr&bm&(edge>k*cost0)
                if g.sum()<150: continue
                v=pnl[g]
                lo,hi,pv=boot(v,mc_te[g]); sig=(lo>0)==(hi>0)
                tpd=g.sum()/max(nm,1)*96
                f500=float(np.minimum(500,AV[g]).mean())
                pc=float(v.mean())
                print(f"{bname:9s} {h:>5.0f} {k:>4.1f} {int(g.sum()):>8,} "
                      f"{100*g.sum()/max((okr&bm).sum(),1):>5.2f}% {100*pc:>+10.4f}c "
                      f"{('SIG' if sig else '-'):>4s} {v.sum():>8,.0f}$ {np.median(AV[g]):>7,.0f} "
                      f"{tpd*f500*pc:>9,.0f}",flush=True)
                res[f"{COIN}{TAG}|{bname}|{h}|{k}"]={"trades":int(g.sum()),
                    "cents_per_contract":100*pc,"sig":bool(sig),"total_dollars":float(v.sum()),
                    "median_depth":float(np.median(AV[g])),"trades_per_day":float(tpd),
                    "usd_day_500":float(tpd*f500*pc),"ci":[100*lo,100*hi],"p":pv,"r2":r2}
    json.dump(res,open(OUT,"w"),indent=1,default=str)
print(f"\nwrote {OUT}")
