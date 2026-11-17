"""ROUND-TRIP PnL BY PRICE BAND. The exit is charged, because a timed hold has one.

Everything reported before this charged ONE cost and marked out at the mid. That
is a mark, not a PnL: a 5s hold has to be closed, and Kalshi charges the taker
fee on EVERY trade. So:

  round = side*(mid(te+h) - mid(te))
          - half(te)    - 0.07*m(te)*(1-m(te))        enter, crossing
          - half(te+h)  - 0.07*m(te+h)*(1-m(te+h))    exit, crossing

WHY THE TAILS MIGHT STILL WORK. Cost is not uniform across the grid. The tick is
0.0010 on [0,0.10) and [0.90,1.00] against 0.0100 in between, and the fee
0.07*m*(1-m) collapses at the extremes: 0.072c at m=0.02 versus 1.750c at
m=0.50. A full round trip therefore costs ~0.28c in the extreme tails and ~2.4c
mid-book -- an order of magnitude apart. The mid-range provably cannot clear its
round trip; the tails have not been tested.

`mark` is kept alongside `round` only as the upper bound, labelled as such.
Entry anchored at t+10ms with the no-chase guard, since that is the only entry
shown to be achievable.
"""
import sys, os, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
CACHE="/mnt/data/fccache"; SZ="/mnt/data/sizetape"
OUT="/var/lib/kalshi-study/ROUNDTRIP.json"
TAKER=0.07; SCALE=10000.0; DELAY=0.010
COIN=os.environ.get("RT_COIN","BTC")
TAGS=os.environ.get("RT_TAGS","_t0.1,_t0.5,_t1.0").split(",")
HOLDS=[1.0,5.0,30.0]
EDGES=[0.0,0.02,0.05,0.10,0.20,0.35,0.65,0.80,0.90,0.95,0.98,1.0]
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
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0)
    edge=np.abs(p)*m0; take=edge>(half+fee0); side=np.sign(p)
    n=len(p)
    A0=np.full(n,np.nan);B0=np.full(n,np.nan);AD=np.full(n,np.nan);BD=np.full(n,np.nan)
    MD=np.full(n,np.nan);SDv=np.full(n,np.nan);AV=np.full(n,np.nan)
    FW={h:np.full(n,np.nan) for h in HOLDS}; FS={h:np.full(n,np.nan) for h in HOLDS}
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
            sv=np.where(ok,(ba[jc]-bb[jc]),np.nan)
            FS[h][rows]=np.where(age<=h*0.25+0.005,sv,np.nan)
    nw=np.where(side>0,AD<=A0,BD>=B0)
    hm=SDv/2.0; fm=TAKER*MD*(1.0-MD)
    print(f"\n{'='*112}")
    print(f"{COIN}{TAG}  r2={r2:+.6f}  entry t+10ms, guarded, EXIT CHARGED  {nm} markets")
    print('='*112)
    for h in HOLDS:
        fwd=FW[h]; fsp=FS[h]
        hout=fsp/2.0; fout=TAKER*fwd*(1.0-fwd)
        mark =side*(fwd-MD)-hm-fm
        rnd  =mark-hout-fout
        print(f"\n hold={h}s   {'band':>12s} {'trades':>8s} {'cost_in':>8s} {'cost_out':>9s} "
              f"{'mark':>9s} {'ROUND':>9s} {'sig':>4s} {'95% CI round':>20s} {'depth':>6s}")
        for i in range(len(EDGES)-1):
            a,b=EDGES[i],EDGES[i+1]
            tap="*" if (b<=0.10 or a>=0.90) else " "
            g=(m0>=a)&(m0<b)&take&nw&np.isfinite(rnd)&np.isfinite(AV)
            if g.sum()<150: continue
            lo,hi,pv=boot(rnd[g],mc_te[g]); sig=(lo>0)==(hi>0)
            ci_in=100*(hm[g]+fm[g]).mean(); ci_out=100*(hout[g]+fout[g]).mean()
            print(f"          {f'{100*a:.0f}-{100*b:.0f}c':>12s}{tap}{int(g.sum()):>7,} "
                  f"{ci_in:>7.3f}c {ci_out:>8.3f}c {100*mark[g].mean():>+8.4f}c "
                  f"{100*rnd[g].mean():>+8.4f}c {('SIG' if sig else '-'):>4s} "
                  f"[{100*lo:+.4f},{100*hi:+.4f}] {np.median(AV[g]):>6,.0f}",flush=True)
            res[f"{COIN}{TAG}|{h}|{100*a:.0f}-{100*b:.0f}"]={"trades":int(g.sum()),
                "cost_in_c":ci_in,"cost_out_c":ci_out,
                "mark_c":100*mark[g].mean(),"round_c":100*rnd[g].mean(),
                "sig":bool(sig),"ci":[100*lo,100*hi],"p":pv,
                "median_depth":float(np.median(AV[g])),"tapered":bool(b<=0.10 or a>=0.90)}
    json.dump(res,open(OUT,"w"),indent=1,default=str)
print(f"\nwrote {OUT}   (* = tapered tick region)")
