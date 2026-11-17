"""PRICE-BUCKET BREAKDOWN, with the tapered tick regions called out.

Kalshi's grid is tapered: tick = 0.0010 on [0,0.10) and [0.90,1.00], 0.0100 on
[0.10,0.90). So the tails can quote a 0.1c spread, and the taker fee
0.07*m*(1-m) also collapses there (0.333c at m=0.05, 0.137c at m=0.02, vs
1.750c at m=0.50). Cost is therefore an order of magnitude cheaper in the tails.

The question is whether EDGE survives there, or shrinks faster than cost does.
Reported per bucket: cost decomposition, gate pass rate, fake PnL with and
without latency, touch depth, and the share of total dollars.

Buckets are on the ENTRY mid, which is observable before trading -- this is a
cost/venue-structure split, not conditioning the forecast on price.
"""
import sys, os, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
CACHE="/mnt/data/fccache"; SZ="/mnt/data/sizetape"
OUT="/var/lib/kalshi-study/BUCKETS.json"
TAKER=0.07; SCALE=10000.0; DELAY=0.010
COIN=os.environ.get("BK_COIN","BTC")
TAGS=os.environ.get("BK_TAGS","_t1.0,").split(",")
HOLD=float(os.environ.get("BK_HOLD","5.0"))
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
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0); cost0=half+fee0
    edge=np.abs(p)*m0; take=edge>cost0; side=np.sign(p)
    n=len(p)
    A0=np.full(n,np.nan);B0=np.full(n,np.nan)
    AD=np.full(n,np.nan);BD=np.full(n,np.nan);MD=np.full(n,np.nan);SDv=np.full(n,np.nan)
    AV=np.full(n,np.nan);FWD=np.full(n,np.nan);FW0=np.full(n,np.nan)
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
        j=np.searchsorted(tts,tr+DELAY+HOLD,side="right")-1
        ok=j>=0; jc=np.clip(j,0,len(tts)-1)
        v=np.where(ok,tm[jc],np.nan); age=np.where(ok,(tr+DELAY+HOLD)-tts[jc],np.nan)
        FWD[rows]=np.where(age<=HOLD*0.25+0.005,v,np.nan)
        j0=np.searchsorted(tts,tr+HOLD,side="right")-1
        ok0=j0>=0; j0c=np.clip(j0,0,len(tts)-1)
        v0=np.where(ok0,tm[j0c],np.nan); a0=np.where(ok0,(tr+HOLD)-tts[j0c],np.nan)
        FW0[rows]=np.where(a0<=HOLD*0.25+0.005,v0,np.nan)
    nw=np.where(side>0,AD<=A0,BD>=B0)
    hm=SDv/2.0; fm=TAKER*MD*(1.0-MD)
    pnl_lat=side*(FWD-MD)-hm-fm
    pnl_0  =side*(FW0-m0)-half-fee0
    print(f"\n{'='*118}")
    print(f"{COIN}{TAG}  r2={r2:+.6f}  hold={HOLD}s  delay={1000*DELAY:.0f}ms  "
          f"{nm} test markets  (tick: 0.1c on [0,10)&[90,100], 1.0c on [10,90))")
    print('='*118)
    print(f"{'price band':>13s} {'rows':>9s} {'gated':>8s} {'take%':>6s} {'spr':>7s} "
          f"{'fee':>7s} {'cost':>7s} {'edge':>7s} {'depth':>7s} "
          f"{'PnL 0ms':>9s} {'sig':>4s} {'PnL 10ms':>9s} {'sig':>4s} {'$share':>7s}")
    tot=0.0; rows_out=[]
    for i in range(len(EDGES)-1):
        a,b=EDGES[i],EDGES[i+1]
        inb=(m0>=a)&(m0<b)
        g0=inb&take&np.isfinite(pnl_0)
        gl=inb&take&nw&np.isfinite(pnl_lat)&np.isfinite(AV)
        if inb.sum()<50: continue
        t0c=f"{100*a:.0f}-{100*b:.0f}c"
        tap="*" if (b<=0.10 or a>=0.90) else " "
        if g0.sum()>=100:
            lo,hi,pv=boot(pnl_0[g0],mc_te[g0]); s0g=(lo>0)==(hi>0)
            p0=100*pnl_0[g0].mean(); d0=pnl_0[g0].sum()
        else: p0=float("nan"); s0g=False; d0=0.0
        if gl.sum()>=100:
            lo2,hi2,pv2=boot(pnl_lat[gl],mc_te[gl]); slg=(lo2>0)==(hi2>0)
            pl=100*pnl_lat[gl].mean()
        else: pl=float("nan"); slg=False
        tot+=max(d0,0)
        rows_out.append((t0c,tap,int(inb.sum()),int(g0.sum()),100*g0.sum()/max(inb.sum(),1),
            100*s0[inb].mean(),100*fee0[inb].mean(),100*cost0[inb].mean(),
            100*edge[g0].mean() if g0.sum()>0 else float("nan"),
            np.median(AV[gl]) if gl.sum()>0 else float("nan"),p0,s0g,pl,slg,d0))
    for (t0c,tap,nin,ng,tk,sp,fe,co,ed,dp,p0,s0g,pl,slg,d0) in rows_out:
        print(f"{t0c:>12s}{tap} {nin:>9,} {ng:>8,} {tk:>5.2f}% {sp:>6.3f}c "
              f"{fe:>6.3f}c {co:>6.3f}c {ed:>6.3f}c {dp:>7,.0f} "
              f"{p0:>+8.4f}c {('SIG' if s0g else '-'):>4s} {pl:>+8.4f}c "
              f"{('SIG' if slg else '-'):>4s} {100*d0/max(tot,1e-9):>6.1f}%")
        res[f"{COIN}{TAG}|{t0c}"]={"rows":nin,"gated":ng,"take_pct":tk,"spread_c":sp,
            "fee_c":fe,"cost_c":co,"edge_c":ed,"median_depth":float(dp) if dp==dp else None,
            "pnl_0ms_c":p0,"sig_0ms":bool(s0g),"pnl_10ms_c":pl,"sig_10ms":bool(slg),
            "dollar_share_pct":100*d0/max(tot,1e-9)}
    print("  * = tapered tick region (0.1c tick)")
json.dump(res,open(OUT,"w"),indent=1,default=str); print(f"\nwrote {OUT}")
