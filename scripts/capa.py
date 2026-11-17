"""CAPACITY: how many contracts are actually available at the price we hit.

Everything to this point is priced PER CONTRACT at the quoted price. That is
only a strategy if the size exists. This reads touch-level resting size at the
entry anchor and asks how much of a desired clip would actually fill.

Operating point taken from the delay sweep: 10ms execution delay, the no-chase
guard, horizons 1s and 5s -- the cells that are significantly positive.

  buy  -> we lift the ask, so available = ask_size_1 at t+delay
  sell -> we hit the bid, so available = bid_size_1 at t+delay

Sizes are 1/100-contract units straight from the book, so /100 for contracts.

Filled clip is min(desired, available at the touch) -- a deliberately
CONSERVATIVE single-level model. It refuses to walk the book, which is right for
this strategy: the PnL was measured against the touch price, so eating deeper
levels would pay a worse price than anything measured here.
"""
import sys, os, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
COINS=os.environ.get("CAP_COINS","BTC,ETH,XRP,NEAR").split(",")
CACHE="/mnt/data/fccache"; SZ="/mnt/data/sizetape"
SUFFIX=os.environ.get("CAP_SUFFIX","")
OUT=f"/var/lib/kalshi-study/CAPACITY{SUFFIX.replace('.','p')}.json"
TAKER=0.07; SCALE=10000.0; DELAY=0.010
CLIPS=[10,50,100,500,1000,5000]
MKT_PER_DAY=96          # 15-minute markets per coin per day
res={}
def save():
    t=OUT+".tmp"; json.dump(res,open(t,"w"),indent=1,default=str); os.replace(t,OUT)

print(f"cache suffix={SUFFIX!r}  -> {OUT}",flush=True)
for coin in COINS:
    cf=f"{CACHE}/{coin}{SUFFIX}.npz"
    if not os.path.exists(cf): print(f"{coin}: no cache",flush=True); continue
    z=np.load(cf,allow_pickle=True)
    p=z["p"];m0=z["m0"];s0=z["s0"];t0=z["t0"];mc_te=z["mc_te"]
    munq=z["munq"];c_te=z["c_te"]
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0)
    take=(np.abs(p)*m0)>(half+fee0); side=np.sign(p)
    n=len(p)
    avail=np.full(n,np.nan); ask0=np.full(n,np.nan); bid0=np.full(n,np.nan)
    askD=np.full(n,np.nan); bidD=np.full(n,np.nan)
    have=0; miss=0
    for ci in c_te:
        f=f"{SZ}/sz_{str(munq[ci])}.parquet"
        if not os.path.exists(f): miss+=1; continue
        have+=1
        tt=pq.read_table(f,columns=["ts","best_bid","best_ask","bid_size_1","ask_size_1"])
        tts=tt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        bb=tt.column("best_bid").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        ba=tt.column("best_ask").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        bs=tt.column("bid_size_1").to_numpy(zero_copy_only=False).astype("float64")/100.0
        as_=tt.column("ask_size_1").to_numpy(zero_copy_only=False).astype("float64")/100.0
        o=np.argsort(tts); tts,bb,ba,bs,as_=tts[o],bb[o],ba[o],bs[o],as_[o]
        rows=np.flatnonzero(mc_te==ci)
        if len(rows)==0: continue
        tr=t0[rows]
        def at(tgt,arr):
            j=np.searchsorted(tts,tgt,side="right")-1
            ok=j>=0; j=np.clip(j,0,len(tts)-1)
            return np.where(ok,arr[j],np.nan)
        ask0[rows]=at(tr,ba); bid0[rows]=at(tr,bb)
        askD[rows]=at(tr+DELAY,ba); bidD[rows]=at(tr+DELAY,bb)
        sd=side[rows]
        avail[rows]=np.where(sd>0, at(tr+DELAY,as_), at(tr+DELAY,bs))
    print(f"\n=== {coin}: {have} test markets with a size tape, {miss} missing",flush=True)
    nw=np.where(side>0, askD<=ask0, bidD>=bid0)
    msk=take&nw&np.isfinite(avail)
    if msk.sum()<100: print(f"   too few rows ({int(msk.sum())})",flush=True); continue
    av=avail[msk]
    trades_per_mkt=msk.sum()/max(have,1)
    tpd=trades_per_mkt*MKT_PER_DAY
    print(f"   guarded trades={int(msk.sum()):,}  = {trades_per_mkt:.1f}/market "
          f"= {tpd:,.0f}/day",flush=True)
    qs=[1,5,10,25,50,75,90]
    print(f"   touch size available (contracts): " +
          "  ".join(f"p{q}={np.percentile(av,q):,.0f}" for q in qs),flush=True)
    res[f"{coin}|avail"]={"trades":int(msk.sum()),"trades_per_market":float(trades_per_mkt),
        "trades_per_day":float(tpd),"markets":have,
        "pctl":{f"p{q}":float(np.percentile(av,q)) for q in qs},
        "mean":float(av.mean()),"median":float(np.median(av))}
    print(f"   {'clip':>6s} {'%fully filled':>14s} {'avg filled':>11s} "
          f"{'$/day @0.30c':>13s} {'$/day @0.36c':>13s}",flush=True)
    for cl in CLIPS:
        filled=np.minimum(cl,av)
        full=float((av>=cl).mean())
        afl=float(filled.mean())
        print(f"   {cl:>6d} {100*full:>13.1f}% {afl:>11.1f} "
              f"{tpd*afl*0.0030:>12,.0f} {tpd*afl*0.0036:>12,.0f}",flush=True)
        res[f"{coin}|clip|{cl}"]={"frac_fully_filled":full,"avg_filled_contracts":afl,
            "usd_per_day_at_0.30c":float(tpd*afl*0.0030),
            "usd_per_day_at_0.36c":float(tpd*afl*0.0036)}
    save(); gc.collect()
save(); print(f"\nwrote {OUT}")
