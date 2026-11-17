"""CALIBRATION + LATENCY ANCHORS + EXTRA HORIZONS, off cached predictions.

Everything here reuses the SAME cached 0.1s-target predictions, so the trades
are identical to the fake-PnL table and nothing is retrained.

FAKE PnL, as defined for this project:
    side*(mid(t_e + h) - mid(t_e)) - half_spread(t_e) - 0.07*m(t_e)*(1-m(t_e))
one cost, charged once at entry, no exit of any kind. t_e is the ENTRY anchor:

  baseline  t_e = t                 decision and entry coincide (unachievable)
  delayed   t_e = t + 20ms          order always lands, 20ms late. The pure
                                    cost of latency.
  guarded   t_e = t + 20ms, and the trade happens ONLY if the touch has not
                                    moved against us: ask(t+20) <= ask(t) for a
                                    buy, bid(t+20) >= bid(t) for a sell. A
                                    no-chase rule -- it refuses to pay up.

CALIBRATION is reported on the MOVE, not on PnL. Subtracting the same cost from
predicted and realised inflates the ratio (the r/e column read 1.38 on BTC where
the move basis reads 1.11); the move basis is the honest one.

  slope of actual on predicted, via weighted least squares through the origin
  (sum(pred*act)/sum(pred^2)) -- 1.0 is perfect, <1 over-confident.

Two conditionings, and only one of them is legitimate:
  pred>cost   conditions on the FORECAST. Causal, unbiased.
  act>half    conditions on the OUTCOME. **BIASED UPWARD BY CONSTRUCTION** --
              selecting rows where the actual move was large guarantees
              actual/predicted > 1 regardless of model quality. Reported because
              it was asked for, labelled because it cannot be read as skill.
"""
import sys, os, re, glob, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pyarrow.parquet as pq
COINS=os.environ.get("CAL_COINS","BTC,ETH,XRP,NEAR").split(",")
CACHE="/mnt/data/fccache"; TT="/mnt/data/touchtape"
OUT="/var/lib/kalshi-study/CALIB.json"
TAKER=0.07; SCALE=10000.0; LAT=0.020
HOR=[0.1,1.0,2.0,5.0,10.0,20.0,30.0]
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
    cf=f"{CACHE}/{coin}.npz"
    if not os.path.exists(cf): print(f"{coin}: no cache yet",flush=True); continue
    z=np.load(cf,allow_pickle=True)
    p=z["p"];m0=z["m0"];s0=z["s0"];t0=z["t0"];mc_te=z["mc_te"]
    munq=z["munq"];c_te=z["c_te"];r2=float(z["r2"])
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0); cost0=half+fee0
    edge=np.abs(p)*m0; take=edge>cost0; side=np.sign(p)
    print(f"\n=== {coin}  r2(0.1s)={r2:+.6f}  {int(take.sum()):,} gated of {len(p):,}",flush=True)

    # walk each test market's touch tape once: forward mids at every horizon,
    # plus the touch at t and at t+20ms
    n=len(p)
    FW={h:np.full(n,np.nan) for h in HOR}
    FWL={h:np.full(n,np.nan) for h in HOR}      # anchored at t+LAT
    ask0=np.full(n,np.nan); bid0=np.full(n,np.nan)
    askL=np.full(n,np.nan); bidL=np.full(n,np.nan)
    midL=np.full(n,np.nan); sprL=np.full(n,np.nan)
    miss=0
    for ci in c_te:
        f=f"{TT}/tt_{str(munq[ci])}.parquet"
        if not os.path.exists(f): miss+=1; continue
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
            ok=j>=0
            j=np.clip(j,0,len(tts)-1)
            return np.where(ok,arr[j],np.nan)
        ask0[rows]=at(tr,ba); bid0[rows]=at(tr,bb)
        askL[rows]=at(tr+LAT,ba); bidL[rows]=at(tr+LAT,bb)
        midL[rows]=at(tr+LAT,tm); sprL[rows]=at(tr+LAT,ba)-at(tr+LAT,bb)
        for h in HOR:
            # forward mid must be a real observation at or before the target,
            # and not stale by more than 25% of the horizon
            j=np.searchsorted(tts,tr+h,side="right")-1
            ok=j>=0; jc=np.clip(j,0,len(tts)-1)
            v=np.where(ok,tm[jc],np.nan); age=np.where(ok,(tr+h)-tts[jc],np.nan)
            FW[h][rows]=np.where(age<=h*0.25+0.005,v,np.nan)
            j2=np.searchsorted(tts,tr+LAT+h,side="right")-1
            ok2=j2>=0; j2c=np.clip(j2,0,len(tts)-1)
            v2=np.where(ok2,tm[j2c],np.nan); age2=np.where(ok2,(tr+LAT+h)-tts[j2c],np.nan)
            FWL[h][rows]=np.where(age2<=h*0.25+0.005,v2,np.nan)
    print(f"   {miss} test markets had no touch tape",flush=True)

    # ---------------- anchors ----------------
    halfL=sprL/2.0; feeL=TAKER*midL*(1.0-midL)
    noworse=np.where(side>0, askL<=ask0, bidL>=bid0)
    print(f"\n{'anchor':9s} {'horiz':>6s} {'trades':>8s} {'take%':>7s} {'PnL/tr':>9s} "
          f"{'p':>7s} {'sig':>4s} {'95% CI':>22s}")
    for anc in ("baseline","delayed","guarded"):
        for h in HOR:
            if anc=="baseline":
                msk=take&np.isfinite(FW[h])&np.isfinite(m0)
                pnl=side*(FW[h]-m0)-cost0
            else:
                base=take&np.isfinite(FWL[h])&np.isfinite(midL)&np.isfinite(sprL)&(sprL>=0)
                msk=base if anc=="delayed" else (base&noworse)
                pnl=side*(FWL[h]-midL)-halfL-feeL
            msk=msk&np.isfinite(pnl)
            if msk.sum()<100: continue
            v=pnl[msk]; g=mc_te[msk]
            lo,hi,pv=boot(v,g); sig=(lo>0)==(hi>0)
            print(f"{anc:9s} {h:>6.1f} {int(msk.sum()):>8,} {100*msk.sum()/len(p):>6.2f}% "
                  f"{100*v.mean():>+8.4f}c {pv:>7.4f} {('SIG' if sig else '-'):>4s} "
                  f"[{100*lo:+.4f},{100*hi:+.4f}]c",flush=True)
            res[f"{coin}|{anc}|{h}"]={"trades":int(msk.sum()),
                "take_rate":float(msk.sum()/len(p)),"pnl_cents":float(100*v.mean()),
                "ci_cents":[float(100*lo),float(100*hi)],"p":pv,"sig":bool(sig)}
    res[f"{coin}|guard_rate"]={"noworse_frac_of_gated":float(noworse[take].mean())}
    print(f"   guard passes on {100*noworse[take].mean():.2f}% of gated trades",flush=True)

    # ---------------- calibration on the MOVE ----------------
    print(f"\ncalibration (move basis), h=0.1s and 1.0s")
    print(f"{'h':>5s} {'subset':11s} {'n':>9s} {'pred':>9s} {'actual':>9s} "
          f"{'act/pred':>9s} {'slope':>8s}")
    for h in (0.1,1.0):
        pm=p*m0                       # signed predicted move, dollars
        am=FW[h]-m0                   # signed actual move
        base=np.isfinite(pm)&np.isfinite(am)
        subs=[("all",base),
              ("pred>cost",base&take),
              ("act>half",base&(np.abs(am)>half))]   # BIASED: selects on outcome
        for nm,msk in subs:
            if msk.sum()<100: continue
            a=am[msk]; q=pm[msk]
            sl=float((q*a).sum()/max((q*q).sum(),1e-18))
            print(f"{h:>5.1f} {nm:11s} {int(msk.sum()):>9,} "
                  f"{100*np.abs(q).mean():>8.4f}c {100*np.abs(a).mean():>8.4f}c "
                  f"{np.abs(a).mean()/max(np.abs(q).mean(),1e-18):>9.3f} {sl:>8.3f}"
                  + ("   <- biased, selects on outcome" if nm=="act>half" else ""),flush=True)
            res[f"{coin}|calib|{h}|{nm}"]={"n":int(msk.sum()),
                "pred_cents":float(100*np.abs(q).mean()),
                "act_cents":float(100*np.abs(a).mean()),
                "ratio":float(np.abs(a).mean()/max(np.abs(q).mean(),1e-18)),
                "slope_through_origin":sl}
    # decile shape on gated-eligible rows, h=0.1s
    pm=p*m0; am=FW[0.1]-m0; b=np.isfinite(pm)&np.isfinite(am)
    q=pm[b]; a=am[b]
    dec=np.clip(((np.argsort(np.argsort(np.abs(q)))/max(len(q)-1,1))*10).astype(int),0,9)
    print(f"\npredicted-magnitude deciles, h=0.1s (all test rows)")
    print(f"{'dec':>4s} {'n':>9s} {'pred':>9s} {'actual':>9s} {'act/pred':>9s}")
    for dd in range(10):
        s_=dec==dd
        if s_.sum()<50: continue
        print(f"{dd:>4d} {int(s_.sum()):>9,} {100*np.abs(q[s_]).mean():>8.4f}c "
              f"{100*(np.sign(q[s_])*a[s_]).mean():>8.4f}c "
              f"{(np.sign(q[s_])*a[s_]).mean()/max(np.abs(q[s_]).mean(),1e-18):>9.3f}",flush=True)
    save(); gc.collect()
save(); print(f"\nwrote {OUT}")
