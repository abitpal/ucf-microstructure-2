"""TRADEABILITY: train on the 0.1s move, trade when edge > cost, PnL by horizon.

STRATEGY (the existing one, unchanged):
  at every book delta, if |predicted move| > (half spread + taker fee), cross and
  take the predicted direction. No hedge, no exit rule -- hold to the horizon.
  The 0.02->0.1s work was only ever a signal-strength probe; the model traded
  here predicts the 0.1s move from t.

PREDICTION   y = (mid(t+0.1) - mid(t)) / mid(t)       mid-to-mid, /mid
COST         half spread at entry, plus Kalshi's taker fee
                 fee = 0.07 * C * P * (1-P)            (TAKER_MULTIPLIER=0.07)
             charged on entry. Two exits are reported because they bracket
             reality and this project has already conflated them once:
               mark   exit marked at the mid       -- no second crossing
               round  exit crosses the book again  -- pays the half spread twice
             The truth for a real exit is between them, nearer `round`.

HORIZONS     0.1s, 1s, 2s, 5s, 30s, and settlement.
             Settlement uses the market's LAST observed mid as the proxy for
             the 0/1 outcome -- the panels do not carry the settled result, so
             this is an approximation and is labelled as one.

Forward mids come from the full-resolution mid tape, so a 0.1s horizon is 0.1s
and not "the next sampled row" (median inter-row gap is 22-40ms).
"""
import sys, os, re, glob, json, gc, hashlib; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from pathlib import Path
import torch.nn as nn
from tools.panel_model import scale_free, r2_zero
torch.set_num_threads(int(os.environ.get("TC_THREADS","12")))
COINS=os.environ.get("TC_COINS","BTC,ETH,XRP,NEAR").split(",")
PANELS=os.environ["REPRO_PANELS"]; MIDS=os.environ.get("REPRO_MIDS", "/var/lib/kalshi-study/midtape")
REPRO_OUT=Path(os.environ["REPRO_OUTPUT_DIR"]); REPRO_OUT.mkdir(parents=True,exist_ok=True)
OUT=str(REPRO_OUT / "tradecheck_reproduction.json")
HOR=[0.1,1.0,2.0,5.0,30.0]
TAKER=0.07
W,L,DROP=640,5,0.2; SEEDS=(11,23); BS=1024; MAXROWS=1_000_000
SCALE=10000.0          # panel prices are deci-cents

res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT))
    except Exception: res={}
def save():
    t=OUT+".tmp"; json.dump(res,open(t,"w"),indent=1,default=str); os.replace(t,OUT)

class Net(nn.Module):
    def __init__(s,nin):
        super().__init__()
        q=[nn.Linear(nin,W),nn.ReLU(),nn.Dropout(DROP)]
        for _ in range(L): q+=[nn.Linear(W,W),nn.ReLU(),nn.Dropout(DROP)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(W,1)
    def forward(s,x): return s.head(s.body(x))
def fit(X,y,w,Xv,yv,wv,seed,steps=4000,lr=1e-3,wd=0.3,pat=10):
    torch.manual_seed(seed); m=Net(X.shape[1])
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=lr,total_steps=steps,pct_start=.15)
    A=torch.from_numpy(X);T=torch.from_numpy(y).float().view(-1,1)
    Wq=torch.from_numpy(w).float().view(-1,1);V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//20)
    while st<steps and bad<pat:
        pm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=pm[i:i+BS];o.zero_grad()
            ((Wq[ix]*(m(A[ix])-T[ix])**2).sum()/Wq[ix].sum()).backward();o.step()
            if st<steps-1: sc.step()
            st+=1
            if st%ev==0 or st>=steps:
                m.eval()
                with torch.no_grad(): v=r2_zero(yv,m(V).numpy().ravel(),wv)
                if v>best[0]: best=(v,[p.detach().clone() for p in m.parameters()]);bad=0
                else: bad+=1
                if bad>=pat or st>=steps: break
    if best[1] is not None:
        with torch.no_grad():
            for p,b in zip(m.parameters(),best[1]): p.copy_(b)
    m.eval(); return m
def pred(m,A,ch=1_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy().ravel())
    return np.concatenate(o)
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                                  "SEP","OCT","NOV","DEC"])}
def tkey(t):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",t)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)

def load(coin):
    """rows: features, mid, spread, forward mids at every horizon, settlement proxy."""
    paths=sorted(glob.glob(f"{PANELS}/rich_{coin}/panel_KX{coin}15M-*.parquet"))
    if not paths: return None
    # Preserve the exact input sequence and identify full-resolution midpoint tapes.
    sequence=[]
    for path in paths:
        name=os.path.basename(path)
        midpoint=os.path.join(MIDS,"mid_"+name[len("panel_"):])
        item={"panel":name,"midpoint":os.path.basename(midpoint),"present":os.path.exists(midpoint)}
        if item["present"]:
            digest=hashlib.sha256()
            with open(midpoint,"rb") as f:
                for block in iter(lambda:f.read(4*1024*1024),b""): digest.update(block)
            item["midpoint_sha256"]=digest.hexdigest()
        sequence.append(item)
    with open(REPRO_OUT/f"{coin}_input_sequence.json","w") as f: json.dump(sequence,f,indent=2)
    used={x["midpoint"] for x in sequence if x["present"]}
    available={os.path.basename(x) for x in glob.glob(os.path.join(MIDS,f"mid_KX{coin}15M-*.parquet"))}
    missing=sorted(available-used)
    with open(REPRO_OUT/f"{coin}_midpoints_without_panels.json","w") as f: json.dump(missing,f,indent=2)
    print(f"{coin}: {len(missing)} available midpoint tapes lack corresponding panels",flush=True)
    probe=paths[::max(1,len(paths)//12)][:12]; union=set()
    for q in probe:
        rq=pd.read_parquet(q)
        kq=[c for c in rq.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sq=scale_free(rq[kq])
        union|={c for c in sq.columns if not c.startswith(("y_","ry_","market","_day"))}
        del rq,sq
    feats=sorted(union)
    print(f"   {coin}: {len(paths)} panels, {len(feats)} features "
          f"({sum(1 for c in feats if c.startswith(('p_','xmkt')))} perp/cross)",flush=True)
    F=[];MID=[];SPR=[];FW={h:[] for h in HOR};SET=[];MKT=[];skip=0
    for i,p in enumerate(paths,1):
        tk=re.sub(r"^panel_|\.parquet$","",os.path.basename(p))
        mp=f"{MIDS}/mid_{tk}.parquet"
        if not os.path.exists(mp): skip+=1; continue
        raw=pd.read_parquet(p)
        if "market" not in raw.columns: raw["market"]=tk
        keep=[c for c in raw.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sf=scale_free(raw[keep])
        A=np.empty((len(raw),len(feats)),dtype="float32")
        for j,c in enumerate(feats):
            A[:,j]=sf[c].to_numpy(dtype="float32") if c in sf.columns else np.nan
        ts=raw["ts"].to_numpy(dtype="float64")
        mid=raw["k_mid"].to_numpy(dtype="float64")/SCALE
        spr=raw["k_spread"].to_numpy(dtype="float64")/SCALE
        mt=pq.read_table(mp,columns=["ts","mid"])
        mts=mt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        mmd=mt.column("mid").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        o=np.argsort(mts); mts,mmd=mts[o],mmd[o]
        def at(lag):
            j=np.searchsorted(mts,ts+lag,side="left"); ok=j<len(mts)
            jc=np.clip(j,0,len(mts)-1)
            v=np.where(ok,mmd[jc],np.nan)
            got=np.where(ok,mts[jc]-ts,np.nan)
            return np.where(got<=lag*1.25+0.005,v,np.nan)
        F.append(A);MID.append(mid);SPR.append(spr)
        for h in HOR: FW[h].append(at(h))
        SET.append(np.full(len(ts),mmd[-1]))          # settlement proxy: last mid
        MKT.append(raw["market"].to_numpy(dtype=object))
        del raw,sf,A; gc.collect()
        if i%300==0: print(f"      {i}/{len(paths)}",flush=True)
    if not F: return None
    print(f"   {coin}: {skip} markets had no mid tape",flush=True)
    return (np.vstack(F),np.concatenate(MID),np.concatenate(SPR),
            {h:np.concatenate(v) for h,v in FW.items()},
            np.concatenate(SET),np.concatenate(MKT),feats)

print(f"{'coin':5s} {'horiz':>8s} {'trades':>9s} {'hit':>6s} "
      f"{'mark c/tr':>9s} {'round c/tr':>10s} {'mark bp':>10s} {'round bp':>10s} "
      f"{'r/e':>6s} {'sig':>4s} {'round 95% CI':>20s}")
for coin in COINS:
    if f"{coin}|0.1" in res: print(f"{coin}: cached"); continue
    L_=load(coin)
    if L_ is None: raise RuntimeError(f"No panel and midpoint data for {coin}: {PANELS}, {MIDS}")
    X,mid,spr,FW,SETT,mkt,feats=L_
    mc,munq=pd.factorize(pd.Series(mkt),sort=True); mc=mc.astype("int32")
    munq=np.asarray(munq,dtype=object)
    tko=np.array([tkey(m) for m in munq],dtype="int64")
    order=np.argsort(tko); nte=int(len(order)*0.30)
    c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mc,cs))
    # target: the 0.1s move, mid-to-mid, divided by the current mid
    y=np.where(mid>0,(FW[0.1]-mid)/mid,np.nan)
    ok=np.isfinite(y)&np.isfinite(mid)&np.isfinite(spr)&(mid>0)
    i_te=sel(c_te)[ok[sel(c_te)]]; i_ho=sel(c_ho)[ok[sel(c_ho)]]; i_tr=sel(c_tr)[ok[sel(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"{coin:5s} too few rows"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    def prep(ix,lo=None,hi=None,mu=None,sg=None):
        A=X[ix].copy()
        if lo is None:
            lo=np.nanquantile(A[:300000],.01,axis=0); hi=np.nanquantile(A[:300000],.99,axis=0)
        np.clip(A,lo,hi,out=A)
        if mu is None: mu=np.nanmean(A,0); sg=np.nanstd(A,0); sg[sg==0]=1.0
        A-=mu; A/=sg; np.nan_to_num(A,copy=False)
        return A,lo,hi,mu,sg
    XA,lo,hi,mu,sg=prep(idxA)
    Xh,_,_,_,_=prep(i_ho,lo,hi,mu,sg); Xte,_,_,_,_=prep(i_te,lo,hi,mu,sg)
    q=np.nanquantile(y[i_tr],[.01,.99])
    yA=np.clip(y[idxA],q[0],q[1]); ym,ys=float(yA.mean()),float(yA.std()) or 1.0
    wA=np.ones(len(idxA))/len(idxA); wh=np.ones(len(i_ho))/len(i_ho)
    wte=np.ones(len(i_te))/len(i_te)
    # Preserve the exact feature ordering, preprocessing, model weights and split.
    # This loop is equivalent to the source list comprehension; only saving is new.
    ps=[]
    for s_ in SEEDS:
        model=fit(XA,((yA-ym)/ys).astype("float32"),wA,Xh,
                  ((y[i_ho]-ym)/ys),wh,s_)
        ps.append(pred(model,Xte)*ys+ym)
        torch.save(model.state_dict(),REPRO_OUT/f"{coin}_seed{s_}.pt")
        del model
    p=np.mean(ps,axis=0)
    np.savez_compressed(REPRO_OUT/f"{coin}_preprocessing.npz",lo=lo,hi=hi,mu=mu,sg=sg,ym=ym,ys=ys,features=np.asarray(feats,dtype=str))
    split={"train":munq[c_tr].tolist(),"validation":munq[c_ho].tolist(),"test":munq[c_te].tolist(),
           "train_rows_available":len(i_tr),"train_rows_sampled":len(idxA),"validation_rows":len(i_ho),"test_rows":len(i_te),
           "feature_count":len(feats),"features":feats,"seed":0,"model_seeds":list(SEEDS),"target":"(mid(t+0.1)-mid(t))/mid(t)",
           "panel_path":PANELS,"midtape_path":MIDS,"sampling_note":"train indices refer to the input concatenation order"}
    with open(REPRO_OUT/f"{coin}_split.json","w") as f: json.dump(split,f,indent=2)
    np.savez_compressed(REPRO_OUT/f"{coin}_predictions.npz",prediction=p,target=y[i_te],mid=mid[i_te],spread=spr[i_te],market=munq[mc[i_te]].astype(str),train_sample_indices=idxA,validation_row_indices=i_ho,test_row_indices=i_te)

    r2=float(r2_zero(y[i_te],p,wte))
    m0=mid[i_te]; s0=spr[i_te]; mc_te=mc[i_te]
    half=s0/2.0
    fee=TAKER*m0*(1.0-m0)                     # per contract, Kalshi taker
    cost=half+fee
    edge=np.abs(p)*m0                         # predicted move in dollars
    take=edge>cost
    side=np.sign(p)
    for h in HOR+["settle"]:
        fwd=SETT[i_te] if h=="settle" else FW[h][i_te]
        good=take&np.isfinite(fwd)
        n=int(good.sum())
        if n==0:
            print(f"{coin:5s} {str(h):>10s} {0:>10} {'-':>6} {'-':>6} {'-':>10} {'-':>11} {'-':>11}")
            continue
        realized=side[good]*(fwd[good]-m0[good])
        mark=realized-cost[good]
        # ROUND TRIP OWES TWO FEES, NOT ONE.
        #
        # Kalshi charges the taker fee on EVERY trade, so closing costs another
        # half spread AND another fee. The first version subtracted only the
        # second half spread, which understated the exit by the larger of the two
        # components (mean fee 1.13c against 0.40c of half spread) and made every
        # horizon look profitable. The exit fee is computed at the EXIT mid,
        # since 0.07*P*(1-P) depends on the price actually traded.
        if h=="settle":
            # settlement is not a trade: the contract resolves to 0 or 1, so
            # there is no second crossing and no second fee.
            rnd=mark.copy()
        else:
            fee_out=TAKER*fwd[good]*(1.0-fwd[good])
            rnd=realized-cost[good]-(half[good]+fee_out)
        hit=float((realized>0).mean())
        # --- PER NOTIONAL. Capital at risk is the price actually paid: a long
        #     YES costs mid+half, a short YES is a long NO at (1-mid)+half. PnL
        #     divided by that is return on capital deployed, which is the only
        #     way the horizons and the coins are comparable -- a 2c gain on a 5c
        #     contract is not the same trade as 2c on a 50c one.
        cap=np.where(side[good]>0, m0[good]+half[good], (1.0-m0[good])+half[good])
        rn_mark=mark/np.maximum(cap,1e-9)
        rn_rnd =rnd /np.maximum(cap,1e-9)
        # --- PREDICTED vs REALISED. The model's own expectation for each taken
        #     trade is |pred|*mid - cost. Comparing it with what actually
        #     happened is a calibration test: a ratio near 1 means the edge is
        #     real and correctly sized, far below 1 means the model is
        #     over-confident exactly where it chooses to trade.
        exp_mark=edge[good]-cost[good]
        ratio=float(mark.mean()/exp_mark.mean()) if exp_mark.mean()!=0 else float("nan")
        # --- SIGNIFICANCE, bootstrapped over test MARKETS (not rows). Rows
        #     inside a market are heavily dependent, so a row bootstrap would
        #     overstate precision by orders of magnitude.
        def mkt_boot(v,B=2000,seed=0):
            g=mc_te[good]; u,inv=np.unique(g,return_inverse=True)
            sm=np.bincount(inv,weights=v,minlength=len(u))
            ct=np.bincount(inv,minlength=len(u)).astype("float64")
            rng=np.random.default_rng(seed)
            pk=rng.integers(0,len(u),size=(B,len(u)))
            d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
            return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
                    float(min((d<=0).mean(),(d>=0).mean())*2))
        lo_m,hi_m,p_m=mkt_boot(mark)
        lo_r,hi_r,p_r=mkt_boot(rnd)
        sig_m=(lo_m>0)==(hi_m>0); sig_r=(lo_r>0)==(hi_r>0)
        nmk=len(np.unique(mc_te[good]))
        print(f"{coin:5s} {str(h):>8s} {n:>9,} {100*hit:>5.1f}% "
              f"{100*mark.mean():>+8.4f}c {100*rnd.mean():>+9.4f}c "
              f"{1e4*rn_mark.mean():>+9.1f}bp {1e4*rn_rnd.mean():>+9.1f}bp "
              f"{ratio:>6.2f} {('SIG' if sig_r else '-'):>4s} "
              f"[{100*lo_r:+.4f},{100*hi_r:+.4f}]c mk={nmk}",flush=True)
        res[f"{coin}|{h}"]={"r2_0p1s":r2,"trades":n,"rate":float(n/len(m0)),
            "hit":hit,"mark_cents_per_trade":float(100*mark.mean()),
            "round_cents_per_trade":float(100*rnd.mean()),
            "mark_total_dollars":float(mark.sum()),
            "mark_bp_of_notional":float(1e4*rn_mark.mean()),
            "round_bp_of_notional":float(1e4*rn_rnd.mean()),
            "mean_capital_cents":float(100*cap.mean()),
            "expected_mark_cents":float(100*exp_mark.mean()),
            "realised_over_expected":ratio,
            "round_ci_cents":[float(100*lo_r),float(100*hi_r)],
            "round_p":p_r,"round_sig":bool(sig_r),
            "mark_ci_cents":[float(100*lo_m),float(100*hi_m)],
            "mark_p":p_m,"mark_sig":bool(sig_m),
            "boot_markets":int(nmk),
            # means over the SELECTED trades, not the whole test set -- the
            # population means cannot audit per-trade arithmetic
            "sel_half_spread_cents":float(100*half[good].mean()),
            "sel_fee_in_cents":float(100*fee[good].mean()),
            "sel_edge_cents":float(100*edge[good].mean()),
            "sel_realized_cents":float(100*realized.mean()),
            "pop_half_spread_cents":float(100*half.mean()),
            "pop_fee_cents":float(100*fee.mean()),
            "nfeat":len(feats),"test_markets":len(c_te)}
    save(); del X,XA,Xh,Xte; gc.collect()
save(); print(f"\nwrote {OUT}")
