"""SETTLEMENT PnL: train on the settled outcome, hold to expiry, pay ONE fee.

WHY THIS EXISTS
---------------
`tc.py` trained on the 0.1s mid move and then reported PnL at horizons out to
settlement. That is a 0.1-second forecast held for up to 15 minutes, and the
edge half-life on this tape is ~300ms, so the settlement column was measuring
decay, not strategy. If the plan is to hold to expiry -- which is the only path
that pays a single taker fee, because a contract that resolves is never traded
out of -- then the model has to predict the thing being held to.

TARGET       y = (outcome - mid(t)) / mid(t)      outcome in {0,1}, the REAL
             settled result from GET /markets/{ticker} (`result` yes/no), not
             the last-observed-mid proxy the earlier run used.

STRATEGY     edge = |pred| * mid                 predicted move in dollars
             gate   edge > half_spread + 0.07*m*(1-m)
             side   sign(pred)
             PnL  = side*(outcome - mid) - half_spread - fee        ONE fee.

THREE THINGS THIS CONTROLS FOR, each of which has burned this project before
--------------------------------------------------------------------------
1. THE LABEL IS CONSTANT WITHIN A MARKET. Every row of a market carries the
   same 0/1, so the effective sample size is the MARKET COUNT (~1.1k/coin),
   not the row count (~1M). Rows are weighted 1/n_rows_in_market so a busy
   market cannot outvote a quiet one, and every CI bootstraps over markets.
2. CAPACITY. A 640x5 net on ~770 effective training points will memorise.
   The bracket (deep / small / ridge) is fixed in advance and ALL THREE are
   reported, so this cannot turn into a search for whichever one looks best.
3. FORCED-SIDE CONTROLS. Settlement on cheap contracts carries a static bias
   that a constant direction can harvest without any model at all -- it beat
   the model outright in the sports work. `long`/`short` run the same gate and
   the same costs with the side held constant; if the model does not beat both,
   there is no signal here, only the bias.
"""
import sys, os, re, glob, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn
from tools.panel_model import scale_free, r2_zero
torch.set_num_threads(int(os.environ.get("SC_THREADS","12")))
COINS=os.environ.get("SC_COINS","BTC,ETH,XRP,NEAR").split(",")
PANELS="/mnt/data/panels"; MIDS="/var/lib/kalshi-study/midtape"
SETTLE="/var/lib/kalshi-study/SETTLE.json"
OUT="/var/lib/kalshi-study/SETTLECHECK.json"
TAKER=0.07; SCALE=10000.0
BS=1024; MAXROWS=1_000_000; SEEDS=(11,23)
HOLD=[1.0,30.0]            # short holds, for contrast only -- these pay 2 fees
MODELS=[("deep",640,5,0.2,0.3),("small",96,2,0.1,3.0),("ridge",0,0,0.0,0.0)]

OUTC={}
for tk,v in json.load(open(SETTLE)).items():
    r=v.get("result")
    if r=="yes": OUTC[tk]=1.0
    elif r=="no": OUTC[tk]=0.0
print(f"{len(OUTC)} markets with a real settled outcome",flush=True)

res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT))
    except Exception: res={}
def save():
    t=OUT+".tmp"; json.dump(res,open(t,"w"),indent=1,default=str); os.replace(t,OUT)

class Net(nn.Module):
    def __init__(s,nin,W,L,DROP):
        super().__init__()
        q=[nn.Linear(nin,W),nn.ReLU(),nn.Dropout(DROP)]
        for _ in range(L): q+=[nn.Linear(W,W),nn.ReLU(),nn.Dropout(DROP)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(W,1)
    def forward(s,x): return s.head(s.body(x))

def fit(X,y,w,Xv,yv,wv,seed,W,L,DROP,wd,steps=4000,lr=1e-3,pat=10):
    torch.manual_seed(seed); m=Net(X.shape[1],W,L,DROP)
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
    paths=sorted(glob.glob(f"{PANELS}/rich_{coin}/panel_KX{coin}15M-*.parquet"))
    if not paths: return None
    probe=paths[::max(1,len(paths)//12)][:12]; union=set()
    for q in probe:
        rq=pd.read_parquet(q)
        kq=[c for c in rq.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sq=scale_free(rq[kq])
        union|={c for c in sq.columns if not c.startswith(("y_","ry_","market","_day"))}
        del rq,sq
    feats=sorted(union)
    print(f"   {coin}: {len(paths)} panels, {len(feats)} features",flush=True)
    F=[];MID=[];SPR=[];FW={h:[] for h in HOLD};SET=[];MKT=[];skip=0;noout=0
    for i,p in enumerate(paths,1):
        tk=re.sub(r"^panel_|\.parquet$","",os.path.basename(p))
        if tk not in OUTC: noout+=1; continue
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
        for h in HOLD: FW[h].append(at(h))
        SET.append(np.full(len(ts),OUTC[tk]))     # THE REAL 0/1, not a proxy
        MKT.append(raw["market"].to_numpy(dtype=object))
        del raw,sf,A; gc.collect()
        if i%300==0: print(f"      {i}/{len(paths)}",flush=True)
    if not F: return None
    print(f"   {coin}: {skip} no mid tape, {noout} no settled outcome",flush=True)
    return (np.vstack(F),np.concatenate(MID),np.concatenate(SPR),
            {h:np.concatenate(v) for h,v in FW.items()},
            np.concatenate(SET),np.concatenate(MKT),feats)

hdr=(f"{'coin':5s} {'model':6s} {'side':6s} {'trades':>9s} {'take':>6s} {'hit':>6s} "
     f"{'c/trade':>9s} {'bp cap':>9s} {'r/e':>6s} {'sig':>4s} {'95% CI':>20s} {'mk':>5s}")
print("\n"+hdr,flush=True)

for coin in COINS:
    if any(k.startswith(f"{coin}|") for k in res): print(f"{coin}: cached",flush=True); continue
    L_=load(coin)
    if L_ is None: print(f"{coin:5s} no data",flush=True); continue
    X,mid,spr,FWD,OUT01,mkt,feats=L_
    mc,munq=pd.factorize(pd.Series(mkt),sort=True); mc=mc.astype("int32")
    munq=np.asarray(munq,dtype=object)
    tko=np.array([tkey(m) for m in munq],dtype="int64")
    order=np.argsort(tko); nte=int(len(order)*0.30)
    c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mc,cs))
    y=np.where(mid>0,(OUT01-mid)/mid,np.nan)
    ok=np.isfinite(y)&np.isfinite(mid)&np.isfinite(spr)&(mid>0)
    i_te=sel(c_te)[ok[sel(c_te)]]; i_ho=sel(c_ho)[ok[sel(c_ho)]]; i_tr=sel(c_tr)[ok[sel(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"{coin:5s} too few rows",flush=True); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    def mw(ix):
        """1/n_rows_in_market, so each MARKET carries equal total weight."""
        g=mc[ix]; u,inv=np.unique(g,return_inverse=True)
        ct=np.bincount(inv).astype("float64")
        w=1.0/ct[inv]; return (w/w.sum()).astype("float64")
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
    wA=mw(idxA); wh=mw(i_ho); wte=mw(i_te)
    m0=mid[i_te]; s0=spr[i_te]; mc_te=mc[i_te]; out_te=OUT01[i_te]
    half=s0/2.0; fee=TAKER*m0*(1.0-m0); cost=half+fee

    def mkt_boot(v,mask,B=2000,seed=0):
        g=mc_te[mask]; u,inv=np.unique(g,return_inverse=True)
        sm=np.bincount(inv,weights=v,minlength=len(u))
        ct=np.bincount(inv,minlength=len(u)).astype("float64")
        rng=np.random.default_rng(seed)
        pk=rng.integers(0,len(u),size=(B,len(u)))
        d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
        return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
                float(min((d<=0).mean(),(d>=0).mean())*2),len(u))

    def report(tag,model,side,take,edge):
        good=take&np.isfinite(out_te)
        n=int(good.sum())
        if n==0:
            print(f"{coin:5s} {model:6s} {tag:6s} {0:>9} -- no trades --",flush=True); return
        realized=side[good]*(out_te[good]-m0[good])
        pnl=realized-cost[good]              # ONE fee: settlement is not a trade
        cap=np.where(side[good]>0,m0[good]+half[good],(1.0-m0[good])+half[good])
        rn=pnl/np.maximum(cap,1e-9)
        hit=float((realized>0).mean())
        ex=(edge[good]-cost[good]) if edge is not None else None
        r_e=float(pnl.mean()/ex.mean()) if ex is not None and ex.mean()!=0 else float("nan")
        lo_,hi_,p_,nmk=mkt_boot(pnl,good)
        sig=(lo_>0)==(hi_>0)
        print(f"{coin:5s} {model:6s} {tag:6s} {n:>9,} {100*n/len(m0):>5.2f}% {100*hit:>5.1f}% "
              f"{100*pnl.mean():>+8.4f}c {1e4*rn.mean():>+8.1f}bp {r_e:>6.2f} "
              f"{('SIG' if sig else '-'):>4s} [{100*lo_:+.4f},{100*hi_:+.4f}]c {nmk:>5d}",flush=True)
        res[f"{coin}|{model}|{tag}"]={"trades":n,"take_rate":float(n/len(m0)),"hit":hit,
            "cents_per_trade":float(100*pnl.mean()),"bp_of_capital":float(1e4*rn.mean()),
            "realised_over_expected":r_e,"ci_cents":[float(100*lo_),float(100*hi_)],
            "p":p_,"sig":bool(sig),"boot_markets":nmk,
            "mean_capital_cents":float(100*cap.mean()),
            "sel_half_spread_cents":float(100*half[good].mean()),
            "sel_fee_cents":float(100*fee[good].mean()),
            "nfeat":len(feats),"test_markets":len(c_te)}
        save()

    for name,W_,L_d,DR,WD in MODELS:
        if name=="ridge":
            from sklearn.linear_model import Ridge
            best=(-9e9,None)
            for al in [0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0]:
                r=Ridge(alpha=al).fit(XA,(yA-ym)/ys,sample_weight=wA)
                v=r2_zero(y[i_ho],r.predict(Xh)*ys+ym,wh)
                if v>best[0]: best=(v,r,al)
            p=best[1].predict(Xte)*ys+ym
            print(f"   {coin} ridge alpha={best[2]} holdout r2={best[0]:+.6f}",flush=True)
        else:
            ps=[pred(fit(XA,((yA-ym)/ys).astype("float32"),wA,Xh,((y[i_ho]-ym)/ys),wh,
                         s_,W_,L_d,DR,WD),Xte)*ys+ym for s_ in SEEDS]
            p=np.mean(ps,axis=0)
        r2=float(r2_zero(y[i_te],p,wte))
        print(f"   {coin} {name}: test r2(settle)={r2:+.6f}",flush=True)
        res[f"{coin}|{name}|r2"]={"r2_settle":r2}
        edge=np.abs(p)*m0
        take=edge>cost
        report("model",name,np.sign(p),take,edge)
        # SAME gate, SAME costs, side held constant -- the bias controls.
        report("long",name,np.ones_like(p),take,edge)
        report("short",name,-np.ones_like(p),take,edge)
    # ungated constant side: the pure static bias, no model involved at all
    allrows=np.ones(len(m0),dtype=bool)
    report("long",  "none", np.ones(len(m0)),  allrows, None)
    report("short", "none", -np.ones(len(m0)), allrows, None)
    save(); del X,XA,Xh,Xte; gc.collect()
save(); print(f"\nwrote {OUT}")
