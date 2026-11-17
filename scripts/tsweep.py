"""FILL CHECK: can the 0.1s edge be taken in and RESTED out?

THE QUESTION
------------
Mark-to-mid on the 0.1s signal is strongly positive (+0.66c BTC, p<0.0001) and
crossing back out is significantly negative (-0.53c), because Kalshi charges the
taker fee on EVERY trade. Makers pay nothing on all four of these series
(`fee_type: quadratic`, API-verified), so a resting exit pays one fee instead of
two. Whether that is reachable depends entirely on the FILL RATE, which nothing
on disk could answer before the touch+trade tape existed.

THE TRADE
---------
Enter by crossing, exactly as tc.py does: gate `|pred|*mid > half + fee`.
  long   buy YES at  ask = mid + half,  then POST a sell at  L = mid + half + d
  short  sell YES at bid = mid - half,  then POST a buy  at  L = mid - half - d
d is the target in TICKS (0.0010 below 10c or above 90c, else 0.0100).

  filled      PnL = d - fee_in                 maker fee is zero, so the only
                                               cost is the entry fee
  not filled  cross out at the window-end mid, paying the second half spread
              and a second fee:
              PnL = side*(mid_W - mid_0) - half_0 - fee_0 - half_W - fee_W

Expected PnL = p_fill*(d - fee_in) + (1-p_fill)*(cross-out PnL).

SIDE IS NOT NEEDED TO JUDGE A FILL. Price priority settles it: a print at
price >= L means a buyer paid >= L, and a resting sell at L sits ahead of them
on price, so it would have been consumed first. Same in reverse for a buy.

TWO FILL RULES, because each errs in a known direction
------------------------------------------------------
  print  a print reaches our level. STRICT variant requires strictly through,
         which understates (ignores same-level fills); inclusive overstates
         slightly (ignores time priority at our own level).
  touch  the opposite side of the book reaches our level. OVERSTATES -- the
         touch can move on zero volume.
Reporting the bracket is the point. The football study produced one number that
later needed caveating; this one is built to show its own error bars.

THE CONTROL THAT DECIDES IT
---------------------------
A resting order fills when the market comes to it, which it sometimes does on
its own. So `flip` runs the identical machinery with the side REVERSED and
`rand` with a coin flip. The model only adds something if its fill rate at a
favourable level beats those. If all three fill alike, the fills are just market
noise and the forecast is irrelevant -- which is exactly the trap
kalshi-passive-exit-does-not-fill warns about ("fill probability is lowest
exactly where the fee math is best").
"""
import sys, os, re, glob, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn
from tools.panel_model import scale_free, r2_zero
torch.set_num_threads(int(os.environ.get("FC_THREADS","12")))
COINS=os.environ.get("FC_COINS","BTC,ETH,XRP,NEAR").split(",")
PANELS="/mnt/data/panels"; MIDS="/var/lib/kalshi-study/midtape"
TT="/mnt/data/touchtape"
OUT="/var/lib/kalshi-study/TSWEEP.json"
CACHE="/mnt/data/fccache"
SZ="/mnt/data/sizetape"
TAKER=0.07; SCALE=10000.0
W,NLAY,DROP=640,5,0.2; SEEDS=(11,23); BS=1024; MAXROWS=1_000_000  # NLAY, not L: the driver loops `for L in DELAYS` at module scope
WINS=[1.0,2.0,5.0,15.0,60.0]      # rest windows, seconds
TICKS=[1,2,3,4,6]                  # target in ticks

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
        for _ in range(NLAY): q+=[nn.Linear(W,W),nn.ReLU(),nn.Dropout(DROP)]
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
def ticksize(m):
    return np.where((m<0.10)|(m>=0.90),0.0010,0.0100)


TGTS=[0.1,0.5,1.0,2.0,5.0,10.0,30.0]   # training targets to sweep
MARKS=[1.0,5.0,30.0]                    # hold horizons
DELAYS=[0.005,0.010,0.020]              # execution delay
CLIPS=[100,500,5000]
MKT_PER_DAY=96
LAGS=sorted(set(TGTS)|set(MARKS))

def load(coin):
    """One load pass. Forward mids at every lag used as a target OR a hold, so
    all targets are trained and scored on IDENTICAL rows, splits and features."""
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
    F=[];MID=[];SPR=[];TS=[];MKT=[];FW={L:[] for L in LAGS};skip=0
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
        for L in LAGS:
            j=np.searchsorted(mts,ts+L,side="left"); okj=j<len(mts)
            jc=np.clip(j,0,len(mts)-1)
            v=np.where(okj,mmd[jc],np.nan); got=np.where(okj,mts[jc]-ts,np.nan)
            FW[L].append(np.where(got<=L*1.25+0.005,v,np.nan))
        F.append(A);MID.append(mid);SPR.append(spr);TS.append(ts)
        MKT.append(raw["market"].to_numpy(dtype=object))
        del raw,sf,A; gc.collect()
        if i%300==0: print(f"      {i}/{len(paths)}",flush=True)
    if not F: return None
    print(f"   {coin}: {skip} markets had no mid tape",flush=True)
    return (np.vstack(F),np.concatenate(MID),np.concatenate(SPR),
            {L:np.concatenate(v) for L,v in FW.items()},
            np.concatenate(TS),np.concatenate(MKT),feats)

def boot(v,g,B=1500,seed=0):
    u,inv=np.unique(g,return_inverse=True)
    sm=np.bincount(inv,weights=v,minlength=len(u))
    ct=np.bincount(inv,minlength=len(u)).astype("float64")
    r=np.random.default_rng(seed)
    pk=r.integers(0,len(u),size=(B,len(u)))
    d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
    return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
            float(min((d<=0).mean(),(d>=0).mean())*2))

for coin in COINS:
    L_=load(coin)
    if L_ is None: print(f"{coin}: no data",flush=True); continue
    X,mid,spr,FWD,TSA,mkt,feats=L_
    mc,munq=pd.factorize(pd.Series(mkt),sort=True); mc=mc.astype("int32")
    munq=np.asarray(munq,dtype=object)
    tko=np.array([tkey(m) for m in munq],dtype="int64")
    order=np.argsort(tko); nte=int(len(order)*0.30)
    c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mc,cs))
    ok=np.isfinite(mid)&np.isfinite(spr)&(mid>0)
    for T in LAGS: ok&=np.isfinite(FWD[T])
    i_te=sel(c_te)[ok[sel(c_te)]]; i_ho=sel(c_ho)[ok[sel(c_ho)]]; i_tr=sel(c_tr)[ok[sel(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"{coin}: too few rows",flush=True); continue
    print(f"   {coin}: common rows train={len(i_tr):,} test={len(i_te):,}",flush=True)
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
    m0=mid[i_te]; s0=spr[i_te]; t0=TSA[i_te]; mc_te=mc[i_te]
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0)
    n=len(m0)

    # ---- market tapes once: bid/ask + touch size at t and t+delay, fwd mids
    print("   reading size tapes for test markets...",flush=True)
    A0=np.full(n,np.nan); B0=np.full(n,np.nan)
    AD={L:np.full(n,np.nan) for L in DELAYS}; BD={L:np.full(n,np.nan) for L in DELAYS}
    MD={L:np.full(n,np.nan) for L in DELAYS}; SD={L:np.full(n,np.nan) for L in DELAYS}
    SZA={L:np.full(n,np.nan) for L in DELAYS}; SZB={L:np.full(n,np.nan) for L in DELAYS}
    FWT={(L,h):np.full(n,np.nan) for L in DELAYS for h in MARKS}
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
        def at(tgt,arr):
            j=np.searchsorted(tts,tgt,side="right")-1
            okk=j>=0; j=np.clip(j,0,len(tts)-1)
            return np.where(okk,arr[j],np.nan)
        A0[rows]=at(tr,ba); B0[rows]=at(tr,bb)
        for L in DELAYS:
            AD[L][rows]=at(tr+L,ba); BD[L][rows]=at(tr+L,bb)
            MD[L][rows]=at(tr+L,tm); SD[L][rows]=at(tr+L,ba)-at(tr+L,bb)
            SZA[L][rows]=at(tr+L,as_); SZB[L][rows]=at(tr+L,bs)
            for h in MARKS:
                j=np.searchsorted(tts,tr+L+h,side="right")-1
                okk=j>=0; jc=np.clip(j,0,len(tts)-1)
                v=np.where(okk,tm[jc],np.nan); age=np.where(okk,(tr+L+h)-tts[jc],np.nan)
                FWT[(L,h)][rows]=np.where(age<=h*0.25+0.005,v,np.nan)
    print(f"   {nm} test markets had a size tape",flush=True)

    print(f"\n{'tgt':>5s} {'r2':>10s} {'dly':>5s} {'h':>5s} {'trades':>8s} {'pass%':>6s} "
          f"{'c/contract':>11s} {'sig':>4s} {'tr/day':>8s} {'fill@100':>9s} "
          f"{'$/d@100':>9s} {'$/d@500':>9s} {'$/d@5k':>9s}",flush=True)
    for T in TGTS:
        y=np.where(mid>0,(FWD[T]-mid)/mid,np.nan)
        q=np.nanquantile(y[i_tr],[.01,.99])
        yA=np.clip(y[idxA],q[0],q[1]); ym,ys=float(yA.mean()),float(yA.std()) or 1.0
        wA=np.ones(len(idxA))/len(idxA); wh=np.ones(len(i_ho))/len(i_ho)
        ps=[pred(fit(XA,((yA-ym)/ys).astype("float32"),wA,Xh,((y[i_ho]-ym)/ys),wh,s_),Xte)*ys+ym
            for s_ in SEEDS]
        p=np.mean(ps,axis=0)
        r2=float(r2_zero(y[i_te],p,np.ones(len(i_te))/len(i_te)))
        np.savez(f"{CACHE}/{coin}_T{T}.npz",p=p,m0=m0,s0=s0,t0=t0,mc_te=mc_te,
                 munq=munq.astype("U64"),c_te=c_te,r2=r2)
        take=(np.abs(p)*m0)>(half+fee0); side=np.sign(p)
        for L in DELAYS:
            nw=np.where(side>0, AD[L]<=A0, BD[L]>=B0)
            av=np.where(side>0, SZA[L], SZB[L])
            for h in MARKS:
                hm=SD[L]/2.0; fm=TAKER*MD[L]*(1.0-MD[L])
                pnl=side*(FWT[(L,h)]-MD[L])-hm-fm
                msk=take&nw&np.isfinite(pnl)&np.isfinite(SD[L])&(SD[L]>=0)&np.isfinite(av)
                if msk.sum()<200: continue
                v=pnl[msk]; g=mc_te[msk]
                lo_,hi_,pv=boot(v,g); sig=(lo_>0)==(hi_>0)
                pr=float(nw[take&np.isfinite(AD[L])].mean())
                tpd=msk.sum()/max(nm,1)*MKT_PER_DAY
                a=av[msk]
                f100=float(np.minimum(100,a).mean())
                f500=float(np.minimum(500,a).mean())
                f5k=float(np.minimum(5000,a).mean())
                pc=float(v.mean())
                print(f"{T:>5.1f} {r2:>+10.6f} {1000*L:>4.0f}m {h:>5.1f} {int(msk.sum()):>8,} "
                      f"{100*pr:>5.1f}% {100*pc:>+10.4f}c {('SIG' if sig else '-'):>4s} "
                      f"{tpd:>8,.0f} {f100:>9.1f} {tpd*f100*pc:>9,.0f} "
                      f"{tpd*f500*pc:>9,.0f} {tpd*f5k*pc:>9,.0f}",flush=True)
                res[f"{coin}|{T}|{L}|{h}"]={"r2":r2,"trades":int(msk.sum()),
                    "guard_pass":pr,"cents_per_contract":100*pc,"sig":bool(sig),
                    "ci_cents":[100*lo_,100*hi_],"p":pv,"trades_per_day":float(tpd),
                    "avg_fill_100":f100,"avg_fill_500":f500,"avg_fill_5000":f5k,
                    "usd_day_100":float(tpd*f100*pc),"usd_day_500":float(tpd*f500*pc),
                    "usd_day_5000":float(tpd*f5k*pc),
                    "median_touch_ct":float(np.median(a))}
        save()
    del X,XA,Xh,Xte; gc.collect()
save(); print(f"\nwrote {OUT}")
