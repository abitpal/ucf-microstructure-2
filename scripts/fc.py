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
OUT="/var/lib/kalshi-study/FILLCHECK.json"
TAKER=0.07; SCALE=10000.0
W,L,DROP=640,5,0.2; SEEDS=(11,23); BS=1024; MAXROWS=1_000_000
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
def ticksize(m):
    return np.where((m<0.10)|(m>=0.90),0.0010,0.0100)

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
    F=[];MID=[];SPR=[];FW=[];TS=[];MKT=[];skip=0
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
        j=np.searchsorted(mts,ts+0.1,side="left"); okj=j<len(mts)
        jc=np.clip(j,0,len(mts)-1)
        v=np.where(okj,mmd[jc],np.nan); got=np.where(okj,mts[jc]-ts,np.nan)
        F.append(A);MID.append(mid);SPR.append(spr);TS.append(ts)
        FW.append(np.where(got<=0.1*1.25+0.005,v,np.nan))
        MKT.append(raw["market"].to_numpy(dtype=object))
        del raw,sf,A; gc.collect()
        if i%300==0: print(f"      {i}/{len(paths)}",flush=True)
    if not F: return None
    print(f"   {coin}: {skip} markets had no mid tape",flush=True)
    return (np.vstack(F),np.concatenate(MID),np.concatenate(SPR),
            np.concatenate(FW),np.concatenate(TS),np.concatenate(MKT),feats)

hdr=(f"{'coin':5s} {'arm':5s} {'tk':>3s} {'win':>5s} {'n':>8s} "
     f"{'fill%prn':>9s} {'fill%tch':>9s} {'E[prn]':>9s} {'E[tch]':>9s} "
     f"{'cross':>9s} {'sig':>4s} {'95% CI prn':>20s}")
print("\n"+hdr,flush=True)

for coin in COINS:
    if any(k.startswith(f"{coin}|") for k in res): print(f"{coin}: cached",flush=True); continue
    L_=load(coin)
    if L_ is None: print(f"{coin:5s} no data",flush=True); continue
    X,mid,spr,FW1,TSA,mkt,feats=L_
    mc,munq=pd.factorize(pd.Series(mkt),sort=True); mc=mc.astype("int32")
    munq=np.asarray(munq,dtype=object)
    tko=np.array([tkey(m) for m in munq],dtype="int64")
    order=np.argsort(tko); nte=int(len(order)*0.30)
    c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mc,cs))
    y=np.where(mid>0,(FW1-mid)/mid,np.nan)
    ok=np.isfinite(y)&np.isfinite(mid)&np.isfinite(spr)&(mid>0)
    i_te=sel(c_te)[ok[sel(c_te)]]; i_ho=sel(c_ho)[ok[sel(c_ho)]]; i_tr=sel(c_tr)[ok[sel(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"{coin:5s} too few rows",flush=True); continue
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
    ps=[pred(fit(XA,((yA-ym)/ys).astype("float32"),wA,Xh,((y[i_ho]-ym)/ys),wh,s_),Xte)*ys+ym
        for s_ in SEEDS]
    p=np.mean(ps,axis=0)
    print(f"   {coin}: test r2(0.1s)={r2_zero(y[i_te],p,np.ones(len(i_te))/len(i_te)):+.6f}",flush=True)
    del XA,Xh,Xte; gc.collect()

    m0=mid[i_te]; s0=spr[i_te]; t0=TSA[i_te]; mc_te=mc[i_te]
    half=s0/2.0; fee0=TAKER*m0*(1.0-m0); tks=ticksize(m0)
    edge=np.abs(p)*m0
    take=edge>(half+fee0)
    rng=np.random.default_rng(7)
    ARMS={"model":np.sign(p),"flip":-np.sign(p),
          "rand":rng.choice([-1.0,1.0],size=len(p))}

    # ---- walk each test market's touch tape once, for every (arm,tick,window)
    keys=[(a,tk,w) for a in ARMS for tk in TICKS for w in WINS]
    acc={k:{"fp":[],"ft":[],"pf":[],"pn":[],"mk":[]} for k in keys}
    miss=0
    for ci in c_te:
        tkr=munq[ci]
        f=f"{TT}/tt_{tkr}.parquet"
        if not os.path.exists(f): miss+=1; continue
        tt=pq.read_table(f,columns=["ts","best_bid","best_ask","last_trade","trade_count"])
        tts=tt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        bb=tt.column("best_bid").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        ba=tt.column("best_ask").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        lt=tt.column("last_trade").to_numpy(zero_copy_only=False).astype("float64")/SCALE
        tc=tt.column("trade_count").to_numpy(zero_copy_only=False).astype("float64")
        o=np.argsort(tts); tts,bb,ba,lt,tc=tts[o],bb[o],ba[o],lt[o],tc[o]
        # prints: rows where the counter advanced
        # np.diff(..., prepend=x) returns an array the SAME length as the input,
        # so this must be a whole-array mask -- an earlier version assigned it to
        # adv[1:] and raised a broadcast error on every market.
        dtc=np.diff(np.where(np.isfinite(tc),tc,0.0),prepend=0.0)
        adv=np.isfinite(tc)&np.isfinite(lt)&(dtc>0)
        pts=tts[adv]; ppx=lt[adv]
        tmid=np.where(np.isfinite(bb)&np.isfinite(ba),(bb+ba)/2.0,np.nan)
        rows=np.flatnonzero((mc_te==ci)&take)
        if len(rows)==0 or len(pts)==0: continue
        tr=t0[rows]
        # NaN-free copies so a window max/min needs no nan-aware reduction (and
        # raises no all-NaN warning): a missing price simply cannot be the
        # extreme in the direction being tested.
        ppx_hi=np.where(np.isfinite(ppx),ppx,-np.inf)
        ppx_lo=np.where(np.isfinite(ppx),ppx, np.inf)
        # A resting SELL at L sits in the ASK ladder: it fills once the asks
        # below L are consumed, i.e. best_ask reaches L. Testing best_bid >= L
        # instead describes a CROSSED book and is far stricter -- which made the
        # deliberately-overstating touch rule come out below the print rule and
        # broke the bracket. Mirror image for a resting BUY and best_bid.
        ba_hi =np.where(np.isfinite(ba), ba, -np.inf)
        bb_lo =np.where(np.isfinite(bb), bb,  np.inf)
        # Window extremes depend ONLY on the window, not on the arm or the tick
        # target, so they are computed once per window instead of 45 times per
        # trade. Everything downstream is then pure arithmetic over rows.
        WX={}
        for w in WINS:
            j0=np.searchsorted(pts,tr,side="right"); j1=np.searchsorted(pts,tr+w,side="right")
            k0=np.searchsorted(tts,tr,side="right"); k1=np.searchsorted(tts,tr+w,side="right")
            n_=len(rows)
            pmax=np.full(n_,-np.inf); pmin=np.full(n_,np.inf)
            amax=np.full(n_,-np.inf); bmin=np.full(n_,np.inf)
            mW=np.full(n_,np.nan);    hW=np.full(n_,np.nan)
            for q in range(n_):
                if j1[q]>j0[q]:
                    pmax[q]=ppx_hi[j0[q]:j1[q]].max()
                    pmin[q]=ppx_lo[j0[q]:j1[q]].min()
                if k1[q]>k0[q]:
                    amax[q]=ba_hi[k0[q]:k1[q]].max()
                    bmin[q]=bb_lo[k0[q]:k1[q]].min()
                    kx=k1[q]-1
                    mW[q]=tmid[kx]; hW[q]=(ba[kx]-bb[kx])/2.0
            WX[w]=(pmax,pmin,amax,bmin,mW,hW)
        for a,sd in ARMS.items():
            side=sd[rows]
            for ntk in TICKS:
                d=ntk*tks[rows]
                Lv=np.where(side>0,m0[rows]+half[rows]+d,m0[rows]-half[rows]-d)
                for w in WINS:
                    pmax,pmin,amax,bmin,mW,hW=WX[w]
                    fp=np.where(side>0,pmax>=Lv,pmin<=Lv)
                    ft=np.where(side>0,amax>=Lv,bmin<=Lv)
                    good=np.isfinite(mW)&np.isfinite(hW)
                    if not good.any(): continue
                    feeW=TAKER*mW*(1.0-mW)
                    pnl_fill=d-fee0[rows]
                    pnl_cross=side*(mW-m0[rows])-half[rows]-fee0[rows]-hW-feeW
                    K=(a,ntk,w)
                    acc[K]["fp"].append(fp[good]); acc[K]["ft"].append(ft[good])
                    acc[K]["pf"].append(pnl_fill[good]); acc[K]["pn"].append(pnl_cross[good])
                    acc[K]["mk"].append(np.full(int(good.sum()),ci,dtype="int64"))
    print(f"   {coin}: {miss} test markets had no touch tape",flush=True)

    def boot(v,g,B=2000,seed=0):
        u,inv=np.unique(g,return_inverse=True)
        sm=np.bincount(inv,weights=v,minlength=len(u))
        ct=np.bincount(inv,minlength=len(u)).astype("float64")
        r=np.random.default_rng(seed)
        pk=r.integers(0,len(u),size=(B,len(u)))
        d=sm[pk].sum(1)/np.maximum(ct[pk].sum(1),1e-9)
        return (float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),
                float(min((d<=0).mean(),(d>=0).mean())*2))
    for (a,ntk,w) in keys:
        A_=acc[(a,ntk,w)]
        if not A_["fp"]: continue
        fp=np.concatenate(A_["fp"]);ft=np.concatenate(A_["ft"])
        pf=np.concatenate(A_["pf"]);pn=np.concatenate(A_["pn"])
        g=np.concatenate(A_["mk"]); n=len(fp)
        ep=np.where(fp,pf,pn); et=np.where(ft,pf,pn)
        lo_,hi_,pv=boot(ep,g)
        sig=(lo_>0)==(hi_>0)
        print(f"{coin:5s} {a:5s} {ntk:>3d} {w:>5.0f} {n:>8,} "
              f"{100*fp.mean():>8.2f}% {100*ft.mean():>8.2f}% "
              f"{100*ep.mean():>+8.4f}c {100*et.mean():>+8.4f}c {100*pn.mean():>+8.4f}c "
              f"{('SIG' if sig else '-'):>4s} [{100*lo_:+.4f},{100*hi_:+.4f}]c",flush=True)
        res[f"{coin}|{a}|{ntk}|{w}"]={"n":n,
            "mean_mid":float(m0[take].mean()),
            "mean_tick_cents":float(100*tks[take].mean()),
            "mean_fee_cents":float(100*fee0[take].mean()),
            "fill_print":float(fp.mean()),"fill_touch":float(ft.mean()),
            "exp_print_cents":float(100*ep.mean()),"exp_touch_cents":float(100*et.mean()),
            "cross_cents":float(100*pn.mean()),"pnl_if_filled_cents":float(100*pf.mean()),
            "ci_cents":[float(100*lo_),float(100*hi_)],"p":pv,"sig":bool(sig)}
    save(); del X; gc.collect()
save(); print(f"\nwrote {OUT}")
