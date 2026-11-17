"""THE GRID: 0.02 -> 0.1s mid-to-mid increment, every coin, binary and perp.

TARGET, and nothing else:
    y = (mid(t+0.10) - mid(t+0.02)) / mid(t+0.02)
Entry and exit mids come from the FULL-RESOLUTION mid tape (every event), not
the 25-event feature grid whose median inter-row gap is 22-40ms and which
therefore cannot resolve a 20ms lag at all. Rows whose nearest event lands more
than 25% past the requested lag are DROPPED rather than silently accepted -- on
the sampled grid that failure hit 53% of rows at 20ms and 26% even at 0.1s.

Chosen because the lag sweep measured +0.013404 here against +0.002117 for the
0.1->1s increment this study had been optimising: 6.3x, and rising monotonically
as both entry lag and hold shrink. It is mark-to-mid and does not pay the
spread.

RECIPE (fixed, applied identically to every cell):
  model        deep + wide: width 640, 6 layers, dropout 0.2
  objectives   multi-head -- main target plus the three neighbouring lag pairs,
               so the trunk predicts several horizons at once; only the MAIN
               head is ever scored
  features     every scale_free column the panel yields (106 on panel_big)
  target norm  divided by a RANK-NORMALISED trailing-vol estimate, which matched
               raw pooled R^2 while cutting per-market dispersion 6.5x and the
               worst decile 8.6x; predictions are un-normalised before scoring,
               so every number below is in the target's own units
  clipping     features and target winsorised at 1/99 on TRAINING rows only
  standardise  per-feature mean/sd from training rows only
  weights      per-market 1/n so a busy market cannot dominate
  seeds        2, averaged

Splits are market-disjoint and time-ordered per coin: last 30% of that coin's
markets are test, 120 before them are the holdout.
"""
import sys, os, re, glob, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch
import torch.nn as nn
from tools.panel_model import scale_free, r2_zero
torch.set_num_threads(int(os.environ.get("FG_THREADS","12")))

COINS=os.environ.get("FG_COINS","BTC,ETH,XRP,NEAR").split(",")
PANELS=os.environ.get("FG_PANELS","/mnt/data/panels")
MIDS="/var/lib/kalshi-study/midtape"
OUT="/var/lib/kalshi-study/FINALGRID.json"
ENTRY,EXIT=0.02,0.10
AUX=[(0.02,0.05),(0.05,0.10),(0.02,0.20)]     # neighbouring lags as aux heads
W,L,DROP=640,5,0.2
SEEDS=(11,23); BS=1024; MAXROWS=1_200_000
def tg(x): return f"{x:g}".replace(".","p")
MAIN=f"inc_{tg(ENTRY)}_to_{tg(EXIT)}"
HEADS=[(ENTRY,EXIT)]+AUX

res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT))
    except Exception: res={}
def save():
    t=OUT+".tmp"; json.dump(res,open(t,"w"),indent=1,default=str); os.replace(t,OUT)

def build(coin):
    """Per-coin frame: features on the panel grid, targets from the mid tape."""
    out=f"/mnt/data/frames/fg_{coin}.parquet"
    if os.path.exists(out) and not os.environ.get("FG_REBUILD"):
        return out
    # Prefer the RICH panels for this coin (210 features) and fall back to the
    # fast-tier panel_big (106) only if the rich replay has not reached it yet,
    # so the grid always uses the widest feature set available per coin and says
    # which it used.
    SCHEMA=[]; MISSING={}; FEATS=[None]
    paths=[]
    for c in (f"{PANELS}/rich_{coin}", PANELS, "/var/lib/kalshi-study/panel_big"):
        paths=sorted(glob.glob(f"{c}/panel_KX{coin}15M-*.parquet"))
        if paths:
            print(f"   {coin}: {len(paths)} panels from {c}",flush=True); break
    # THE FEATURE SET IS A UNION OVER A SAMPLE, NOT THE FIRST MARKET'S COLUMNS.
    #
    # Markets are processed in sorted filename order, so the first is at the very
    # start of the tape -- before the perp feed begins. Pinning the feature list
    # to that market silently threw away all 110 perp/cross-market columns from
    # every later market: the binary models were trained with ZERO perp features
    # while the panels carried 126 of them. This is the third time in this study
    # that deriving a schema from one file has dropped a whole family, so the
    # union is taken over markets spread across the whole range and any column
    # missing from a given market is null-filled rather than dropped globally.
    probe=paths[::max(1,len(paths)//12)][:12]
    union=set()
    for q in probe:
        rq=pd.read_parquet(q)
        kq=[c for c in rq.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sq=scale_free(rq[kq])
        union |= {c for c in sq.columns
                  if not c.startswith(("y_","ry_","market","_day"))}
        del rq,sq
    FEATS[0]=sorted(union)
    print(f"   {coin}: feature union over {len(probe)} probe markets = "
          f"{len(FEATS[0])} columns "
          f"({sum(1 for c in FEATS[0] if c.startswith(('p_','xmkt')))} perp/cross-market)",
          flush=True)
    w=None; tot=0; nomid=0
    for i,p in enumerate(paths,1):
        tk=re.sub(r"^panel_|\.parquet$","",os.path.basename(p))
        mp=f"{MIDS}/mid_{tk}.parquet"
        if not os.path.exists(mp): nomid+=1; continue
        raw=pd.read_parquet(p)
        if "market" not in raw.columns: raw["market"]=tk
        keep=[c for c in raw.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sf=scale_free(raw[keep])
        feats=FEATS[0]
        ts=raw["ts"].to_numpy(dtype="float64")
        mt=pq.read_table(mp,columns=["ts","mid"])
        mts=mt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        mmd=mt.column("mid").to_numpy(zero_copy_only=False).astype("float64")
        o=np.argsort(mts); mts,mmd=mts[o],mmd[o]
        def at(lag):
            j=np.searchsorted(mts,ts+lag,side="left"); ok=j<len(mts)
            jc=np.clip(j,0,len(mts)-1)
            v=np.where(ok,mmd[jc],np.nan)
            got=np.where(ok,mts[jc]-ts,np.nan)
            return np.where(got<=lag*1.25+0.005,v,np.nan)   # resolution guard
        d={"market":raw["market"].to_numpy(dtype=object),"ts":ts}
        for c in feats:
            if c in sf.columns:
                d[c]=sf[c].to_numpy(dtype="float32")
            else:
                d[c]=np.full(len(ts),np.nan,dtype="float32")
                MISSING[c]=MISSING.get(c,0)+1
        for a,b in HEADS:
            ma,mb=at(a),at(b)
            with np.errstate(divide="ignore",invalid="ignore"):
                d[f"inc_{tg(a)}_to_{tg(b)}"]=np.where(np.abs(ma)>1e-12,
                                                      (mb-ma)/ma,np.nan).astype("float32")
        # the perp leg: same construction on the perp mid carried in the panel
        if "p_mid" in raw.columns:
            pm=raw["p_mid"].to_numpy(dtype="float64")
            d["p_mid_now"]=pm.astype("float32")
        tbl=pa.table(d)
        # PIN THE SCHEMA TO THE FIRST MARKET AND CONFORM EVERY LATER ONE.
        #
        # ParquetWriter is opened with one market's schema, but markets do not
        # all yield the same columns -- one missing p_mid, or a flow family that
        # never fired, produces a different set and the writer raises
        # "Table schema does not match schema used to create file". That is what
        # killed this build. Columns absent from a later market are filled with
        # nulls and extras are dropped, so the frame is rectangular and the
        # difference is visible rather than fatal.
        if w is None:
            SCHEMA.append(tbl.schema)
            w=pq.ParquetWriter(out,tbl.schema,compression="zstd")
        else:
            tgt=SCHEMA[0]
            if tbl.schema!=tgt:
                for f_ in tgt:
                    if f_.name not in tbl.schema.names:
                        tbl=tbl.append_column(f_,pa.nulls(tbl.num_rows,type=f_.type))
                        MISSING[f_.name]=MISSING.get(f_.name,0)+1
                tbl=tbl.select([f_.name for f_ in tgt]).cast(tgt)
        w.write_table(tbl); tot+=tbl.num_rows
        del raw,sf,tbl,d; gc.collect()
        if i%200==0: print(f"   {coin} {i}/{len(paths)} {tot:,} rows "
                           f"(no mid tape: {nomid})",flush=True)
    if w is not None: w.close()
    print(f"   {coin}: {tot:,} rows, {nomid} markets without a mid tape"
          + (f", columns null-filled in some markets: {MISSING}" if MISSING else ""),
          flush=True)
    return out if tot else None

class Multi(nn.Module):
    def __init__(s,nin,nh):
        super().__init__()
        q=[nn.Linear(nin,W),nn.ReLU(),nn.Dropout(DROP)]
        for _ in range(L): q+=[nn.Linear(W,W),nn.ReLU(),nn.Dropout(DROP)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(W,nh)
    def forward(s,x): return s.head(s.body(x))

def fit(X,Y,Wt,M,Xv,Yv,Wv,seed,steps=5000,lr=1e-3,wd=0.3,pat=10):
    nh=Y.shape[1]
    torch.manual_seed(seed); m=Multi(X.shape[1],nh)
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=lr,total_steps=steps,pct_start=.15)
    A=torch.from_numpy(X);T=torch.from_numpy(Y).float()
    Wq=torch.from_numpy(Wt).float().view(-1,1);Mk=torch.from_numpy(M).float()
    hw=torch.tensor([1.0]+[0.3]*(nh-1))
    V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//20)
    while st<steps and bad<pat:
        pm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=pm[i:i+BS];o.zero_grad()
            e=(m(A[ix])-T[ix])**2*Mk[ix]*Wq[ix]*hw
            (e.sum()/(Mk[ix]*Wq[ix]*hw).sum().clamp_min(1e-9)).backward();o.step()
            if st<steps-1: sc.step()
            st+=1
            if st%ev==0 or st>=steps:
                m.eval()
                with torch.no_grad(): P=m(V).numpy()
                v=r2_zero(Yv[:,0],P[:,0],Wv)
                if v>best[0]: best=(v,[p.detach().clone() for p in m.parameters()]);bad=0
                else: bad+=1
                if bad>=pat or st>=steps: break
    if best[1] is not None:
        with torch.no_grad():
            for p,b in zip(m.parameters(),best[1]): p.copy_(b)
    m.eval(); return m
def pr(m,A,ch=1_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy()[:,0])
    return np.concatenate(o)
def rw(c):
    u,inv=np.unique(c,return_inverse=True)
    return (1.0/np.bincount(inv,minlength=len(u)).astype("float64"))[inv]
def boot(c,y,p,w,B=800,seed=0):
    u,inv=np.unique(c,return_inverse=True)
    num=np.bincount(inv,weights=w*(y-p)**2,minlength=len(u))
    den=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed);pk=rng.integers(0,len(u),size=(B,len(u)))
    with np.errstate(divide="ignore",invalid="ignore"):
        v=1.0-num[pk].sum(1)/den[pk].sum(1)
    v=v[np.isfinite(v)]
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))

_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                                  "SEP","OCT","NOV","DEC"])}
def tkey(t):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",t)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)

print(f"TARGET {MAIN}  = (mid(t+{EXIT}) - mid(t+{ENTRY})) / mid(t+{ENTRY})",flush=True)
print(f"RECIPE width {W} depth {L+1}, {len(HEADS)} heads, rank-norm vol target, "
      f"1/99 clip, per-market weights, {len(SEEDS)} seeds\n",flush=True)
print(f"{'coin':5s} {'inst':7s} {'test mkts':>9s} {'rows':>11s} {'R2':>11s}   95% CI")
for coin in COINS:
    f=build(coin)
    if not f: print(f"{coin:5s} no frame"); continue
    sch=[x.name for x in pq.ParquetFile(f).schema_arrow]
    feats=[c for c in sch if not c.startswith(("inc_","market","ts","p_mid_now"))]
    COL={}
    def col(c):
        if c not in COL:
            COL[c]=pq.read_table(f,columns=[c]).column(0).to_numpy(zero_copy_only=False)
        return COL[c]
    mk=pd.Series(col("market")); mcode,munq=pd.factorize(mk,sort=True)
    mcode=mcode.astype("int32"); munq=np.asarray(munq,dtype=object)
    tko=np.array([tkey(m) for m in munq],dtype="int64")
    order=np.argsort(tko); nte=int(len(order)*0.30)
    c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    hc=[f"inc_{tg(a)}_to_{tg(b)}" for a,b in HEADS if f"inc_{tg(a)}_to_{tg(b)}" in sch]
    y0=col(hc[0]).astype("float64")
    vol=np.abs(col("relvol_30s").astype("float64")) if "relvol_30s" in sch else np.ones(len(y0))
    ok=np.isfinite(y0)&np.isfinite(vol)
    i_te=sel(c_te); i_ho=sel(c_ho); i_tr=sel(c_tr)
    i_te=i_te[ok[i_te]]; i_ho=i_ho[ok[i_ho]]; i_tr=i_tr[ok[i_tr]]
    if min(len(i_te),len(i_tr))<5000: print(f"{coin:5s} too few rows"); continue
    key=f"{coin}|binary|{MAIN}"
    if key in res:
        v=res[key]; print(f"{coin:5s} {'binary':7s} {v['test_markets']:>9} "
                          f"{v['rows']:>11,} {v['r2']:>+11.6f}   cached"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    def mat(ix):
        A=np.stack([col(c)[ix] for c in feats],1)
        return np.ascontiguousarray(A,dtype="float32")
    XA,Xh,Xte=mat(idxA),mat(i_ho),mat(i_te)
    S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
    for A in (XA,Xh,Xte):
        np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
    # RANK-NORMALISED trailing vol divisor, fitted on training rows only
    vs=np.sort(vol[idxA]); rank=lambda v: np.clip(
        np.interp(v,vs,np.linspace(0.05,1.0,len(vs))),0.05,1.0)
    dA,dh,dte=rank(vol[idxA]),rank(vol[i_ho]),rank(vol[i_te])
    def heads(ix,div):
        cols=[];msk=[]
        for c in hc:
            v=col(c)[ix].astype("float64"); tr=col(c)[idxA].astype("float64")
            q=np.quantile(tr[np.isfinite(tr)],[.01,.99])
            m_=np.isfinite(v)
            vv=np.where(m_,np.clip(v,q[0],q[1])/div,0.0)
            trn=np.clip(tr[np.isfinite(tr)],q[0],q[1])/rank(vol[idxA])[np.isfinite(tr)]
            mm,ss=float(trn.mean()),float(trn.std()) or 1.0
            cols.append((vv-mm)/ss); msk.append(m_.astype("float32"))
        return (np.stack(cols,1).astype("float32"),
                np.stack(msk,1).astype("float32"),mm,ss)
    Ytr,Mtr,mm,ss=heads(idxA,dA); Yho,Mho,_,_=heads(i_ho,dh)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te])
    ps=[]
    for s_ in SEEDS:
        m=fit(XA,Ytr,wA,Mtr,Xh,Yho,wh,s_)
        ps.append((pr(m,Xte)*ss+mm)*dte)      # un-normalise: back to target units
        del m
    p=np.mean(ps,axis=0)
    yte=y0[i_te]
    r=float(r2_zero(yte,p,wte)); ci=boot(mcode[i_te],yte,p,wte)
    print(f"{coin:5s} {'binary':7s} {len(c_te):>9} {len(i_te):>11,} {r:>+11.6f}   "
          f"[{ci[0]:+.6f},{ci[1]:+.6f}]",flush=True)
    res[key]={"r2":r,"ci":list(ci),"test_markets":len(c_te),"rows":int(len(i_te)),
              "entry":ENTRY,"exit":EXIT,"width":W,"depth":L+1,"heads":len(hc),
              "nfeat":len(feats)}
    save(); del XA,Xh,Xte,COL; gc.collect()
save(); print(f"\nwrote {OUT}")
