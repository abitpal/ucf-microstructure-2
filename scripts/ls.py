"""EXACT-LAG INCREMENT SWEEP: how much edge survives each amount of latency.

Every increment number in this study so far resolved its entry price by
searchsorted over the SAMPLED panel, whose median inter-row gap is 22ms on BTC.
So a 20ms entry is unresolvable 53% of the time and even 0.1s is wrong for 26%
of rows -- their nominal entry lands anywhere out to 945ms. The full-resolution
mid tape (ts, mid at EVERY event) fixes that: entry and exit mids now come from
the complete series while features stay on the 25-event feature grid.

Measured feed latency is ~25ms one-way with a ~60ms round trip, so the lags
that matter are 0.02-0.1s, not the 0.1s this study has been assuming.

  (a, b) pairs      entry lag a -> exit b
  0.02 -> 0.10      80ms hold: act as fast as the feed allows, exit where the
                    old target STARTED
  0.02 -> 1.00      fastest realistic entry, 1s hold
  0.05 -> 1.00      round-trip-realistic entry
  0.10 -> 1.00      what the study has been using (now exactly resolved)
  0.02 -> 0.50
  0.10 -> 0.50

ALL MARK-TO-MID. These use the orderbook mid at both ends; they do NOT pay the
spread. A buy fills at the ask, so a real round trip gives up ~2x the half
spread, measured at 0.45c on these books against ~0.52c of edge -- so a
positive R^2 here is necessary for a trade to exist, not sufficient.

Per-market join on ts, so nothing depends on row-order assumptions between the
panels, the frame and the tape.
"""
import sys, os, re, glob, json, gc; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from tools.panel_model import scale_free, r2_zero
PAIRS=[(0.02,0.10),(0.02,0.50),(0.02,1.00),(0.05,1.00),(0.10,0.50),(0.10,1.00)]
PANELS=os.environ.get("LS_PANELS","/var/lib/kalshi-study/panel_rich")
MIDS=os.environ.get("LS_MIDS","/var/lib/kalshi-study/midtape")
OUTF=os.environ.get("LS_FRAME","/mnt/scratch/frames/frame_lag.parquet")
OUT="/var/lib/kalshi-study/LAGSWEEP.json"

def tag(x): return f"{x:g}".replace(".","p")
INC=[f"ry_k_inc_{tag(a)}_to_{tag(b)}s" for a,b in PAIRS]

def build():
    paths=sorted(glob.glob(f"{PANELS}/panel_KX*.parquet"))
    print(f"{len(paths)} panels -> {OUTF}",flush=True)
    w=None; total=0; miss=0; feats=None
    for i,p in enumerate(paths,1):
        tk=re.sub(r"^panel_|\.parquet$","",os.path.basename(p))
        mp=f"{MIDS}/mid_{tk}.parquet"
        if not os.path.exists(mp): miss+=1; continue
        raw=pd.read_parquet(p)
        if "market" not in raw.columns: raw["market"]=tk
        keep=[c for c in raw.columns if c.startswith(("k_","p_","y_","ry_")) or c=="market"]
        sf=scale_free(raw[keep])
        if feats is None:
            feats=[c for c in sf.columns if not c.startswith(("y_","ry_","market","_day"))]
        ts=raw["ts"].to_numpy(dtype="float64")
        mt=pq.read_table(mp,columns=["ts","mid"])
        mts=mt.column("ts").to_numpy(zero_copy_only=False).astype("float64")
        mmd=mt.column("mid").to_numpy(zero_copy_only=False).astype("float64")
        o=np.argsort(mts); mts=mts[o]; mmd=mmd[o]
        def at(lag):
            j=np.searchsorted(mts,ts+lag,side="left")
            ok=j<len(mts)
            v=np.where(ok,mmd[np.clip(j,0,len(mts)-1)],np.nan)
            # reject a resolution further than 25% past the requested lag --
            # that is the sampled-grid failure this whole exercise exists to
            # avoid, and it must not be silently tolerated here either
            got=np.where(ok,mts[np.clip(j,0,len(mts)-1)]-ts,np.nan)
            return np.where(got<=lag*1.25+0.005,v,np.nan)
        cols={}
        for a,b in PAIRS:
            ma,mb=at(a),at(b)
            with np.errstate(divide="ignore",invalid="ignore"):
                cols[f"ry_k_inc_{tag(a)}_to_{tag(b)}s"]=np.where(
                    np.abs(ma)>1e-12,(mb-ma)/ma,np.nan).astype("float32")
        d={"market":raw["market"].to_numpy(dtype=object),
           "ts":ts.astype("float64")}
        for c in feats: d[c]=sf[c].to_numpy(dtype="float32")
        d.update(cols)
        tbl=pa.table(d)
        if w is None: w=pq.ParquetWriter(OUTF,tbl.schema,compression="zstd")
        w.write_table(tbl); total+=tbl.num_rows
        del raw,sf,tbl,d,cols; gc.collect()
        if i%150==0: print(f"  {i}/{len(paths)}  {total:,} rows  "
                           f"missing-mid {miss}",flush=True)
    if w is not None: w.close()
    print(f"wrote {OUTF}  {total:,} rows  (skipped {miss} without a mid tape)",flush=True)

if not os.path.exists(OUTF) or os.environ.get("LS_REBUILD"):
    build()

import torch, torch.nn as nn
torch.set_num_threads(int(os.environ.get("LS_THREADS","10")))
sch=[f.name for f in pq.ParquetFile(OUTF).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","ts","_day"))]
have=[c for c in INC if c in sch]
print(f"\n{len(feats)} features; increments present: {have}",flush=True)
COL={}
def col(c):
    if c not in COL:
        COL[c]=pq.read_table(OUTF,columns=[c]).column(0).to_numpy(zero_copy_only=False)
    return COL[c]
mk=pd.Series(col("market"))
mcode,munq=pd.factorize(mk,sort=True); mcode=mcode.astype("int32")
munq=np.asarray(munq,dtype=object)
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                                  "SEP","OCT","NOV","DEC"])}
def tkey(t):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",t)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)
tk_of=np.array([tkey(m) for m in munq],dtype="int64")
order=np.argsort(tk_of); nte=int(len(order)*0.30)
c_te=order[-nte:]; pool=order[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
sel=lambda cs: np.flatnonzero(np.isin(mcode,cs))
class Net(nn.Module):
    def __init__(s,nin,w=320,L=5,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))
def fit(X,y,w,Xv,yv,wv,seed,steps=4000,lr=1e-3,wd=0.3,pat=10,BS=1024):
    torch.manual_seed(seed); m=Net(X.shape[1])
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=lr,total_steps=steps,pct_start=.15)
    A=torch.from_numpy(X);T=torch.from_numpy(y).float().view(-1,1)
    W=torch.from_numpy(w).float().view(-1,1);V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//20)
    while st<steps and bad<pat:
        pm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=pm[i:i+BS];o.zero_grad()
            ((W[ix]*(m(A[ix])-T[ix])**2).sum()/W[ix].sum()).backward();o.step()
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
def pr(m,A,ch=1_500_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy().ravel())
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
res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT))
    except Exception: res={}
print(f"\n{'target':30s} {'entry':>6s} {'hold':>6s} {'n_rows':>11s} {'R2':>10s}   95% CI")
for (a,b),t in zip(PAIRS,INC):
    if t not in sch: continue
    if f"{t}" in res: print(f"{t:30s} cached"); continue
    y=col(t).astype("float64")
    ok=np.isfinite(y)
    i_te=sel(c_te); i_ho=sel(c_ho); i_tr=sel(c_tr)
    i_te=i_te[ok[i_te]]; i_ho=i_ho[ok[i_ho]]; i_tr=i_tr[ok[i_tr]]
    if min(len(i_te),len(i_tr))<5000: print(f"{t:30s} too few rows"); continue
    idxA=i_tr if len(i_tr)<=1_000_000 else np.random.default_rng(0).choice(i_tr,1_000_000,replace=False)
    def mat(ix):
        A=np.stack([col(c)[ix] for c in feats],1)
        return np.ascontiguousarray(A,dtype="float32")
    XA,Xh,Xte=mat(idxA),mat(i_ho),mat(i_te)
    S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
    for A in (XA,Xh,Xte):
        np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
    q=np.quantile(y[i_tr],[.01,.99])
    yA=y[idxA].clip(q[0],q[1]); ym=float(yA.mean()); ys=float(yA.std()) or 1.0
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te])
    ps=[pr(fit(XA,((yA-ym)/ys).astype("float32"),wA,Xh,
                ((y[i_ho]-ym)/ys),wh,s_),Xte)*ys+ym for s_ in (11,23)]
    p=np.mean(ps,axis=0)
    r=float(r2_zero(y[i_te],p,wte)); ci=boot(mcode[i_te],y[i_te],p,wte)
    print(f"{t:30s} {a:6.2f} {b-a:6.2f} {len(i_te):>11,} {r:+10.6f}   "
          f"[{ci[0]:+.6f},{ci[1]:+.6f}]",flush=True)
    res[t]={"r2":r,"ci":list(ci),"entry":a,"hold":b-a,"exit":b,
            "test_markets":len(c_te),"rows":int(len(i_te))}
    json.dump(res,open(OUT,"w"),indent=1)
    del XA,Xh,Xte; gc.collect()
json.dump(res,open(OUT,"w"),indent=1); print(f"\nwrote {OUT}")
