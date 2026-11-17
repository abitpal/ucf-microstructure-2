"""IS VOL-NORMALISING MORE STABLE? -- measured as dispersion, not mean.

The claim that matters was never "higher R^2". The dashboard that established
causal vol-norming said it plainly: lower fold spread, worst fold turned
positive, "robustness is not a higher mean". My first test scored pooled mean
R^2 on one test set, which cannot see that -- so it was the wrong instrument
and its negative verdict does not bear on the claim.

Measured here, all in RAW units (predictions back-transformed by multiplying by
the divisor, then scored against the untouched target):

  pooled R^2            the headline
  per-market R^2        mean, sd, 10th percentile, % of markets negative
  worst-decile mean     the stability number the original claim was about

Three divisors, because the first one may be the whole problem: relvol_30s is
vol/mid and the mid runs down to 15 deci-cents, so the divisor itself has a
1/mid blow-up and dividing by it can manufacture outliers rather than damp
them. Ridge back-transformed to -0.41962 in the first run, which is not
"marginally worse" -- it is a divisor exploding.

  raw          no normalisation
  volnorm      / relvol_30s                       (as first implemented)
  volnorm_w    / winsorised relvol_30s            (5th-95th pct, per fold)
  volnorm_r    / rank-normalised relvol_30s       (bounded by construction)
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from tools.panel_model import r2_zero
torch.set_num_threads(8)
F="/var/lib/kalshi-study/frame_big.parquet"
OUT="/var/lib/kalshi-study/VOLSTAB.json"
CELLS=[("BTC","ry_k_1s"),("XRP","ry_k_1s")]
MAXROWS=900_000; BS=1024; SEEDS=(11,23)
sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
mser=pd.read_parquet(F,columns=["market"])["market"]
mcode,munq=pd.factorize(mser,sort=True); mcode=mcode.astype("int32")
munq=np.asarray(munq,dtype=object); del mser; gc.collect()
coin_of=np.array([m.split("15M")[0].replace("KX","") for m in munq],dtype=object)
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                                  "SEP","OCT","NOV","DEC"])}
def tkey(tk):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",tk)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)
tkey_of=np.array([tkey(m) for m in munq],dtype="int64")
VOL=pq.read_table(F,columns=["relvol_30s"]).column(0).to_numpy(zero_copy_only=False).astype("float64")
# ONE COIN'S ROWS AT A TIME. A full-length matrix is 34.6M x 106 x 4 = 13.7 GiB
# before any per-cell copy, and this job was OOM-killed at a 30GB cap while four
# other studies held the rest of the box. BTC is 20.9M of those rows and XRP
# 4.6M, so loading per coin cuts the resident set by 40-85% and the column-at-a-
# time read keeps the transient to one column (~0.28 GiB).
_COLCACHE={}
def load_rows(ix):
    A=np.empty((len(ix),len(feats)),dtype="float32")
    for j,c in enumerate(feats):
        col=_COLCACHE.get(c)
        if col is None:
            col=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False)
        A[:,j]=col[ix].astype("float32",copy=False)
        del col
        if j%25==0: gc.collect()
    np.nan_to_num(A,copy=False)
    return A
class Net(nn.Module):
    def __init__(s,nin,w=165,L=3,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))
def fit(Xt,y,w,Xv,yv,wv,seed,steps=5000,lr=1e-3,wd=3.0,pat=12):
    torch.manual_seed(seed); m=Net(Xt.shape[1])
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=lr,total_steps=steps,pct_start=.15)
    A=torch.from_numpy(Xt);T=torch.from_numpy(y).float().view(-1,1)
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
def pr(m,A,ch=2_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy().ravel())
    return np.concatenate(o)
def rw(c):
    u,inv=np.unique(c,return_inverse=True)
    return (1.0/np.bincount(inv,minlength=len(u)).astype("float64"))[inv]
def permkt(c,y,p,w):
    u,inv=np.unique(c,return_inverse=True)
    num=np.bincount(inv,weights=w*(y-p)**2,minlength=len(u))
    den=np.bincount(inv,weights=w*y*y,minlength=len(u))
    ok=den>0
    return 1.0-num[ok]/den[ok]
res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT)); print(f"resuming with {len(res)} keys",flush=True)
    except Exception: res={}
for coin,t in CELLS:
    Y=pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float64")
    oc=np.flatnonzero(coin_of==coin); oc=oc[np.argsort(tkey_of[oc])]
    nte=int(len(oc)*.30); c_te=oc[-nte:]; pool=oc[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    good=np.isfinite(Y)&np.isfinite(VOL)&(VOL>0)
    i_te=g(c_te); i_ho=g(c_ho); i_tr=g(c_tr)
    i_te=i_te[good[i_te]]; i_ho=i_ho[good[i_ho]]; i_tr=i_tr[good[i_tr]]
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    XA=load_rows(idxA); Xh=load_rows(i_ho); Xte=load_rows(i_te); gc.collect()
    S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S; gc.collect()
    for A in (XA,Xh,Xte):
        np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ct=mcode[i_te]
    yte=Y[i_te]
    vlo,vhi=np.quantile(VOL[i_tr],[.05,.95])
    rk=pd.Series(VOL[i_tr]).rank(pct=True)
    def divisor(kind,ix):
        if kind=="raw": return np.ones(len(ix))
        v=VOL[ix].copy()
        if kind=="volnorm": return np.maximum(v,1e-5)
        if kind=="volnorm_w": return np.clip(v,vlo,vhi)
        q=np.interp(v,np.sort(VOL[i_tr]),np.linspace(.05,1.0,len(i_tr)))
        return np.clip(q,.05,1.0)
    print(f"\n=== {coin}/{t} | test {len(c_te)} mkts / {len(i_te):,} rows ===",flush=True)
    print(f"{'divisor':11s} {'pooled':>9s} {'pm mean':>9s} {'pm sd':>8s} "
          f"{'pm p10':>9s} {'worst dec':>10s} {'% neg':>6s}",flush=True)
    for kind in ("raw","volnorm","volnorm_w","volnorm_r"):
        dA=divisor(kind,idxA); dh=divisor(kind,i_ho); dte=divisor(kind,i_te)
        yA=Y[idxA]/dA; q=np.quantile(yA,[.01,.99]); yA=yA.clip(q[0],q[1])
        yh=Y[i_ho]/dh
        ps=[pr(fit(XA,yA,wA,Xh,yh,wh,s_),Xte) for s_ in SEEDS]
        p_raw=np.mean(ps,axis=0)*dte          # back to RAW units
        pm=permkt(ct,yte,p_raw,wte)
        pooled=float(r2_zero(yte,p_raw,wte))
        wd=float(np.mean(np.sort(pm)[:max(1,len(pm)//10)]))
        print(f"{kind:11s} {pooled:+9.5f} {pm.mean():+9.5f} {pm.std():8.5f} "
              f"{np.percentile(pm,10):+9.5f} {wd:+10.5f} {100*np.mean(pm<0):5.1f}%",flush=True)
        res[f"{coin}|{t}|{kind}"]={"pooled":pooled,"pm_mean":float(pm.mean()),
            "pm_sd":float(pm.std()),"pm_p10":float(np.percentile(pm,10)),
            "worst_decile":wd,"pct_neg":float(np.mean(pm<0))}
        json.dump(res,open(OUT,"w"),indent=1)
    del XA,Xh,Xte; gc.collect()
print(f"\nwrote {OUT}")
