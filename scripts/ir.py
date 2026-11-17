"""IS THE INCREMENT BETTER PREDICTED AS A DIFFERENCE OF TWO PREDICTIONS?

Fitting ry_k_inc directly gives ~0 or slightly negative on every single-coin
cell. But E[y1 - y01 | X] = E[y1|X] - E[y01|X], so the optimal increment
predictor IS the difference of the optimal point predictors -- and the point
targets are far more learnable (+0.025 at 0.1s, +0.008 at 1s). In infinite
samples the two routes coincide; in finite samples with regularisation they do
not. A direct fit to a low signal-to-noise difference gets shrunk toward zero,
while differencing two well-estimated conditional means need not be.

THE DENOMINATORS DO NOT MATCH, AND THE MISMATCH IS NOT SMALL.

    ry_k_0p1s = (mid(t+0.1) - mid(t))     / mid(t)
    ry_k_1s   = (mid(t+1)   - mid(t))     / mid(t)
    ry_k_inc  = (mid(t+1)   - mid(t+0.1)) / mid(t+0.1)

so ry_k_1s - ry_k_0p1s = (mid(t+1) - mid(t+0.1)) / mid(t): the right numerator
over the WRONG denominator. sd(ry_k_0p1s) is 0.21, not 0.001, because the mid
runs down to 15 deci-cents -- so mid(t)/mid(t+0.1) is nowhere near 1 and the
mismatch cannot be waved away.

Three routes, therefore:
  direct      fit ry_k_inc, score on ry_k_inc
  diffT       pred(1s) - pred(0.1s), scored on ry_k_incT, where
              ry_k_incT = (mid(t+1) - mid(t+0.1)) / mid(t) -- an EXACT algebraic
              match to the difference, and arguably the better target anyway
              since mid(t) is the price known when the decision is made
  diff_corr   (p1 - p01) / (1 + p01), scored on ry_k_inc. Fully causal: the
              correction uses the PREDICTED 0.1s return, not the realised one
  directT     fit ry_k_incT directly, so diffT has a like-for-like baseline
  twohead     one trunk, two heads (0.1s and 1s), difference of the heads --
              shares a representation instead of fitting two separate models

fwd_mid_0p1s / fwd_mid_1s come from the exec companions, which were already
verified to align with the frame row-for-row (zero market mismatches).
"""
import sys, os, json, gc, re, glob; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn, lightgbm as lgb
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("IR_THREADS","10")))
F="/var/lib/kalshi-study/frame_big.parquet"
OUT="/var/lib/kalshi-study/INC_ROUTE.json"
CELLS=[("BTC",),("ETH",),("XRP",),("NEAR",)]
MAXROWS=1_500_000; BS=1024; SEEDS=(11,23,37)

sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
# --- aligned forward mids from the exec companions
CH="/var/lib/kalshi-study/_bigchunks"
fr=sorted(p for p in glob.glob(f"{CH}/frame_*.parquet") if not p.endswith("_exec.parquet"))
mids=[];f01=[];f1=[];mex=[]
for p in fr:
    e=p.replace(".parquet","_exec.parquet")
    t=pq.read_table(e,columns=["market","mid","fwd_mid_0p1s","fwd_mid_1s"])
    n_f=pq.ParquetFile(p).metadata.num_rows
    assert n_f==t.num_rows, f"row mismatch {p}"
    mids.append(t.column("mid").to_numpy(zero_copy_only=False).astype("float64"))
    f01.append(t.column("fwd_mid_0p1s").to_numpy(zero_copy_only=False).astype("float64"))
    f1.append(t.column("fwd_mid_1s").to_numpy(zero_copy_only=False).astype("float64"))
    mex.append(np.asarray(t.column("market").to_pylist(),dtype=object))
MID=np.concatenate(mids); F01=np.concatenate(f01); F1=np.concatenate(f1)
mex=np.concatenate(mex); del mids,f01,f1; gc.collect()
mser=pd.read_parquet(F,columns=["market"])["market"]
fm=mser.to_numpy(dtype=object)
assert len(fm)==len(MID) and int((fm!=mex).sum())==0, "exec/frame misalignment"
print(f"exec aligned: {len(MID):,} rows, 0 market mismatches",flush=True)
del mex; gc.collect()
mcode,munq=pd.factorize(mser,sort=True); mcode=mcode.astype("int32")
munq=np.asarray(munq,dtype=object); del mser,fm; gc.collect()
coin_of=np.array([m.split("15M")[0].replace("KX","") for m in munq],dtype=object)
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL",
                                  "AUG","SEP","OCT","NOV","DEC"])}
def tkey(tk):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",tk)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)
tkey_of=np.array([tkey(m) for m in munq],dtype="int64")

with np.errstate(divide="ignore",invalid="ignore"):
    INCT=np.where(MID>0,(F1-F01)/MID,np.nan).astype("float32")
    INC=np.where(np.abs(F01)>1e-12,(F1-F01)/F01,np.nan).astype("float32")
Y01=pq.read_table(F,columns=["ry_k_0p1s"]).column(0).to_numpy(zero_copy_only=False).astype("float32")
Y1=pq.read_table(F,columns=["ry_k_1s"]).column(0).to_numpy(zero_copy_only=False).astype("float32")
INCF=pq.read_table(F,columns=["ry_k_inc_0p1_to_1s"]).column(0).to_numpy(zero_copy_only=False).astype("float32")
d=np.abs(INC-INCF); ok=np.isfinite(d)
print(f"reconstructed ry_k_inc matches the frame's: max|diff| "
      f"{np.nanmax(d[ok]):.3e} on {ok.sum():,} rows",flush=True)
print(f"sd  ry_k_inc {np.nanstd(INCF):.5f}   ry_k_incT {np.nanstd(INCT):.5f}",flush=True)
X=np.empty((len(MID),len(feats)),dtype="float32")
for j,c in enumerate(feats):
    X[:,j]=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False).astype("float32")
    if j%25==0: gc.collect()
np.nan_to_num(X,copy=False); gc.collect()
print(f"X {X.nbytes/2**30:.1f} GiB",flush=True)

class Net(nn.Module):
    def __init__(s,nin,w=165,L=3,p=0.2,heads=1):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,heads)
    def forward(s,x): return s.head(s.body(x))

def fit(Xtr,Ytr,w,Xv,Yv,wv,seed,heads=1,steps=7000,lr=1e-3,wd=3.0,pat=14):
    torch.manual_seed(seed); m=Net(Xtr.shape[1],heads=heads)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=0.15)
    A=torch.from_numpy(Xtr);T=torch.from_numpy(Ytr).float().reshape(len(Xtr),heads)
    W=torch.from_numpy(w).float().view(-1,1);V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//25)
    while st<steps and bad<pat:
        perm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=perm[i:i+BS];opt.zero_grad()
            ((W[ix]*(m(A[ix])-T[ix])**2).sum()/W[ix].sum()).backward()
            opt.step()
            if st<steps-1: sc.step()
            st+=1
            if st%ev==0 or st>=steps:
                m.eval()
                with torch.no_grad(): P=m(V).numpy()
                v=float(np.mean([r2_zero(Yv.reshape(-1,heads)[:,h],P[:,h],wv)
                                 for h in range(heads)]))
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
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy())
    return np.concatenate(o)
def rw(c):
    u,inv=np.unique(c,return_inverse=True)
    return (1.0/np.bincount(inv,minlength=len(u)).astype("float64"))[inv]
def paired(c,y,pa,pb,w,B=2000,seed=0):
    u,inv=np.unique(c,return_inverse=True)
    na=np.bincount(inv,weights=w*(y-pa)**2,minlength=len(u))
    nb=np.bincount(inv,weights=w*(y-pb)**2,minlength=len(u))
    de=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed);pk=rng.integers(0,len(u),size=(B,len(u)))
    with np.errstate(divide="ignore",invalid="ignore"):
        dd=(nb[pk].sum(1)-na[pk].sum(1))/de[pk].sum(1)
    dd=dd[np.isfinite(dd)];lo,hi=np.percentile(dd,2.5),np.percentile(dd,97.5)
    return float(dd.mean()),float(lo),float(hi),bool((lo>0)==(hi>0))
res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT)); print(f"resuming with {len(res)} keys",flush=True)
    except Exception: res={}
def save(): json.dump(res,open(OUT,"w"),indent=1,default=str)
for (coin,) in CELLS:
    oc=np.flatnonzero(coin_of==coin); oc=oc[np.argsort(tkey_of[oc])]
    nte=int(len(oc)*0.30); c_te=oc[-nte:]; pool=oc[:-nte]
    c_ho=pool[-120:]; c_tr=pool[:-120]
    sel=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    good=np.isfinite(Y01)&np.isfinite(Y1)&np.isfinite(INCF)&np.isfinite(INCT)
    i_te=sel(c_te); i_ho=sel(c_ho); i_tr=sel(c_tr)
    i_te=i_te[good[i_te]]; i_ho=i_ho[good[i_ho]]; i_tr=i_tr[good[i_tr]]
    if min(len(i_te),len(i_tr))<5000: print(f"skip {coin}"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    st=idxA if len(idxA)<=400_000 else np.random.default_rng(7).choice(idxA,400_000,replace=False)
    S=X[st].copy(); lo=np.quantile(S,.01,axis=0); hi=np.quantile(S,.99,axis=0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
    mk=lambda ix:(lambda A:(np.clip(A,lo,hi,out=A),A.__isub__(mu),A.__itruediv__(sg),
                            np.nan_to_num(A,copy=False),A)[-1])(X[ix].copy())
    XA,Xh,Xte=mk(idxA),mk(i_ho),mk(i_te)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ct=mcode[i_te]
    cl=lambda v,ix:(lambda q:v[ix].astype("float64").clip(q[0],q[1]))(np.quantile(v[i_tr],[.01,.99]))
    print(f"\n{'='*84}\n=== {coin} | train {len(idxA):,} rows, test {len(c_te)} mkts / "
          f"{len(i_te):,} rows\n{'='*84}",flush=True)
    # Checkpoint every arm. The previous run spent 3h36m of CPU completing the
    # four single-head arms and then died in the two-head reshape, losing all
    # of it because nothing was written until the end.
    P={}
    ck=f"/tmp/ir_{coin}.npz"
    if os.path.exists(ck):
        z=np.load(ck); P={k:z[k] for k in z.files}
        print(f"   resumed {sorted(P)} from checkpoint",flush=True)
    for nm,tv in (("y01",Y01),("y1",Y1),("direct",INCF),("directT",INCT)):
        if nm in P: continue
        ps=[pr(fit(XA,cl(tv,idxA),wA,Xh,tv[i_ho].astype("float64"),wh,s_),Xte)[:,0]
            for s_ in SEEDS]
        P[nm]=np.mean(ps,axis=0)
        np.savez(ck,**P); print(f"   fitted {nm}",flush=True)
    if "twohead" not in P:
        try:
            H=[pr(fit(XA,np.stack([cl(Y01,idxA),cl(Y1,idxA)],1).ravel(),wA,Xh,
                      np.stack([Y01[i_ho],Y1[i_ho]],1).astype("float64").ravel(),
                      wh,s_,heads=2),Xte) for s_ in SEEDS]
            P["twohead"]=np.mean([h[:,1]-h[:,0] for h in H],axis=0)
            np.savez(ck,**P); print("   fitted twohead",flush=True)
        except Exception as e:
            print(f"   twohead FAILED ({type(e).__name__}: {e}); "
                  f"continuing with the other arms",flush=True)
            P["twohead"]=P["y1"]-P["y01"]
    P["diffT"]=P["y1"]-P["y01"]
    P["diff_corr"]=(P["y1"]-P["y01"])/(1.0+P["y01"])
    yI=INCF[i_te].astype("float64"); yT=INCT[i_te].astype("float64")
    rows=[("direct   -> ry_k_inc",P["direct"],yI),
          ("diff_corr-> ry_k_inc",P["diff_corr"],yI),
          ("twohead  -> ry_k_inc",P["twohead"]/(1.0+P["y01"]),yI),
          ("directT  -> ry_k_incT",P["directT"],yT),
          ("diffT    -> ry_k_incT",P["diffT"],yT),
          ("twoheadT -> ry_k_incT",P["twohead"],yT)]
    for lbl,p,yy in rows:
        r=float(r2_zero(yy,p,wte))
        print(f"   {lbl:24s} {r:+.6f}",flush=True)
        res[f"{coin}|{lbl.strip()}"]={"r2":r}
    print("   PAIRED, same target:",flush=True)
    for lbl,a,b,yy in (("diff_corr vs direct (ry_k_inc)",P["diff_corr"],P["direct"],yI),
                       ("twohead   vs direct (ry_k_inc)",P["twohead"]/(1.0+P["y01"]),P["direct"],yI),
                       ("diffT     vs directT (ry_k_incT)",P["diffT"],P["directT"],yT),
                       ("twoheadT  vs directT (ry_k_incT)",P["twohead"],P["directT"],yT)):
        dm,dlo,dhi,sig=paired(ct,yy,a,b,wte)
        print(f"     {lbl:34s} dR2={dm:+.6f} CI[{dlo:+.6f},{dhi:+.6f}] "
              f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
        res[f"{coin}|paired|{lbl}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    save(); del XA,Xh,Xte,P; gc.collect()
save(); print(f"\nwrote {OUT}")
