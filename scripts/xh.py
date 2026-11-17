"""IS THE 0.1s MODEL A BETTER 1s PREDICTOR THAN THE 1s MODEL?

The 0.1s label is much less noisy than the 1s label (+0.024 vs +0.008 on BTC),
and both are driven by the same book pressure. So a model fitted to the clean
surrogate may transfer to the 1s horizon better than one fitted to the noisy
target directly -- the usual bias/variance trade of learning from a proxy.

THE COMPARISON IS ONLY FAIR WITH A SCALE FACTOR. A 0.1s prediction is of a
smaller-magnitude quantity, so scoring it raw against the 1s target penalises
the units rather than the information. Each prediction therefore gets one
scalar `a` fitted by weighted least squares ON THE HOLDOUT MARKETS -- never on
test -- and the scaled prediction is what gets scored. That makes the question
purely "which prediction carries more information about this horizon".

Every (source model, scored target) pair is reported, so the matrix is
symmetric: 0.1s-model on the 1s target, 1s-model on the 0.1s target, each
against its own-horizon baseline, with a paired market bootstrap.

The increment is included as a third scored target, where the 0.1s model should
be actively bad: the increment is 1s MINUS 0.1s, so whatever the 0.1s model
captures is the part being subtracted out.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("XH_THREADS","6")))
F="/var/lib/kalshi-study/frame_big.parquet"
OUT="/var/lib/kalshi-study/XHORIZON.json"
COINS=os.environ.get("XH_COINS","BTC,XRP").split(",")
SRC=["ry_k_0p1s","ry_k_1s","ry_k_5s"]
SCORED=["ry_k_0p1s","ry_k_1s","ry_k_inc_0p1_to_1s"]
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
Y={t:pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float64")
   for t in sorted(set(SRC+SCORED))}
X=np.empty((len(mcode),len(feats)),dtype="float32")
for j,c in enumerate(feats):
    X[:,j]=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False).astype("float32")
np.nan_to_num(X,copy=False); gc.collect()
class Net(nn.Module):
    def __init__(s,nin,w=165,L=3,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))
def fit(Xt,y,w,Xv,yv,wv,seed,steps=6000,lr=1e-3,wd=3.0,pat=12):
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
def paired(c,y,pa,pb,w,B=2000,seed=0):
    u,inv=np.unique(c,return_inverse=True)
    na=np.bincount(inv,weights=w*(y-pa)**2,minlength=len(u))
    nb=np.bincount(inv,weights=w*(y-pb)**2,minlength=len(u))
    de=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed);pk=rng.integers(0,len(u),size=(B,len(u)))
    with np.errstate(divide="ignore",invalid="ignore"):
        d=(nb[pk].sum(1)-na[pk].sum(1))/de[pk].sum(1)
    d=d[np.isfinite(d)];lo,hi=np.percentile(d,2.5),np.percentile(d,97.5)
    return float(d.mean()),float(lo),float(hi),bool((lo>0)==(hi>0))
res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT)); print(f"resuming with {len(res)} keys",flush=True)
    except Exception: res={}
for coin in COINS:
    oc=np.flatnonzero(coin_of==coin); oc=oc[np.argsort(tkey_of[oc])]
    nte=int(len(oc)*.30); c_te=oc[-nte:]; pool=oc[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    good=np.ones(len(mcode),bool)
    for t in set(SRC+SCORED): good &= np.isfinite(Y[t])
    i_te=g(c_te)[good[g(c_te)]]; i_ho=g(c_ho)[good[g(c_ho)]]; i_tr=g(c_tr)[good[g(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"skip {coin}"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    S=X[idxA[:300_000]].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
    mk=lambda ix:(lambda A:(np.clip(A,lo,hi,out=A),A.__isub__(mu),A.__itruediv__(sg),
                            np.nan_to_num(A,copy=False),A)[-1])(X[ix].copy())
    XA,Xh,Xte=mk(idxA),mk(i_ho),mk(i_te)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ct=mcode[i_te]
    PH={};PT={}
    for s in SRC:
        q=np.quantile(Y[s][i_tr],[.01,.99])
        yA=Y[s][idxA].clip(q[0],q[1])
        ms=[fit(XA,yA,wA,Xh,Y[s][i_ho],wh,sd_) for sd_ in SEEDS]
        PH[s]=np.mean([pr(m,Xh) for m in ms],axis=0)
        PT[s]=np.mean([pr(m,Xte) for m in ms],axis=0)
        del ms; gc.collect()
    print(f"\n{'='*86}\n=== {coin} | train {len(idxA):,} rows, test {len(c_te)} mkts / "
          f"{len(i_te):,} rows\n{'='*86}",flush=True)
    print("rows: model TRAINED on   |  columns: target SCORED on "
          "(one holdout-fitted scalar each)",flush=True)
    hdr="".join(f"{t:>26s}" for t in SCORED)
    print(f"{'trained on':16s}{hdr}",flush=True)
    tab={}
    for s in SRC:
        row=""
        for t in SCORED:
            a=float(np.sum(wh*PH[s]*Y[t][i_ho])/max(np.sum(wh*PH[s]**2),1e-18))
            p=PT[s]*a
            r=float(r2_zero(Y[t][i_te],p,wte))
            tab[(s,t)]=(r,a,p)
            row+=f"   {r:+.6f} (a={a:5.2f})"
        print(f"{s:16s}{row}",flush=True)
        for t in SCORED:
            res[f"{coin}|{s}->{t}"]={"r2":tab[(s,t)][0],"scale":tab[(s,t)][1]}
    print("\n  PAIRED: cross-horizon model vs the model trained on that target",flush=True)
    for t in SCORED:
        if t not in SRC: 
            base_s=t if t in SRC else "ry_k_1s"
        else:
            base_s=t
        for s in SRC:
            if s==base_s: continue
            dm,dlo,dhi,sig=paired(ct,Y[t][i_te],tab[(s,t)][2],tab[(base_s,t)][2],wte)
            print(f"    on {t:22s}: trained-on-{s:11s} vs trained-on-{base_s:11s} "
                  f"dR2={dm:+.6f} CI[{dlo:+.6f},{dhi:+.6f}] "
                  f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
            res[f"{coin}|paired|{t}|{s}_vs_{base_s}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    json.dump(res,open(OUT,"w"),indent=1)
    del XA,Xh,Xte,PH,PT; gc.collect()
json.dump(res,open(OUT,"w"),indent=1); print(f"\nwrote {OUT}")
