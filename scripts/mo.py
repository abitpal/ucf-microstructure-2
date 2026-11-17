"""DOES EXTRA CAPACITY BUY EXTRA OBJECTIVES? -- the interaction, not the main effect.

Capacity on the 210-feature rich set wanted the largest net offered (width 640,
depth 6, both at the grid ceiling), against a 106-feature set where capacity was
flat across a 77x parameter range. So there is now headroom, and the question is
whether it can be spent on predicting MORE THINGS rather than just the one
target harder.

"Do auxiliary heads help" is the wrong question -- it was asked before and
answered "no", at width 96. The right one is an INTERACTION: does the gain from
adding objectives GROW with capacity? A small trunk has to steal representation
from the main task to serve the auxiliaries, so aux heads should hurt it; a
large trunk may have room for both. Those two predictions differ, and only a
2D design separates them.

  objectives  1 -> 2 -> 4 -> 8      (main; +0.1s; +5s,+30s; +4 book auxiliaries)
  capacity    (165,d4) (640,d6) (1536,d8)
  aux weight  0.1 and 1.0          (does the main task need protecting?)

The 0.1s head is the most interesting auxiliary and the reason to expect
anything: a separate study found that a model trained on the 0.1s label
predicts the 1s target SIGNIFICANTLY BETTER than a model trained on 1s
(+0.003358, CI [+0.001309,+0.005573]), because the short label is far less
noisy. A two-head net gets that for free instead of by rescaling.

ONLY THE MAIN HEAD IS SCORED, always on the untouched main target. Every target
is z-scored per head so one head's units cannot dominate the shared loss --
sd(ry_k_0p1s) and sd(ry_aux_vol30_1s) differ by orders of magnitude, and
without this the "multi-task" model is really single-task on whichever head has
the biggest numbers.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("MO_THREADS","10")))
F=os.environ.get("MO_FRAME","/var/lib/kalshi-study/frame_rich.parquet")
OUT="/var/lib/kalshi-study/MULTIOBJ.json"
MAIN=os.environ.get("MO_MAIN","ry_k_inc_0p1_to_1s")
CAPS=[("small",165,3),("large",320,5),("xlarge",640,5)]
LAMS=[0.1,1.0]
MAXROWS=900_000; BS=1024; SEEDS=(11,23)
sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
def have(*n): return [x for x in n if x in sch]
TIERS={1:[MAIN],
       2:[MAIN]+have("ry_k_0p1s"),
       4:[MAIN]+have("ry_k_0p1s","ry_k_5s","ry_k_30s"),
       8:[MAIN]+have("ry_k_0p1s","ry_k_5s","ry_k_30s",
                     "ry_aux_mid_1s","ry_aux_imb1_1s","ry_aux_spread_1s",
                     "ry_aux_vol30_1s")}
TIERS={k:[t for i,t in enumerate(v) if t not in v[:i]] for k,v in TIERS.items()}
TIERS={k:v for k,v in TIERS.items() if len(v)>=1}
print(f"{len(feats)} features; objective tiers:",flush=True)
for k,v in sorted(TIERS.items()): print(f"   {len(v)} heads: {v}",flush=True)
NEED=sorted({t for v in TIERS.values() for t in v})
Y={t:pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float64")
   for t in NEED}
mser=pd.read_parquet(F,columns=["market"])["market"]
mcode,munq=pd.factorize(mser,sort=True); mcode=mcode.astype("int32")
munq=np.asarray(munq,dtype=object); del mser; gc.collect()
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG",
                                  "SEP","OCT","NOV","DEC"])}
def tkey(tk):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",tk)
    if not g: return 0
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)
tkey_of=np.array([tkey(m) for m in munq],dtype="int64")
class Multi(nn.Module):
    def __init__(s,nin,w,L,nh,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,nh)
    def forward(s,x): return s.head(s.body(x))
def fit(X,Y2,W,M,Xv,Yv,Wv,Mv,seed,w,L,lam,wd=0.3,lr=1e-3,steps=6000,pat=12):
    nh=Y2.shape[1]
    torch.manual_seed(seed); m=Multi(X.shape[1],w,L,nh)
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=lr,total_steps=steps,pct_start=.15)
    hw=torch.tensor([1.0]+[lam]*(nh-1),dtype=torch.float32)
    A=torch.from_numpy(X);T=torch.from_numpy(Y2).float()
    Wt=torch.from_numpy(W).float().view(-1,1);Mk=torch.from_numpy(M).float()
    V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//20)
    while st<steps and bad<pat:
        pm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=pm[i:i+BS];o.zero_grad()
            pred=m(A[ix])
            # masked, head-weighted MSE: a missing label contributes nothing
            e=(pred-T[ix])**2*Mk[ix]*Wt[ix]*hw
            (e.sum()/(Mk[ix]*Wt[ix]*hw).sum().clamp_min(1e-9)).backward()
            o.step()
            if st<steps-1: sc.step()
            st+=1
            if st%ev==0 or st>=steps:
                m.eval()
                with torch.no_grad(): P=m(V).numpy()
                v=r2_zero(Yv[:,0],P[:,0],Wv)      # MAIN HEAD ONLY
                if v>best[0]: best=(v,[p.detach().clone() for p in m.parameters()]);bad=0
                else: bad+=1
                if bad>=pat or st>=steps: break
    if best[1] is not None:
        with torch.no_grad():
            for p,b in zip(m.parameters(),best[1]): p.copy_(b)
    m.eval(); return m,float(best[0])
def pr(m,A,ch=1_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy()[:,0])
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
    try: res=json.load(open(OUT))
    except Exception: res={}
def save(): json.dump(res,open(OUT,"w"),indent=1,default=str)
allm=np.argsort(tkey_of); nte=int(len(allm)*0.30)
c_te=allm[-nte:]; pool=allm[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
ok=np.isfinite(Y[MAIN])
i_te=g(c_te)[ok[g(c_te)]]; i_ho=g(c_ho)[ok[g(c_ho)]]; i_tr=g(c_tr)[ok[g(c_tr)]]
idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
def mk(ix):
    A=np.stack([pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False)[ix]
                for c in feats],1).astype("float32")
    np.nan_to_num(A,copy=False); return A
COL={}
def colv(c):
    if c not in COL:
        COL[c]=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False).astype("float32")
    return COL[c]
def mk2(ix): 
    A=np.stack([colv(c)[ix] for c in feats],1); 
    A=np.ascontiguousarray(A,dtype="float32"); np.nan_to_num(A,copy=False); return A
XA,Xh,Xte=mk2(idxA),mk2(i_ho),mk2(i_te)
S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
for A in (XA,Xh,Xte):
    np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ctm=mcode[i_te]
yte=Y[MAIN][i_te]
print(f"\nmain={MAIN} | train {len(idxA):,} rows, test {len(c_te)} mkts / "
      f"{len(i_te):,} rows",flush=True)
def heads(tl,ix):
    """z-scored per head (training stats), plus a finite-ness mask."""
    cols=[];msk=[]
    for t in tl:
        v=Y[t][ix].copy(); tr=Y[t][idxA]
        q=np.quantile(tr[np.isfinite(tr)],[.01,.99])
        m_=np.isfinite(v); v=np.where(m_,np.clip(v,q[0],q[1]),0.0)
        mm=float(np.nanmean(np.clip(tr,q[0],q[1]))); ss=float(np.nanstd(np.clip(tr,q[0],q[1]))) or 1.0
        cols.append((v-mm)/ss); msk.append(m_.astype("float32"))
    return np.stack(cols,1).astype("float32"), np.stack(msk,1).astype("float32")
P={}
for cname,w_,L in CAPS:
    for nobj in sorted(TIERS):
        tl=TIERS[nobj]
        lams=[0.0] if len(tl)==1 else LAMS
        for lam in lams:
            key=f"{cname}|n{len(tl)}|lam{lam}"
            if f"{MAIN}|{key}" in res and key in P: continue
            Ytr,Mtr=heads(tl,idxA); Yho,Mho=heads(tl,i_ho)
            m,hv=fit(XA,Ytr,wA,Mtr,Xh,Yho,wh,Mho,SEEDS[0],w_,L,lam)
            ps=[pr(m,Xte)]
            for s_ in SEEDS[1:]:
                m2,_=fit(XA,Ytr,wA,Mtr,Xh,Yho,wh,Mho,s_,w_,L,lam)
                ps.append(pr(m2,Xte)); del m2
            # un-z the main head back to the target's own units
            tr=Y[MAIN][idxA]; q=np.quantile(tr[np.isfinite(tr)],[.01,.99])
            mm=float(np.nanmean(np.clip(tr,q[0],q[1]))); ss=float(np.nanstd(np.clip(tr,q[0],q[1]))) or 1.0
            pm_=np.mean(ps,axis=0)*ss+mm
            P[key]=pm_
            r=float(r2_zero(yte,pm_,wte))
            npar=len(feats)*w_+L*w_*w_
            print(f"   {cname:7s} w={w_:<5} depth={L+1} {len(tl)} heads lam={lam:<4g} "
                  f"params={npar/1e6:5.2f}M  holdout {hv:+.6f}  test {r:+.6f}",flush=True)
            res[f"{MAIN}|{key}"]={"r2":r,"holdout":hv,"heads":len(tl),"lam":lam,
                                  "width":w_,"depth":L+1,"params":int(npar),
                                  "targets":tl}
            save(); del m,ps,Ytr,Mtr,Yho,Mho; gc.collect()
print("\n  PAIRED: more objectives vs 1 objective, AT THE SAME CAPACITY",flush=True)
gains={}
for cname,w_,L in CAPS:
    b=P.get(f"{cname}|n1|lam0.0")
    if b is None: continue
    for nobj in sorted(TIERS):
        if len(TIERS[nobj])==1: continue
        for lam in LAMS:
            k=f"{cname}|n{len(TIERS[nobj])}|lam{lam}"
            if k not in P: continue
            dm,dlo,dhi,sig=paired(ctm,yte,P[k],b,wte)
            print(f"    {cname:7s} {len(TIERS[nobj])} heads lam={lam:<4g} "
                  f"dR2={dm:+.6f} CI[{dlo:+.6f},{dhi:+.6f}] "
                  f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
            res[f"{MAIN}|gain|{k}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
            gains[(cname,len(TIERS[nobj]),lam)]=dm
save()
print("\n  THE INTERACTION -- does the objective gain grow with capacity?",flush=True)
for nobj in sorted({k[1] for k in gains}):
    for lam in LAMS:
        row=[(c,gains.get((c,nobj,lam))) for c,_,_ in CAPS]
        row=[(c,v) for c,v in row if v is not None]
        if len(row)<2: continue
        s_=" -> ".join(f"{c}:{v:+.6f}" for c,v in row)
        print(f"    {nobj} heads lam={lam:<4g}  {s_}",flush=True)
        res[f"{MAIN}|interaction|n{nobj}_lam{lam}"]={c:v for c,v in row}
save(); print(f"\nwrote {OUT}")
