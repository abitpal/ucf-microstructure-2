"""SEED ENSEMBLES AND MODEL BLENDS, DONE PROPERLY.

Two things the main grid cannot answer, because it runs two seeds and averages
model families with equal weights.

1. HOW MANY SEEDS IS ENOUGH? 16 nets are trained per cell and the bag is scored
   at n = 1, 2, 4, 8, 16 using the first n, alongside the full distribution of
   single-seed scores. On the small frames seed noise was large enough that
   three-of-three negative seeds bagged to positive; on frame_big the completed
   grid cells show a mean bagging gain of only ~1 bip and no cell where seeds
   disagree on sign. This measures the curve instead of arguing from either
   anecdote.

2. DOES BLENDING FAMILIES HELP, OR DID EQUAL WEIGHTS JUST HANDICAP THE BEST
   MODEL? In every completed grid cell the equal-weight blend of
   ridge/lgbm/mlp scored BELOW the mlp alone -- averaging drags a strong model
   toward two weaker ones. Weights are therefore fitted here by non-negative
   least squares ON THE HOLDOUT MARKETS, never on test, and the equal-weight
   blend is kept as the comparison so the difference between "blending" and
   "blending badly" is visible.

Everything is scored on the same held-out markets per cell, and the headline
comparisons get a paired market bootstrap on the pooled R^2 difference.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("SE_THREADS","8")))

F="/var/lib/kalshi-study/frame_big.parquet"
OUT=os.environ.get("SE_OUT","/var/lib/kalshi-study/SEED_ENSEMBLE.json")
CELLS=[("BTC","ry_k_0p1s"),("BTC","ry_k_1s"),("BTC","ry_k_inc_0p1_to_1s"),
       ("XRP","ry_k_1s"),("BTC","y_p_1s")]
NSEEDS=16; CHECK=[1,2,4,8,16]
MAXROWS=1_200_000; BS=1024

sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
TGS=sorted({t for _,t in CELLS})
pf=pq.ParquetFile(F); NROW=pf.metadata.num_rows
mser=pd.read_parquet(F,columns=["market"])["market"]
mcode,munq=pd.factorize(mser,sort=True); mcode=mcode.astype("int32")
munq=np.asarray(munq,dtype=object); del mser; gc.collect()
coin_of=np.array([m.split("15M")[0].replace("KX","") for m in munq],dtype=object)
_MON={m:i+1 for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL",
                                  "AUG","SEP","OCT","NOV","DEC"])}
def tkey(tk):
    g=re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})-",tk)
    if not g: return 0
    yy,mon,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mon,0)*100+int(dd))*10000+int(hm)
tkey_of=np.array([tkey(m) for m in munq],dtype="int64")
Y={t:pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float32")
   for t in TGS}
X_all=np.empty((NROW,len(feats)),dtype="float32")
for j,c in enumerate(feats):
    col=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False)
    X_all[:,j]=col.astype("float32",copy=False); del col
    if j%25==0: gc.collect()
gc.collect()
print(f"loaded {NROW:,} x {len(feats)} ({X_all.nbytes/2**30:.1f} GiB)",flush=True)

class Net(nn.Module):
    def __init__(s,nin,w=165,L=3,pdrop=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(pdrop)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(pdrop)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))

def fit_mlp(X,y,w,Xv,yv,wv,seed,steps=6000,lr=1e-3,wd=3.0,patience=14):
    torch.manual_seed(seed); m=Net(X.shape[1])
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    scd=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=0.15)
    Xt=torch.from_numpy(X);yt=torch.from_numpy(y).float().view(-1,1)
    wt=torch.from_numpy(w).float().view(-1,1);Xvt=torch.from_numpy(Xv)
    n=len(Xt);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//25)
    while st<steps and bad<patience:
        perm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=perm[i:i+BS];opt.zero_grad()
            ((wt[ix]*(m(Xt[ix])-yt[ix])**2).sum()/wt[ix].sum()).backward()
            opt.step()
            if st<steps-1: scd.step()
            st+=1
            if st%ev==0 or st>=steps:
                m.eval()
                with torch.no_grad(): v=r2_zero(yv,m(Xvt).numpy().ravel(),wv)
                if v>best[0]: best=(v,[p.detach().clone() for p in m.parameters()]);bad=0
                else: bad+=1
                if bad>=patience or st>=steps: break
    if best[1] is not None:
        with torch.no_grad():
            for p,b in zip(m.parameters(),best[1]): p.copy_(b)
    m.eval(); return m

def mp(m,X,chunk=2_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),chunk): o.append(m(torch.from_numpy(X[i:i+chunk])).numpy().ravel())
    return np.concatenate(o)

def rw(codes):
    u,inv=np.unique(codes,return_inverse=True)
    n=np.bincount(inv,minlength=len(u)).astype("float64")
    return (1.0/n)[inv]

def boot_ci(codes,y,p,w,B=800,seed=0):
    u,inv=np.unique(codes,return_inverse=True)
    num=np.bincount(inv,weights=w*(y-p)**2,minlength=len(u))
    den=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed); pk=rng.integers(0,len(u),size=(B,len(u)))
    with np.errstate(divide="ignore",invalid="ignore"):
        v=1.0-num[pk].sum(1)/den[pk].sum(1)
    v=v[np.isfinite(v)]
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))

def paired(codes,y,pa,pb,w,B=2000,seed=0):
    u,inv=np.unique(codes,return_inverse=True)
    na=np.bincount(inv,weights=w*(y-pa)**2,minlength=len(u))
    nb=np.bincount(inv,weights=w*(y-pb)**2,minlength=len(u))
    den=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed); pk=rng.integers(0,len(u),size=(B,len(u)))
    D=den[pk].sum(1)
    with np.errstate(divide="ignore",invalid="ignore"):
        d=(nb[pk].sum(1)-na[pk].sum(1))/D
    d=d[np.isfinite(d)]
    lo,hi=np.percentile(d,2.5),np.percentile(d,97.5)
    return float(d.mean()),float(lo),float(hi),bool((lo>0)==(hi>0))

res={}
if os.path.exists(OUT):
    try: res=json.load(open(OUT)); print(f"resuming with {len(res)} keys",flush=True)
    except Exception: res={}
def save(): json.dump(res,open(OUT,"w"),indent=1,default=str)

for coin,t in CELLS:
    y=Y[t]
    oc=np.flatnonzero(coin_of==coin); oc=oc[np.argsort(tkey_of[oc])]
    nte=int(len(oc)*0.30); c_te=oc[-nte:]; pool=oc[:-nte]
    c_ho=pool[-120:]; c_tr=pool[:-120]
    f=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    i_te,i_ho,i_tr=f(c_te),f(c_ho),f(c_tr)
    i_te=i_te[np.isfinite(y[i_te])]; i_ho=i_ho[np.isfinite(y[i_ho])]
    i_tr=i_tr[np.isfinite(y[i_tr])]
    if min(len(i_te),len(i_tr))<5000: print(f"skip {coin}/{t}"); continue
    st=i_tr if len(i_tr)<=400_000 else np.random.default_rng(7).choice(i_tr,400_000,replace=False)
    Xs=X_all[st]; lo=np.quantile(Xs,.01,axis=0); hi=np.quantile(Xs,.99,axis=0)
    np.clip(Xs,lo,hi,out=Xs); mu=Xs.mean(0); sg=Xs.std(0); sg[sg==0]=1.0; del Xs
    def mk(ix):
        A=X_all[ix].copy(); np.clip(A,lo,hi,out=A); A-=mu; A/=sg
        np.nan_to_num(A,copy=False); return A
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    XA=mk(idxA); Xh=mk(i_ho); Xte=mk(i_te)
    ylo,yhi=np.quantile(y[i_tr],.01),np.quantile(y[i_tr],.99)
    yA=y[idxA].astype("float64").clip(ylo,yhi)
    yh=y[i_ho].astype("float64"); yte=y[i_te].astype("float64")
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te])
    ct=mcode[i_te]
    print(f"\n{'='*86}\n=== {coin} / {t} | train {len(XA):,} rows, "
          f"test {len(c_te)} mkts / {len(i_te):,} rows\n{'='*86}",flush=True)

    # ---- seed ensemble curve
    Pte=[];Pho=[];singles=[]
    for k in range(NSEEDS):
        m=fit_mlp(XA,yA,wA,Xh,yh,wh,seed=101+7*k)
        pte=mp(m,Xte); pho=mp(m,Xh); del m
        Pte.append(pte); Pho.append(pho)
        singles.append(float(r2_zero(yte,pte,wte)))
        if (k+1) in CHECK:
            bag=np.mean(Pte[:k+1],axis=0)
            r=float(r2_zero(yte,bag,wte)); ci=boot_ci(ct,yte,bag,wte)
            print(f"   mlp bag n={k+1:<3} {r:+.5f}  95% CI [{ci[0]:+.5f},{ci[1]:+.5f}]",flush=True)
            res[f"{coin}|{t}|mlp_bag_{k+1}"]={"r2":r,"ci":list(ci)}
            save()
    print(f"   single seeds: mean {np.mean(singles):+.5f}  sd {np.std(singles):.5f}  "
          f"min {min(singles):+.5f}  max {max(singles):+.5f}",flush=True)
    res[f"{coin}|{t}|mlp_singles"]=singles
    mlp_te=np.mean(Pte,axis=0); mlp_ho=np.mean(Pho,axis=0)

    # ---- the other families
    bb=(-9e9,None)
    for al in (0.3,1,3,10,30,100,300,1000):
        rr=Ridge(alpha=al).fit(XA,yA,sample_weight=wA)
        v=r2_zero(yh,rr.predict(Xh),wh)
        if v>bb[0]: bb=(v,rr)
    rid_te,rid_ho=bb[1].predict(Xte),bb[1].predict(Xh)
    best=(-9e9,None,None)
    for cfg in (dict(num_leaves=7,min_child_samples=5000,learning_rate=0.02),
                dict(num_leaves=31,min_child_samples=2000,learning_rate=0.03)):
        g=lgb.LGBMRegressor(n_estimators=1500,feature_fraction=.6,subsample=.7,
            subsample_freq=1,reg_lambda=10.0,verbose=-1,seed=11,n_jobs=8,**cfg)
        g.fit(XA,yA,sample_weight=wA,eval_set=[(Xh,yh)],eval_sample_weight=[wh],
              callbacks=[lgb.early_stopping(60,verbose=False)])
        v=r2_zero(yh,g.predict(Xh),wh)
        if v>best[0]: best=(v,g,cfg)
    g=best[1]; lgb_te,lgb_ho=g.predict(Xte),g.predict(Xh)
    print(f"   ridge {r2_zero(yte,rid_te,wte):+.5f}   "
          f"lgbm {r2_zero(yte,lgb_te,wte):+.5f} (leaves={best[2]['num_leaves']}, "
          f"trees={g.best_iteration_})   mlp_bag16 {r2_zero(yte,mlp_te,wte):+.5f}",flush=True)
    res[f"{coin}|{t}|ridge"]={"r2":float(r2_zero(yte,rid_te,wte))}
    res[f"{coin}|{t}|lgbm"]={"r2":float(r2_zero(yte,lgb_te,wte)),
                             "cfg":best[2],"trees":int(g.best_iteration_ or 0)}

    # ---- blends. THREE ARMS, BECAUSE UNCONSTRAINED NNLS CONFLATES TWO THINGS.
    #
    # Plain non-negative least squares put no constraint on the sum, and the
    # fitted weights came out summing to 1.13, 0.70, 1.07 and -- on the perp --
    # 3.90. A sum of 3.9 is not a blend: every model here is regularised and so
    # shrunk toward zero, and the regression was mostly undoing that shrinkage
    # by scaling the prediction up. Reading "ridge and lgbm carry the perp" off
    # that was wrong, because the gain was calibration, not combination.
    #
    #   blend_simplex  w >= 0 and sum(w) == 1  -- pure combination, no rescale
    #   blend_nnls     w >= 0, sum free        -- combination AND rescale
    #   scale_only     best single model x one fitted scalar
    #
    # scale_only is the diagnostic: if it captures most of what blend_nnls
    # gains, the gain was never about blending.
    Ate=np.vstack([rid_te,lgb_te,mlp_te]).T
    Aho=np.vstack([rid_ho,lgb_ho,mlp_ho]).T
    eq=Ate.mean(axis=1)
    sw=np.sqrt(wh)
    cf,_=nnls(Aho*sw[:,None], yh*sw)
    fit_b=Ate@cf
    # exact simplex search -- three weights, so a 0.01 grid is cheap and avoids
    # any optimiser pathology
    step=0.01; bestw=None; bestv=-9e9
    gr=np.arange(0.0,1.0+1e-9,step)
    for w1 in gr:
        for w2 in gr[:len(gr)-int(round(w1/step))]:
            w3=1.0-w1-w2
            if w3<-1e-9: continue
            pr=Aho@np.array([w1,w2,w3])
            v=r2_zero(yh,pr,wh)
            if v>bestv: bestv=v; bestw=np.array([w1,w2,max(w3,0.0)])
    simp=Ate@bestw
    # single best model on the holdout, times one fitted scalar
    hos={"ridge":rid_ho,"lgbm":lgb_ho,"mlp":mlp_ho}
    tes={"ridge":rid_te,"lgbm":lgb_te,"mlp":mlp_te}
    bk=max(hos,key=lambda k:r2_zero(yh,hos[k],wh))
    a=float(np.sum(wh*hos[bk]*yh)/max(np.sum(wh*hos[bk]**2),1e-18))
    scal=tes[bk]*a
    print(f"   blend weights: simplex(sum=1) ridge/lgbm/mlp = {bestw.round(3).tolist()}"
          f" | nnls(sum free) = {cf.round(3).tolist()} (sum {cf.sum():.3f})"
          f" | scale_only on {bk}, gain {a:.3f}",flush=True)
    res[f"{coin}|{t}|blend_weights"]={"simplex":bestw.tolist(),"nnls":cf.tolist(),
                                      "nnls_sum":float(cf.sum()),
                                      "scale_model":bk,"scale_gain":a}
    for nm,p in (("blend_equal",eq),("blend_simplex",simp),
                 ("blend_nnls",fit_b),("scale_only",scal)):
        r=float(r2_zero(yte,p,wte)); ci=boot_ci(ct,yte,p,wte)
        print(f"   {nm:13s} {r:+.5f}  95% CI [{ci[0]:+.5f},{ci[1]:+.5f}]",flush=True)
        res[f"{coin}|{t}|{nm}"]={"r2":r,"ci":list(ci)}
    print("   PAIRED vs mlp_bag16:",flush=True)
    for nm,p in (("blend_equal",eq),("blend_simplex",simp),("blend_nnls",fit_b),
                 ("scale_only",scal),("ridge",rid_te),("lgbm",lgb_te)):
        dm,dlo,dhi,sig=paired(ct,yte,p,mlp_te,wte)
        print(f"     {nm:13s} dR2={dm:+.5f} CI[{dlo:+.5f},{dhi:+.5f}] "
              f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
        res[f"{coin}|{t}|paired|{nm}-mlpbag16"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    save()
    del XA,Xh,Xte,Pte,Pho; gc.collect()
save(); print(f"\nwrote {OUT}")
