"""A FAIR MODEL COMPARISON -- every model gets holdout hyperparameter selection.

"Ridge beats the MLP on the perp" is not believable, and checking how the
comparison was built shows why it should not be believed: the tuning was
asymmetric.

    ridge   alpha chosen from 8 values on the holdout        tuned
    lgbm    2 leaf/child/lr configs chosen on the holdout    tuned
    mlp     wd=3.0, lr=1e-3, fixed, every target             NOT tuned

And the targets are on very different scales: sd(y_p_1s)=1.586 against
sd(ry_k_1s)=0.212, a factor of 7.5. MSE scales as y^2, so the data-term
gradient is ~56x larger on the perp while weight decay stays a fixed absolute
penalty -- the net's effective regularisation silently changes with the target
and was never adapted. Ridge and LightGBM both re-select per target, so only
the net was handicapped. The MLP's own seed spread on y_p_1s gave it away:
-0.00108 to +0.00037, sd 0.00034 around a mean of +0.00016, which is a model
that is barely fitting rather than one discovering that the perp is linear.

Two corrections:
  1. THE TARGET IS Z-SCORED on training rows before fitting, and predictions
     are un-scaled before scoring. R^2 is computed against the untouched target
     in its own units, so nothing about the metric changes -- but wd and lr now
     mean the same thing for every target.
  2. THE MLP GETS A SWEEP: wd x lr x width, selected on the holdout, the same
     privilege ridge and lgbm already had.

If ridge still wins the perp after this, that is a result. As it stood it was a
statement about my hyperparameters.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import Ridge
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("FF_THREADS","10")))
F="/var/lib/kalshi-study/frame_big.parquet"
OUT="/var/lib/kalshi-study/FAIRFIGHT.json"
CELLS=[("BTC","y_p_1s"),("BTC","y_p_5s"),("XRP","y_p_1s"),("BTC","ry_k_1s")]
MAXROWS=900_000; BS=1024; SEEDS=(11,23)
# SWEPT ACROSS SIX ORDERS OF MAGNITUDE, BOTH DIRECTIONS.
#
# The economic prior is that a perp is close to efficient, so little signal is
# there and heavy shrinkage should win. The first sweep said the opposite: the
# best holdout sat at wd=0.3 and wd=30 collapsed the model to +0.00025, while
# ridge picked alpha=0.03 -- the FLOOR of its grid, i.e. the optimum was outside
# it. But that grid was truncated at both ends, and in patching it I extended
# only downward, which encodes my own reading rather than testing it. So the
# range now spans wd 1e-3..1e3 and alpha 1e-5..1e5 and the data decides.
#
# A grid whose WINNER sits at either edge is not a result, it is a grid that was
# too small -- the run reports when that happens instead of quietly returning
# the edge.
# TWO-STAGE SWEEP, HALF-DECADE STEPS.
#
# 10x steps (0.3, 3.0) cannot locate an optimum -- they can only tell you which
# decade it is in, and the first run's winner sat one step from a cell that
# collapsed the model entirely (wd=30 -> +0.00025). Half-decade spacing over six
# orders of magnitude is 13 points, which is affordable only if lr and width are
# not multiplied through it. So: sweep wd finely at a fixed lr/width, then
# refine lr and width around the winning wd.
WD_FINE=[1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1000.0]
REFINE_LR=[3e-4,1e-3,3e-3]
REFINE_W=[64,165,512]
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
class Net(nn.Module):
    def __init__(s,nin,w,L=3,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))
def fit(Xt,y,w,Xv,yv,wv,seed,wd,lr,width,steps=6000,pat=12):
    torch.manual_seed(seed); m=Net(Xt.shape[1],width)
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
    m.eval(); return m,float(best[0])
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
    try: res=json.load(open(OUT))
    except Exception: res={}
for coin,t in CELLS:
    if f"{coin}|{t}|mlp_tuned" in res: print(f"cached {coin}/{t}"); continue
    Y=pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float64")
    oc=np.flatnonzero(coin_of==coin); oc=oc[np.argsort(tkey_of[oc])]
    nte=int(len(oc)*.30); c_te=oc[-nte:]; pool=oc[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
    g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    ok=np.isfinite(Y)
    i_te=g(c_te)[ok[g(c_te)]]; i_ho=g(c_ho)[ok[g(c_ho)]]; i_tr=g(c_tr)[ok[g(c_tr)]]
    if min(len(i_te),len(i_tr))<5000: print(f"skip {coin}/{t}"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    def load(ix):
        A=np.empty((len(ix),len(feats)),dtype="float32")
        for j,c in enumerate(feats):
            col=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False)
            A[:,j]=col[ix].astype("float32",copy=False); del col
            if j%30==0: gc.collect()
        np.nan_to_num(A,copy=False); return A
    XA,Xh,Xte=load(idxA),load(i_ho),load(i_te); gc.collect()
    S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
    np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
    for A in (XA,Xh,Xte):
        np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ct=mcode[i_te]
    q=np.quantile(Y[i_tr],[.01,.99])
    yA_raw=Y[idxA].clip(q[0],q[1]); yh_raw=Y[i_ho]; yte=Y[i_te]
    # Z-SCORE the target on TRAINING rows only; predictions are un-scaled before
    # scoring, so R^2 is always against the untouched target in its own units.
    ym=float(yA_raw.mean()); ys=float(yA_raw.std()) or 1.0
    yA=(yA_raw-ym)/ys; yh=(yh_raw-ym)/ys
    print(f"\n{'='*86}\n=== {coin} / {t} | sd(target)={Y[i_tr].std():.4f} -> z-scored "
          f"(mu={ym:+.4g}, sd={ys:.4g}) | test {len(c_te)} mkts / {len(i_te):,} rows"
          f"\n{'='*86}",flush=True)
    P={}
    bb=(-9e9,None,None)
    for al in (1e-5,1e-4,1e-3,0.01,0.1,1,10,100,1000,10000,100000):
        rr=Ridge(alpha=al).fit(XA,yA,sample_weight=wA)
        v=r2_zero(yh,rr.predict(Xh),wh)
        if v>bb[0]: bb=(v,rr,al)
    _ALPHAS=(1e-5,1e-4,1e-3,0.01,0.1,1,10,100,1000,10000,100000)
    if bb[2] in (_ALPHAS[0],_ALPHAS[-1]):
        print(f"   !! ridge chose alpha={bb[2]:g}, an EDGE of the grid -- the "
              f"optimum is outside the range and this number is a floor/ceiling, "
              f"not a fit",flush=True)
        res[f"{coin}|{t}|ridge_edge_of_grid"]=True
    P["ridge"]=bb[1].predict(Xte)*ys+ym
    res[f"{coin}|{t}|ridge"]={"r2":float(r2_zero(yte,P["ridge"],wte)),"alpha":bb[2],
                              "target_sd":float(Y[i_tr].std())}
    json.dump(res,open(OUT,"w"),indent=1)
    print(f"   ridge      alpha={bb[2]:<6g} -> {r2_zero(yte,P['ridge'],wte):+.6f}",flush=True)
    bl=(-9e9,None,None)
    for cfg in (dict(num_leaves=7,min_child_samples=5000,learning_rate=0.02),
                dict(num_leaves=31,min_child_samples=2000,learning_rate=0.03),
                dict(num_leaves=127,min_child_samples=500,learning_rate=0.05)):
        gg=lgb.LGBMRegressor(n_estimators=2000,feature_fraction=.6,subsample=.7,
            subsample_freq=1,reg_lambda=10.0,verbose=-1,seed=11,n_jobs=10,**cfg)
        gg.fit(XA,yA,sample_weight=wA,eval_set=[(Xh,yh)],eval_sample_weight=[wh],
               callbacks=[lgb.early_stopping(60,verbose=False)])
        v=r2_zero(yh,gg.predict(Xh),wh)
        if v>bl[0]: bl=(v,gg,cfg)
    P["lgbm"]=bl[1].predict(Xte)*ys+ym
    res[f"{coin}|{t}|lgbm"]={"r2":float(r2_zero(yte,P["lgbm"],wte)),
                             "leaves":bl[2]["num_leaves"],
                             "trees":int(bl[1].best_iteration_ or 0)}
    json.dump(res,open(OUT,"w"),indent=1)
    print(f"   lgbm       leaves={bl[2]['num_leaves']:<4} trees={bl[1].best_iteration_:<5} "
          f"-> {r2_zero(yte,P['lgbm'],wte):+.6f}",flush=True)
    # the MLP finally gets the same holdout selection the others had
    def hold(wd,lr,width):
        return float(np.mean([fit(XA,yA,wA,Xh,yh,wh,s_,wd,lr,width)[1]
                              for s_ in SEEDS]))
    best=(-9e9,None)
    print("   STAGE 1 -- weight decay, half-decade steps (lr=1e-3, w=165):",flush=True)
    for wd in WD_FINE:
        hm=hold(wd,1e-3,165)
        print(f"      wd={wd:<7g} holdout {hm:+.6f}",flush=True)
        res[f"{coin}|{t}|sweep|wd{wd}_lr0.001_w165"]={"holdout":hm}
        # SAVE EVERY CONFIG. The previous run burned 7h14m of CPU and persisted
        # nothing, because the dump only happened once a whole cell finished and
        # a VM restart stopped it mid-sweep. The same mistake had already cost
        # incroute 3h36m; fixing it there and not here was the error.
        json.dump(res,open(OUT,"w"),indent=1)
        if hm>best[0]: best=(hm,(wd,1e-3,165))
    wd0=best[1][0]
    print(f"   STAGE 2 -- lr x width refinement around wd={wd0:g}:",flush=True)
    for lr in REFINE_LR:
        for width in REFINE_W:
            if (lr,width)==(1e-3,165): continue
            hm=hold(wd0,lr,width)
            print(f"      wd={wd0:<7g} lr={lr:<7g} w={width:<4} holdout {hm:+.6f}",flush=True)
            res[f"{coin}|{t}|sweep|wd{wd0}_lr{lr}_w{width}"]={"holdout":hm}
            json.dump(res,open(OUT,"w"),indent=1)
            if hm>best[0]: best=(hm,(wd0,lr,width))
    wd,lr,width=best[1]
    _WDS=sorted(WD_FINE)
    if wd in (_WDS[0],_WDS[-1]):
        print(f"   !! mlp chose wd={wd:g}, an EDGE of the swept range -- widen it",
              flush=True)
        res[f"{coin}|{t}|mlp_edge_of_grid"]=True
    ps=[pr(fit(XA,yA,wA,Xh,yh,wh,s_,wd,lr,width)[0],Xte) for s_ in (11,23,37)]
    P["mlp_tuned"]=np.mean(ps,axis=0)*ys+ym
    print(f"   -> mlp tuned wd={wd} lr={lr} w={width} "
          f"-> {r2_zero(yte,P['mlp_tuned'],wte):+.6f}",flush=True)
    # what the OLD fixed setting gave, for the direct comparison
    ps0=[pr(fit(XA,yA,wA,Xh,yh,wh,s_,3.0,1e-3,165)[0],Xte) for s_ in (11,23,37)]
    P["mlp_fixed_wd3"]=np.mean(ps0,axis=0)*ys+ym
    print(f"   -> mlp OLD fixed wd=3 lr=1e-3 w=165 "
          f"-> {r2_zero(yte,P['mlp_fixed_wd3'],wte):+.6f}",flush=True)
    for k,p in P.items():
        res[f"{coin}|{t}|{k}"]={"r2":float(r2_zero(yte,p,wte))}
    # IS THE PERP NUMBER JUST MOMENTUM? A ridge on the perp's own returns and
    # volatility alone. If that reproduces most of the full-feature R^2, the
    # "signal" is the underlying's autocorrelation rather than anything the
    # order book adds, which matters a lot for whether it is tradeable.
    mom=[i for i,c in enumerate(feats)
         if c.startswith(("p_ret_","p_relvol_","ret_","relvol_"))]
    if mom:
        bm2=(-9e9,None)
        for al in (1e-4,1e-2,1,100,10000):
            rr=Ridge(alpha=al).fit(XA[:,mom],yA,sample_weight=wA)
            v=r2_zero(yh,rr.predict(Xh[:,mom]),wh)
            if v>bm2[0]: bm2=(v,rr)
        P["ridge_momentum_only"]=bm2[1].predict(Xte[:,mom])*ys+ym
        r=float(r2_zero(yte,P["ridge_momentum_only"],wte))
        print(f"   ridge_momentum_only ({len(mom)} feats) -> {r:+.6f}",flush=True)
        res[f"{coin}|{t}|ridge_momentum_only"]={"r2":r,"nfeat":len(mom)}
        json.dump(res,open(OUT,"w"),indent=1)
    print("   PAIRED vs ridge:",flush=True)
    for k in [x for x in ("lgbm","mlp_tuned","mlp_fixed_wd3",
                                  "ridge_momentum_only") if x in P]:
        dm,dlo,dhi,sig=paired(ct,yte,P[k],P["ridge"],wte)
        print(f"     {k:14s} dR2={dm:+.6f} CI[{dlo:+.6f},{dhi:+.6f}] "
              f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
        res[f"{coin}|{t}|paired|{k}_vs_ridge"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    json.dump(res,open(OUT,"w"),indent=1)
    del XA,Xh,Xte,P; gc.collect()
json.dump(res,open(OUT,"w"),indent=1); print(f"\nwrote {OUT}")
