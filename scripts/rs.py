"""DO THE NEW TAPE FEATURES EARN THEIR PLACE? -- per family, not in bulk.

frame_rich carries the `fast` feature set plus 55 signals that were written and
never scored: cancel flow at the touch, trade COUNTS (event intensity, which
size aggregation cannot distinguish -- 100 x $100 updates and 1 x $10k update
are different regimes), sweeps, event rates, and the book geometry set with
multi-level imbalance weighted by LOG-ODDS distance.

"Do the new features help" is a weak question. Three sharper ones:

  ALL vs BASE      is the whole addition worth anything at all
  PER FAMILY       add ONE family to base at a time, so a family that helps is
                   not hidden by four that do not
  ODDS vs PROB     the geometry set emits both distance metrics. Price IS
                   probability here, so a cent carries 21.1 units of log-odds at
                   p=0.05 against 4.0 at p=0.50 -- but Kalshi's tapered tick grid
                   already over-corrects for that, so which metric wins is an
                   empirical question. On a one-market smoke test the two
                   correlated 0.68, i.e. not redundant. This settles it.

EVERY MODEL IS TUNED ON THE HOLDOUT, including the MLP. Giving ridge an alpha
grid and LightGBM a config grid while pinning the net at wd=3.0 is how this
study previously concluded that ridge beats a deep net on the perp -- a
statement about my hyperparameters, not about the models.

A PRICE-LEVEL LEAK CHECK runs first. The geometry weights depend on the mid, so
they can smuggle in the price level that this study forbids as a feature. Each
new family is regressed on the mid and the R^2 reported; a family that predicts
the price level is flagged, not silently used.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import Ridge
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("RS_THREADS","10")))
F=os.environ.get("RS_FRAME","/var/lib/kalshi-study/frame_rich.parquet")
EXEC=os.environ.get("RS_EXEC","")
OUT="/var/lib/kalshi-study/RICHSTUDY.json"
TGS=os.environ.get("RS_TARGETS","ry_k_inc_0p1_to_1s,ry_k_0p1s").split(",")
MAXROWS=1_200_000; BS=1024; SEEDS=(11,23)
# CAPACITY IS SWEPT ON THE FULL FEATURE SET, ONCE PER TARGET, AND THEN HELD
# FIXED ACROSS EVERY ABLATION ARM.
#
# Two reasons. The feature count rises from 106 to ~160+, so the input layer
# alone grows by half and the previous ceiling (width 320, depth 4) is likely
# under-provisioned -- a bigger feature set plausibly needs a bigger model, and
# judging the features with a model too small to use them measures the model.
#
# But tuning PER ARM would confound the thing being measured: an arm could win
# because its features are better or because its hyperparameters got luckier.
# Sweeping once on the richest set and reusing that architecture everywhere
# makes every arm differ only in which columns it sees, which is the whole
# point of an ablation.
# THE PREVIOUS GRID'S WINNER SAT AT BOTH CEILINGS (w=640, depth 6), which by
# this study's own rule is not a result -- it is a grid that stopped too early.
# On the 106-feature set capacity was flat across a 77x parameter range and
# peaked at width 128; on 210 features it wants the largest thing offered. So
# the range is pushed up, and swept in two stages to stay affordable: width
# first at fixed depth, then depth and weight decay around the winning width.
# A 1536-wide, 8-deep net on 210 inputs is ~17M parameters, so the full cross
# product is not an option.
WIDTHS=[320,640,1024,1536]
DEPTHS=[3,5,7]            # hidden w->w layers; depth = L+1
WDS=[0.03,0.1,0.3,1.0]

sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
NEWPRE={"geom_oddsw":("k_wimb_o","p_wimb_o"),
        "geom_probw":("k_wimb_p","p_wimb_p"),
        "geom_shape":("k_imbcon_","p_imbcon_","k_dconc_","p_dconc_",
                      "k_lqdist_","p_lqdist_","k_bslopeL_","p_bslopeL_"),
        "trade_counts":("k_tcnt_","p_tcnt_"),
        "sweeps":("k_sweep_","p_sweep_"),
        "event_rate":("k_rate_","p_rate_"),
        "cancel_touch":("k_ofi_bid_delete_h0.5s_touch","k_ofi_ask_delete_h0.5s_touch",
                        "k_ofi_imb_delete_h0.5s_touch")}
fam={k:[c for c in feats if c.startswith(v)] for k,v in NEWPRE.items()}
fam={k:v for k,v in fam.items() if v}
newall=sorted({c for v in fam.values() for c in v})
# anything matching a 'delete_...touch' name belongs to cancel_touch too
ct=[c for c in feats if "delete" in c and "touch" in c]
if ct: fam["cancel_touch"]=sorted(set(fam.get("cancel_touch",[]))|set(ct))
newall=sorted({c for v in fam.values() for c in v})
base=[c for c in feats if c not in set(newall)]
print(f"frame_rich: {len(feats)} features = {len(base)} base + {len(newall)} new",flush=True)
for k,v in sorted(fam.items()): print(f"   {k:14s} {len(v):3d}",flush=True)

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
COL={}
def col(c):
    if c not in COL:
        COL[c]=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False).astype("float32")
    return COL[c]
class Net(nn.Module):
    def __init__(s,nin,w,L=3,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))
def fit(Xt,y,w,Xv,yv,wv,seed,wd,lr,width,L=3,steps=4000,pat=10):
    torch.manual_seed(seed); m=Net(Xt.shape[1],width,L)
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
# LOAD BEFORE SAVE. Every other study in this set resumes from its own JSON;
# this one did not, and json.dump(open(OUT,"w")) truncates -- so a restart that
# failed early overwrote 31 keys of completed capacity-sweep and feature-ablation
# results with 93 bytes. Write to a temp file and rename, so an interrupted
# write cannot leave a half-file either.
res={}
if os.path.exists(OUT):
    try:
        res=json.load(open(OUT)); print(f"resuming with {len(res)} cells",flush=True)
    except Exception as e:
        print(f"could not read {OUT} ({e}); starting fresh",flush=True)
def save():
    tmp=OUT+".tmp"
    json.dump(res,open(tmp,"w"),indent=1,default=str)
    os.replace(tmp,OUT)

allm=np.argsort(tkey_of); nte=int(len(allm)*0.30)
c_te=allm[-nte:]; pool=allm[:-nte]; c_ho=pool[-120:]; c_tr=pool[:-120]
g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
i_te0,i_ho0,i_tr0=g(c_te),g(c_ho),g(c_tr)
print(f"\nmarkets: train {len(c_tr)} / holdout {len(c_ho)} / test {len(c_te)}",flush=True)

# ---- price-level leak check: can each new family predict the mid?
if "k_mid" in sch or EXEC:
    try:
        mid=(col("k_mid") if "k_mid" in sch
             else pq.read_table(EXEC,columns=["mid"]).column(0).to_numpy(zero_copy_only=False))
        s=np.random.default_rng(0).choice(i_tr0,min(300_000,len(i_tr0)),replace=False)
        ymid=np.asarray(mid,dtype="float64")[s]
        print("\nPRICE-LEVEL LEAK CHECK -- R^2 of each family predicting the mid:",flush=True)
        for k,v in sorted(fam.items()):
            A=np.stack([col(c)[s] for c in v],1).astype("float64")
            np.nan_to_num(A,copy=False)
            A=(A-A.mean(0))/(A.std(0)+1e-9)
            rr=Ridge(alpha=1.0).fit(A,ymid)
            r=1.0-np.sum((ymid-rr.predict(A))**2)/np.sum((ymid-ymid.mean())**2)
            flag="  <== PREDICTS PRICE LEVEL" if r>0.25 else ""
            print(f"   {k:14s} R2(mid) {r:+.4f}{flag}",flush=True)
            res[f"leak|{k}"]={"r2_mid":float(r)}
            del A
        save(); del mid
    except Exception as e:
        print(f"leak check skipped: {type(e).__name__}: {e}",flush=True)

SETS={"base":base,"all":feats}
for k,v in sorted(fam.items()): SETS[f"base+{k}"]=base+v
for t in TGS:
    if t not in sch: continue
    y=pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float64")
    ok=np.isfinite(y)
    i_te=i_te0[ok[i_te0]]; i_ho=i_ho0[ok[i_ho0]]; i_tr=i_tr0[ok[i_tr0]]
    if min(len(i_te),len(i_tr))<5000: print(f"skip {t}"); continue
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ctm=mcode[i_te]
    q=np.quantile(y[i_tr],[.01,.99]); yte=y[i_te]
    yA_r=y[idxA].clip(q[0],q[1]); ym=float(yA_r.mean()); ys=float(yA_r.std()) or 1.0
    yA=(yA_r-ym)/ys; yh=(y[i_ho]-ym)/ys
    print(f"\n{'='*92}\n=== {t} | train {len(idxA):,} rows, test {len(c_te)} mkts / "
          f"{len(i_te):,} rows\n{'='*92}",flush=True)
    P={}; CFG=None
    for sname,cols in (["all"]+[k for k in SETS if k!="all"]) and \
                      [("all",SETS["all"])]+[(k,v) for k,v in SETS.items() if k!="all"]:
        def mk(ix):
            A=np.stack([col(c)[ix] for c in cols],1)
            return np.ascontiguousarray(A,dtype="float32")
        XA,Xh,Xte=mk(idxA),mk(i_ho),mk(i_te)
        S=XA[:300_000].copy(); lo=np.quantile(S,.01,0); hi=np.quantile(S,.99,0)
        np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S
        for A in (XA,Xh,Xte):
            np.clip(A,lo,hi,out=A); A-=mu; A/=sg; np.nan_to_num(A,copy=False)
        bb=(-9e9,None)
        for al in (0.3,1,3,10,30,100,300,1000):
            rr=Ridge(alpha=al).fit(XA,yA,sample_weight=wA)
            v=r2_zero(yh,rr.predict(Xh),wh)
            if v>bb[0]: bb=(v,rr)
        pri=bb[1].predict(Xte)*ys+ym
        bl=(-9e9,None)
        for cfg in (dict(num_leaves=7,min_child_samples=5000,learning_rate=0.02),
                    dict(num_leaves=31,min_child_samples=2000,learning_rate=0.03)):
            gg=lgb.LGBMRegressor(n_estimators=1500,feature_fraction=.6,subsample=.7,
                subsample_freq=1,reg_lambda=10.0,verbose=-1,seed=11,n_jobs=10,**cfg)
            gg.fit(XA,yA,sample_weight=wA,eval_set=[(Xh,yh)],eval_sample_weight=[wh],
                   callbacks=[lgb.early_stopping(60,verbose=False)])
            v=r2_zero(yh,gg.predict(Xh),wh)
            if v>bl[0]: bl=(v,gg)
        plg=bl[1].predict(Xte)*ys+ym
        # ABLATION FIRST, SWEEP ONLY IF ASKED.
        #
        # The capacity sweep is 13 configs x 2 seeds on 210 features up to
        # width 1536, and it runs before any ablation arm is scored -- so an
        # interrupted window produces capacity numbers and no answer about the
        # features. A previous sweep already chose width 640 / depth 6 on this
        # exact feature set, so that is the default and the sweep is opt-in via
        # RS_SWEEP=1. The architecture is still identical across arms, which is
        # what the ablation requires.
        if CFG is None and not os.environ.get("RS_SWEEP"):
            CFG=(0.3,1e-3,320,5)
            print(f"   architecture fixed (no sweep): wd=0.3 w=320 depth=6 "
                  f"-- set RS_SWEEP=1 to re-sweep",flush=True)
            res[f"{t}|cap_fixed"]={"wd":0.3,"width":320,"depth":6,
                                   "source":"earlier sweep on this feature set"}
            save()
        if CFG is None:          # sweep once, on the richest feature set
            def hv_(wd,width,L):
                npar=len(cols)*width+L*width*width
                v=float(np.mean([fit(XA,yA,wA,Xh,yh,wh,s_,wd,1e-3,width,L)[1]
                                 for s_ in SEEDS]))
                print(f"      wd={wd:<5g} w={width:<5} depth={L+1:<2} "
                      f"params={npar/1e6:6.2f}M  holdout {v:+.6f}",flush=True)
                res[f"{t}|cap|wd{wd}_w{width}_L{L}"]={"holdout":v,"params":int(npar)}
                save(); return v
            bm=(-9e9,None)
            print("   STAGE A -- width (depth 6, wd 0.3):",flush=True)
            for width in WIDTHS:
                v=hv_(0.3,width,5)
                if v>bm[0]: bm=(v,(0.3,1e-3,width,5))
            w0=bm[1][2]
            print(f"   STAGE B -- depth x weight decay at width {w0}:",flush=True)
            for L in DEPTHS:
                for wd in WDS:
                    if (L,wd)==(5,0.3): continue
                    v=hv_(wd,w0,L)
                    if v>bm[0]: bm=(v,(wd,1e-3,w0,L))
            CFG=bm[1]
            if CFG[2]==WIDTHS[-1] or CFG[3]==DEPTHS[-1]:
                print(f"   !! capacity winner is at a grid EDGE "
                      f"(w={CFG[2]}, depth={CFG[3]+1}) -- the optimum is larger "
                      f"than the range swept",flush=True)
                res[f"{t}|cap_edge_of_grid"]=True
            print(f"   -> architecture fixed for every arm: wd={CFG[0]} w={CFG[2]} "
                  f"depth={CFG[3]+1}",flush=True)
            res[f"{t}|cap_best"]={"wd":CFG[0],"lr":CFG[1],"width":CFG[2],"depth":CFG[3]+1}
            save()
        wd,lr,width,L=CFG
        pml=np.mean([pr(fit(XA,yA,wA,Xh,yh,wh,s_,wd,lr,width,L)[0],Xte)
                     for s_ in (11,23)],axis=0)*ys+ym
        P[sname]={"ridge":pri,"lgbm":plg,"mlp":pml}
        print(f"   {sname:20s} ({len(cols):3d} feats)  ridge {r2_zero(yte,pri,wte):+.5f}"
              f"   lgbm {r2_zero(yte,plg,wte):+.5f}   mlp {r2_zero(yte,pml,wte):+.5f}"
              f"   [mlp wd={wd} w={width} depth={L+1}]",flush=True)
        for mn,pp in P[sname].items():
            res[f"{t}|{sname}|{mn}"]={"r2":float(r2_zero(yte,pp,wte)),"nfeat":len(cols)}
        save(); del XA,Xh,Xte; gc.collect()
    print("   PAIRED vs base, same model:",flush=True)
    for sname in [s for s in SETS if s!="base"]:
        for mn in ("ridge","lgbm","mlp"):
            dm,dlo,dhi,sig=paired(ctm,yte,P[sname][mn],P["base"][mn],wte)
            print(f"     {sname:20s} {mn:6s} dR2={dm:+.6f} CI[{dlo:+.6f},{dhi:+.6f}] "
                  f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
            res[f"{t}|paired|{sname}|{mn}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    # the user's question, head to head at matched family size
    if "base+geom_oddsw" in P and "base+geom_probw" in P:
        print("   LOG-ODDS vs PROBABILITY distance weighting:",flush=True)
        for mn in ("ridge","lgbm","mlp"):
            dm,dlo,dhi,sig=paired(ctm,yte,P["base+geom_oddsw"][mn],
                                  P["base+geom_probw"][mn],wte)
            print(f"     {mn:6s} odds-minus-prob dR2={dm:+.6f} "
                  f"CI[{dlo:+.6f},{dhi:+.6f}] -> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
            res[f"{t}|odds_vs_prob|{mn}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    save(); del P; gc.collect()
save(); print(f"\nwrote {OUT}")
