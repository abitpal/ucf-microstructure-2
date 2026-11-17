"""FEATURE EXPANSION AND CAUSAL VOL-NORMALISED TARGETS, ON frame_big.

Two things, both derivable from what is already on disk -- no tape replay.

1. CAUSAL VOL-NORMALISED TARGETS. An earlier dashboard established this and I
   wrongly reported it as untested: dividing the target by a TRAILING vol
   estimate took ridge from +0.01416 to +0.02523 and turned the worst fold
   positive. The leaking variant (/per_market_sigma) read +0.05422 because its
   sigma spanned the test window; it is not used here. The divisor is
   relvol_30s AT TIME t -- a feature, already in the frame, known at decision
   time -- so there is no lookahead. Scoring is done BOTH ways: on the
   normalised target (the fitting objective) and back-transformed to the raw
   target, because a gain that exists only in normalised units is a gain
   against an easier question, not a better model.

2. DERIVED FEATURES. The 106 columns carry each flow family at several
   half-lives but only ever as levels. What is missing is the SHAPE across
   half-lives, which is what says whether flow is accelerating:

     acceleration   fast - slow                  (e.g. ofi_imb_0.5s - ofi_imb_5s)
     normalised     fast / (|slow| + eps)
     regime scaling flow / trailing vol          (same logic as the target)
     interactions   imbalance x spread_ticks, imbalance x relvol
     log magnitude  sign(x)*log1p(|x|)           for the heavy-tailed flow columns
     level contrast d1 EMA imbalance - d3 EMA imbalance

   All are causal functions of existing columns. This is the cheap half of the
   feature question; queue age, book slope, multi-level depth, event counts and
   250ms horizons need the replay and are not here.

Scored on frame_big with paired market bootstraps, base feature set vs expanded.
"""
import sys, os, json, gc, re; sys.path.insert(0,"/opt/kalshi")
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
import torch.nn as nn, lightgbm as lgb
from sklearn.linear_model import Ridge
from tools.panel_model import r2_zero
torch.set_num_threads(int(os.environ.get("FE_THREADS","8")))

F="/var/lib/kalshi-study/frame_big.parquet"
OUT=os.environ.get("FE_OUT","/var/lib/kalshi-study/FEATEXP.json")
CELLS=[("BTC","ry_k_1s"),("BTC","ry_k_inc_0p1_to_1s"),("XRP","ry_k_1s"),
       ("BTC","ry_k_0p1s")]
MAXROWS=1_200_000; BS=1024; SEEDS=(11,23,37)
EPS=1e-6

sch=[f.name for f in pq.ParquetFile(F).schema_arrow]
base_feats=[c for c in sch if not c.startswith(("y_","ry_","market","_day","coin"))]
TGS=sorted({t for _,t in CELLS})
VOLCOL="relvol_30s"
assert VOLCOL in base_feats, "no trailing vol column to normalise by"
# TIME TO EXPIRY. Built and alignment-checked separately (tte_big.npy): the
# companion files' ts matched the frame row-for-row with zero market
# mismatches, and the expiry convention was verified by requiring ~all rows to
# land inside the 15-minute slot -- which rejected the first reading, where the
# ticker's HHMM was assumed to be the open rather than the close.
# A clock feature, not a price feature, so it is allowed here.
TTE=np.load("/var/lib/kalshi-study/tte_big.npy").astype("float32")

def horizon_of(c):
    m=re.search(r"_h([0-9.]+)s$",c)
    return float(m.group(1)) if m else None
def stem_of(c):
    return re.sub(r"_h[0-9.]+s$","",c)

# families that carry the same quantity at several half-lives
fam={}
for c in base_feats:
    h=horizon_of(c)
    if h is not None: fam.setdefault(stem_of(c),[]).append((h,c))
fam={k:sorted(v) for k,v in fam.items() if len(v)>1}
RET=[c for c in base_feats if re.fullmatch(r"ret_[0-9.]+s",c)]
PRET=[c for c in base_feats if re.fullmatch(r"p_ret_[0-9.]+s",c)]
def rsec(c): return float(re.search(r"_([0-9.]+)s$",c).group(1))
RET=sorted(RET,key=rsec); PRET=sorted(PRET,key=rsec)
IMB=[c for c in base_feats if "imb" in c or "imbalance" in c]

def derive(D):
    """D: dict name->float32 array of the base columns. Returns new columns."""
    new={}
    vol=np.maximum(np.abs(D[VOLCOL]),1e-5)
    # --- 1. acceleration across half-lives, within each family
    for st,hs in fam.items():
        for i in range(len(hs)-1):
            (ha,ca),(hb,cb)=hs[i],hs[i+1]
            a,b=D[ca],D[cb]
            new[f"acc_{st}_{ha:g}m{hb:g}"]=a-b
            new[f"accn_{st}_{ha:g}o{hb:g}"]=a/(np.abs(b)+EPS)
        if len(hs)>2:            # plus one long-span contrast: fastest vs slowest
            (ha,ca),(hb,cb)=hs[0],hs[-1]
            new[f"acc_{st}_{ha:g}m{hb:g}"]=D[ca]-D[cb]
    # --- 2. momentum term structure (adjacent horizons only; all pairs explodes)
    for seq,tag in ((RET,"ret"),(PRET,"pret")):
        for i in range(len(seq)-1):
            new[f"macc_{tag}_{rsec(seq[i]):g}m{rsec(seq[i+1]):g}"]=D[seq[i]]-D[seq[i+1]]
    # --- 3. flow and momentum scaled by the trailing vol regime
    for c in IMB[:20]:
        new[f"vs_{c}"]=D[c]/vol
    for c in RET:
        new[f"vs_{c}"]=D[c]/vol
    # --- 4. interactions: the same imbalance means different things in a wide
    #        book or a volatile regime
    sp=D.get("spread_ticks"); rv=D.get("relvol_1s")
    for c in IMB[:12]:
        if sp is not None: new[f"x_{c}_spt"]=D[c]*sp
        if rv is not None: new[f"x_{c}_rv1"]=D[c]*rv
    # --- 5. signed log magnitude for the heavy-tailed flow columns
    for c in [c for c in base_feats if c.startswith(("k_ofi_imb","k_tofi_imb","p_ofi_imb","p_tofi_imb"))][:24]:
        v=D[c]; new[f"lg_{c}"]=np.sign(v)*np.log1p(np.abs(v))
    # --- 6. time to expiry, and what it conditions.
    #        A 15-minute binary is a different instrument at 12 minutes and at
    #        20 seconds: variance-to-go shrinks with the clock, so the same
    #        imbalance or flow reading means different things. 1/sqrt(tte) is
    #        the natural scaling for a diffusive remaining-variance.
    tte=D.get("__tte__")
    if tte is not None:
        tc=np.clip(tte,1.0,None)
        new["tte_s"]=tte
        new["tte_log"]=np.log1p(np.clip(tte,0.0,None))
        new["tte_invsqrt"]=1.0/np.sqrt(tc)
        for c in (IMB[:10]+RET[:4]):
            new[f"tt_{c}"]=D[c]/np.sqrt(tc)
        for c in ("spread_ticks","relvol_1s","relvol_30s"):
            if c in D: new[f"ttx_{c}"]=D[c]*new["tte_invsqrt"]
    # --- 7. depth-level contrast of the EMA imbalances (touch vs behind it)
    for pre in ("k","p"):
        for h in ("0.5","2"):
            a,b=f"{pre}_ema_imb_d1_h{h}s",f"{pre}_ema_imb_d3_h{h}s"
            if a in D and b in D: new[f"emacon_{pre}_{h}"]=D[a]-D[b]
    return new

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
    yy,mo,dd,hm=g.groups()
    return ((2000+int(yy))*10000+_MON.get(mo,0)*100+int(dd))*10000+int(hm)
tkey_of=np.array([tkey(m) for m in munq],dtype="int64")
Y={t:pq.read_table(F,columns=[t]).column(0).to_numpy(zero_copy_only=False).astype("float32")
   for t in TGS}
D={"__tte__":TTE}
for c in base_feats:
    D[c]=pq.read_table(F,columns=[c]).column(0).to_numpy(zero_copy_only=False).astype("float32")
    gc.collect()
print(f"base {len(base_feats)} columns loaded",flush=True)
# Derived names come from one cheap probe on 1000 rows, so the column list is
# known without materialising anything full-length.
_probe={c:D[c][:1000] for c in list(base_feats)+["__tte__"]}
DERIVED_NAMES=list(derive(_probe).keys()); del _probe
names_base=list(base_feats); names_all=names_base+DERIVED_NAMES
print(f"derived {len(DERIVED_NAMES)} new columns -> {len(names_all)} total; "
      f"a full-length matrix would be "
      f"{NROW*len(names_all)*4/2**30:.0f} GiB, so it is built PER CELL on that "
      f"cell's rows only",flush=True)
VOL=np.maximum(np.abs(D[VOLCOL]),1e-5).astype("float64")
def build(ix,expanded):
    """Columns for the given rows; derived features computed on that slice."""
    sub={c:D[c][ix] for c in list(base_feats)+["__tte__"]}
    cols=list(base_feats)
    blocks=[np.stack([sub[c] for c in cols],axis=1)]
    if expanded:
        nw=derive(sub)
        blocks.append(np.stack([nw[k] for k in DERIVED_NAMES],axis=1))
        del nw
    A=np.concatenate(blocks,axis=1).astype("float32",copy=False)
    del sub,blocks
    np.nan_to_num(A,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    return A
def thin(ix,codes,cap):
    """Keep every k-th row within each market so coverage is unchanged."""
    if len(ix)<=cap: return ix
    k=int(np.ceil(len(ix)/cap))
    out=[]
    for c in np.unique(codes[ix]):
        sel=ix[codes[ix]==c]; out.append(sel[::k])
    return np.sort(np.concatenate(out))

class Net(nn.Module):
    def __init__(s,nin,w=165,L=3,p=0.2):
        super().__init__()
        q=[nn.Linear(nin,w),nn.ReLU(),nn.Dropout(p)]
        for _ in range(L): q+=[nn.Linear(w,w),nn.ReLU(),nn.Dropout(p)]
        s.body=nn.Sequential(*q); s.head=nn.Linear(w,1)
    def forward(s,x): return s.head(s.body(x))

def fit_mlp(Xt_,y,w,Xv,yv,wv,seed,steps=6000,lr=1e-3,wd=3.0,pat=14):
    torch.manual_seed(seed); m=Net(Xt_.shape[1])
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=0.15)
    A=torch.from_numpy(Xt_);yy=torch.from_numpy(y).float().view(-1,1)
    ww=torch.from_numpy(w).float().view(-1,1);V=torch.from_numpy(Xv)
    n=len(A);best=(-9e9,None);bad=0;st=0;ev=max(1,steps//25)
    while st<steps and bad<pat:
        perm=torch.randperm(n)
        for i in range(0,n-BS+1,BS):
            m.train();ix=perm[i:i+BS];opt.zero_grad()
            ((ww[ix]*(m(A[ix])-yy[ix])**2).sum()/ww[ix].sum()).backward()
            opt.step()
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
def mp(m,A,ch=2_000_000):
    o=[]
    with torch.no_grad():
        for i in range(0,len(A),ch): o.append(m(torch.from_numpy(A[i:i+ch])).numpy().ravel())
    return np.concatenate(o)
def rw(c):
    u,inv=np.unique(c,return_inverse=True)
    n=np.bincount(inv,minlength=len(u)).astype("float64"); return (1.0/n)[inv]
def paired(c,y,pa,pb,w,B=2000,seed=0):
    u,inv=np.unique(c,return_inverse=True)
    na=np.bincount(inv,weights=w*(y-pa)**2,minlength=len(u))
    nb=np.bincount(inv,weights=w*(y-pb)**2,minlength=len(u))
    de=np.bincount(inv,weights=w*y*y,minlength=len(u))
    rng=np.random.default_rng(seed); pk=rng.integers(0,len(u),size=(B,len(u)))
    with np.errstate(divide="ignore",invalid="ignore"):
        d=(nb[pk].sum(1)-na[pk].sum(1))/de[pk].sum(1)
    d=d[np.isfinite(d)]; lo,hi=np.percentile(d,2.5),np.percentile(d,97.5)
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
    g=lambda cs: np.flatnonzero(np.isin(mcode,cs))
    i_te,i_ho,i_tr=g(c_te),g(c_ho),g(c_tr)
    ok=lambda ix: ix[np.isfinite(y[ix])&np.isfinite(VOL[ix])]
    i_te,i_ho,i_tr=ok(i_te),ok(i_ho),ok(i_tr)
    if min(len(i_te),len(i_tr))<5000: print(f"skip {coin}/{t}"); continue
    n0=len(i_te); i_te=thin(i_te,mcode,2_000_000)
    if len(i_te)!=n0:
        print(f"   test rows thinned {n0:,} -> {len(i_te):,} "
              f"(every k-th within each market; all {len(c_te)} markets kept)",flush=True)
    idxA=i_tr if len(i_tr)<=MAXROWS else np.random.default_rng(0).choice(i_tr,MAXROWS,replace=False)
    wA=rw(mcode[idxA]); wA/=wA.sum(); wh=rw(mcode[i_ho]); wte=rw(mcode[i_te]); ct=mcode[i_te]
    ylo,yhi=np.quantile(y[i_tr],.01),np.quantile(y[i_tr],.99)
    yte_raw=y[i_te].astype("float64")
    print(f"\n{'='*88}\n=== {coin} / {t} | train {len(idxA):,} rows, "
          f"test {len(c_te)} mkts / {len(i_te):,} rows\n{'='*88}",flush=True)
    for fs_name,expanded in (("base106",False),("expanded",True)):
        st=idxA if len(idxA)<=400_000 else np.random.default_rng(7).choice(idxA,400_000,replace=False)
        S=build(st,expanded)
        lo=np.quantile(S,.01,axis=0); hi=np.quantile(S,.99,axis=0)
        np.clip(S,lo,hi,out=S); mu=S.mean(0); sg=S.std(0); sg[sg==0]=1.0; del S; gc.collect()
        def mk(ix):
            A=build(ix,expanded); np.clip(A,lo,hi,out=A); A-=mu; A/=sg
            np.nan_to_num(A,copy=False); return A
        XA,Xh,Xte=mk(idxA),mk(i_ho),mk(i_te)
        print(f"   [{fs_name}] {XA.shape[1]} features",flush=True)
        for tg_name in ("raw","volnorm"):
            if tg_name=="raw":
                yA=y[idxA].astype("float64").clip(ylo,yhi); yh=y[i_ho].astype("float64")
                scale_te=None
            else:
                # DIVIDE BY TRAILING VOL AT TIME t -- a feature, not a future value
                yA=(y[idxA]/VOL[idxA]); yh=(y[i_ho]/VOL[i_ho]).astype("float64")
                q=np.quantile(yA,[.01,.99]); yA=yA.astype("float64").clip(q[0],q[1])
                scale_te=VOL[i_te]
            P={}
            bb=(-9e9,None)
            for al in (0.3,1,3,10,30,100,300,1000):
                rr=Ridge(alpha=al).fit(XA,yA,sample_weight=wA)
                v=r2_zero(yh,rr.predict(Xh),wh)
                if v>bb[0]: bb=(v,rr)
            P["ridge"]=bb[1].predict(Xte)
            bl=(-9e9,None,None)
            for cfg in (dict(num_leaves=7,min_child_samples=5000,learning_rate=0.02),
                        dict(num_leaves=31,min_child_samples=2000,learning_rate=0.03)):
                gg=lgb.LGBMRegressor(n_estimators=1500,feature_fraction=.6,subsample=.7,
                    subsample_freq=1,reg_lambda=10.0,verbose=-1,seed=11,n_jobs=8,**cfg)
                gg.fit(XA,yA,sample_weight=wA,eval_set=[(Xh,yh)],eval_sample_weight=[wh],
                       callbacks=[lgb.early_stopping(60,verbose=False)])
                v=r2_zero(yh,gg.predict(Xh),wh)
                if v>bl[0]: bl=(v,gg,cfg)
            P["lgbm"]=bl[1].predict(Xte)
            ps=[mp(fit_mlp(XA,yA,wA,Xh,yh,wh,seed=s_),Xte) for s_ in SEEDS]
            P["mlp"]=np.mean(ps,axis=0)
            for k,p in P.items():
                # score in NORMALISED units, and back-transformed to RAW units
                r_n=float(r2_zero((yte_raw/scale_te) if scale_te is not None else yte_raw,
                                  p,wte))
                p_raw=p*scale_te if scale_te is not None else p
                r_r=float(r2_zero(yte_raw,p_raw,wte))
                print(f"   {fs_name:9s} {tg_name:8s} {k:6s} "
                      f"norm-units {r_n:+.5f}   RAW-units {r_r:+.5f}",flush=True)
                res[f"{coin}|{t}|{fs_name}|{tg_name}|{k}"]={"r2_norm":r_n,"r2_raw":r_r,
                    "nfeat":int(XA.shape[1])}
                res.setdefault("_pred_key",{})
                np.save(f"/tmp/p_{coin}_{t}_{fs_name}_{tg_name}_{k}.npy",p_raw)
            save(); del P; gc.collect()
        del XA,Xh,Xte; gc.collect()
    print("   PAIRED (raw units, same test markets):",flush=True)
    def ld(fs,tg,k):
        p=f"/tmp/p_{coin}_{t}_{fs}_{tg}_{k}.npy"
        return np.load(p) if os.path.exists(p) else None
    for k in ("ridge","lgbm","mlp"):
        b=ld("base106","raw",k)
        for lbl,p in (("expanded feats",ld("expanded","raw",k)),
                      ("volnorm target",ld("base106","volnorm",k)),
                      ("both",ld("expanded","volnorm",k))):
            if b is None or p is None: continue
            dm,dlo,dhi,sig=paired(ct,yte_raw,p,b,wte)
            print(f"     {k:6s} {lbl:15s} dR2={dm:+.5f} CI[{dlo:+.5f},{dhi:+.5f}] "
                  f"-> {'SIGNIFICANT' if sig else 'not sig'}",flush=True)
            res[f"{coin}|{t}|paired|{k}|{lbl}"]={"delta":dm,"ci":[dlo,dhi],"sig":sig}
    save()
save(); print(f"\nwrote {OUT}")
