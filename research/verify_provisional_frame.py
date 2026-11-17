"""Validate provisional frame against all available rebuilt panels and samples."""
import glob,json,os
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from tools.panel_model import scale_free

base=Path('/mnt/scratch/reproduction/provisional_frame'); meta=json.loads((base/'meta.json').read_text());features=meta['features']; N=meta['N'];X=np.memmap(base/'X.dat',mode='r',dtype='float32',shape=(N,len(features)))
paths=sorted(glob.glob('/mnt/scratch/reproduction/panels/rich_BTC/panel_KXBTC*.parquet'))
cols=[f'p_{family}_{side}_{suf}_n10' for family in ('lqdist','bslopeL') for side in ('bid','ask') for suf in ('o','p')]
nonzero=0;slope_finite=0;checked=0;unreadable=[]
for p in paths:
 try:
  pf=pq.ParquetFile(p);names=pf.schema_arrow.names;sel=[x for x in cols if x in names];tab=pf.read(columns=sel,use_threads=False)
  for c in sel:
   a=tab.column(c).to_numpy(zero_copy_only=False).astype(float);ok=np.isfinite(a)
   if c.startswith('p_lqdist_'):nonzero+=int((ok&(np.abs(a)>1e-12)).sum())
   else:slope_finite+=int(ok.sum())
  checked+=1
 except Exception as e:unreadable.append([os.path.basename(p),str(e)[:200]])
if nonzero or slope_finite:raise RuntimeError(f'inactive input assumption invalid: {nonzero} nonzero distances, {slope_finite} finite slopes')
if len(paths)-checked>2:raise RuntimeError(f'{len(paths)-checked} panel reads failed')
indices=np.linspace(0,len(paths)-1,min(16,len(paths)),dtype=int); sample=[]; inactive=set(meta['inactive_assumption'])
for i in indices:
 p=paths[i];ticker=Path(p).name[len('panel_'):-len('.parquet')]
 if ticker not in meta['offsets']:continue
 rich=pd.read_parquet(p); fast=pd.read_parquet('/var/lib/kalshi-study/panel_big/'+Path(p).name,columns=['ts','k_mid','k_spread'])
 for name in ['ts','k_mid','k_spread']:
  if not np.array_equal(rich[name].to_numpy(),fast[name].to_numpy(),equal_nan=True):raise RuntimeError('raw mismatch '+ticker+' '+name)
 z=scale_free(rich[[c for c in rich if c.startswith(('k_','p_','y_','ry_')) or c=='market']]);off,n=meta['offsets'][ticker]
 if len(z)!=n:raise RuntimeError('row mismatch '+ticker)
 for j,c in enumerate(features):
  if c in inactive:continue
  a=z[c].to_numpy(dtype='float32') if c in z else np.full(n,np.nan,dtype='float32');b=X[off:off+n,j]
  if not np.array_equal(a,b,equal_nan=True):raise RuntimeError('float32 mismatch '+ticker+' '+c)
 sample.append(ticker)
result={'available_rich_panels_checked':checked,'transient_unreadable':unreadable,'inactive_distance_nonzero':nonzero,'inactive_slope_finite':slope_finite,'samples_bitwise_matched':sample,'feature_count':len(features),'warning':'Only completed panels were checked; other markets remain unverified until full replay completes.'}
(base/'verification.json').write_text(json.dumps(result,indent=2));print('VERIFIED',json.dumps(result),flush=True)
