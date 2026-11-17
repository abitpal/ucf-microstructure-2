"""Assemble a provisional 219-column frame from preserved BTC study artifacts.

Eight perp-ladder features unavailable in the old frame are assumed inactive;
this assumption must be checked against *all* rebuilt rich panels. Do not call
its result an exact reproduction until the complete comparison succeeds.
"""
import glob, hashlib, json, os, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(2)
BASE=Path('/var/lib/kalshi-study'); FAST=BASE/'panel_big'; FRAME=BASE/'frame_rich.parquet'; MIDS=BASE/'midtape'
OUT=Path('/mnt/scratch/reproduction/provisional_frame'); OUT.mkdir(parents=True,exist_ok=True)
feature_file=BASE/'reproduction-btc-20261117'/'smoke_features.json'
features=json.loads(feature_file.read_text()); assert len(features)==219 and features==sorted(features)
frame=pq.ParquetFile(FRAME); schema=frame.schema_arrow.names
common=[c for c in features if c in schema]; missing=[c for c in features if c not in schema]
assert len(common)==210 and len(missing)==9 and 'size_imb_3' in missing, (len(common),missing)
assert all(c=='size_imb_3' or c.startswith(('p_lqdist_','p_bslopeL_')) for c in missing)
# Count in the stored frame, retaining no row-level strings after this pass.
counts=Counter()
for b in frame.iter_batches(batch_size=131072,columns=['market'],use_threads=False): counts.update(b.column(0).to_pylist())
paths=sorted(glob.glob(str(FAST/'panel_KXBTC15M-*.parquet')))
markets=[]; offsets={}; cursor=0
for f in paths:
 ticker=Path(f).name[len('panel_'):-len('.parquet')]
 mid=MIDS/('mid_'+ticker+'.parquet')
 if not mid.exists():continue
 n=pq.ParquetFile(f).metadata.num_rows
 if counts.get(ticker)!=n: raise RuntimeError(f'row-count mismatch {ticker}: fast={n}, frame={counts.get(ticker)}')
 markets.append(ticker);offsets[ticker]=(cursor,n);cursor+=n
N=cursor; print('MARKETS',len(markets),'ROWS',N,'FEATURES',len(features),flush=True)
if len(markets)<1000:raise RuntimeError('too few matched markets')
# Reinitialize data on each run; scratch is ephemeral, outputs are unique.
def mm(name,shape,dtype): return np.memmap(OUT/(name+'.dat'),mode='w+',shape=shape,dtype=dtype)
X=mm('X',(N,len(features)),'float32'); mid=mm('mid',(N,),'float64'); spr=mm('spr',(N,),'float64'); settles=mm('settlement',(N,),'float64'); FW={h:mm('fw_'+str(h),(N,),'float64') for h in (.1,1.,2.,5.,30.)}
feature_index={v:i for i,v in enumerate(features)}; common_indices=[feature_index[c] for c in common]
sizeidx=feature_index['size_imb_3']; inactive=[feature_index[c] for c in missing if c!='size_imb_3']
market_codes={m:i for i,m in enumerate(markets)}; written=Counter(); state_ticker=None; state=None

def load_market(ticker):
 path=FAST/('panel_'+ticker+'.parquet'); cols=['ts','k_mid','k_spread','k_bid_size_3','k_ask_size_3','k_imbalance_1','k_imbalance_3']; raw=pq.read_table(path,columns=cols,use_threads=False).to_pandas()
 ts=raw['ts'].to_numpy(dtype='float64'); m=raw['k_mid'].to_numpy(dtype='float64')/10000.; sp=raw['k_spread'].to_numpy(dtype='float64')/10000.
 u=raw['k_bid_size_3'];v=raw['k_ask_size_3']; den=u+v; size=((u-v)/den.replace(0,np.nan)).to_numpy(dtype='float32')
 tape=pq.read_table(MIDS/('mid_'+ticker+'.parquet'),columns=['ts','mid'],use_threads=False).to_pandas(); mts=tape.ts.to_numpy(dtype='float64'); mmd=tape.mid.to_numpy(dtype='float64')/10000.; order=np.argsort(mts); mts,mmd=mts[order],mmd[order]
 def at(h):
  j=np.searchsorted(mts,ts+h,side='left');valid=j<len(mts);jc=np.clip(j,0,len(mts)-1); values=np.where(valid,mmd[jc],np.nan); got=np.where(valid,mts[jc]-ts,np.nan);return np.where(got<=h*1.25+0.005,values,np.nan)
 off,n=offsets[ticker]; assert len(raw)==n
 mid[off:off+n]=m; spr[off:off+n]=sp; settles[off:off+n]=mmd[-1]
 for h in FW:FW[h][off:off+n]=at(h)
 return {'size':size,'k_imbalance_1':raw.k_imbalance_1.to_numpy(dtype='float32'),'k_imbalance_3':raw.k_imbalance_3.to_numpy(dtype='float32')}

for r in range(frame.metadata.num_row_groups):
 table=frame.read_row_group(r,columns=['market']+common,use_threads=False)
 a=np.asarray(table.column('market').to_pylist(),dtype=object); n=len(a)
 matrix=np.full((n,len(features)),np.nan,dtype='float32')
 for name,i in zip(common,common_indices):matrix[:,i]=table.column(name).to_numpy(zero_copy_only=False).astype('float32',copy=False)
 # These eight columns are known to be inactive on the checked rich panels.
 for i in inactive:
  if features[i].startswith('p_lqdist_'):matrix[:,i]=0.
 changes=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,n]
 for left,right in zip(changes[:-1],changes[1:]):
  ticker=a[left]
  if ticker not in offsets:continue
  if ticker!=state_ticker:state_ticker=ticker;state=load_market(ticker)
  pos=written[ticker];k=right-left; off,total=offsets[ticker]
  if pos+k>total:raise RuntimeError('too many rows '+ticker)
  for col in ('k_imbalance_1','k_imbalance_3'):
   old=matrix[left:right,feature_index[col]]; other=state[col][pos:pos+k]
   if not np.allclose(old,other,rtol=1e-5,atol=1e-7,equal_nan=True):raise RuntimeError('row alignment failed '+ticker+' '+col)
  matrix[left:right,sizeidx]=state['size'][pos:pos+k]
  X[off+pos:off+pos+k]=matrix[left:right];written[ticker]+=k
 del table,a,matrix
 print('FRAME ROW GROUP',r+1,'/',frame.metadata.num_row_groups,'completed_markets',len(written),flush=True)
for m in markets:
 if written[m]!=offsets[m][1]:raise RuntimeError(f'incomplete market {m}: {written[m]} != {offsets[m][1]}')
for v in [X,mid,spr,settles,*FW.values()]:v.flush()
meta={'N':N,'features':features,'markets':markets,'offsets':offsets,'frame':str(FRAME),'fast_panels':str(FAST),'midtapes':str(MIDS),'inactive_assumption':[features[i] for i in inactive], 'warning':'Provisional: assumes eight inactive perp features across all markets; alignment checked on two book features and all row counts.'}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2));print('ASSEMBLY_COMPLETE',flush=True)
