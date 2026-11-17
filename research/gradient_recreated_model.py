"""Held-out input gradients for the two-seed provisional BTC reconstruction.

Run on the Azure analysis VM while the provisional frame is mounted. The eight
assumed inactive inputs are marked and excluded from the ranked active list.
"""
import ast,csv,json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn

torch.set_num_threads(4)
ROOT=Path('/var/lib/kalshi-study/reproduction-btc-20261117')
OUT=ROOT/'provisional-output'; FRAME=Path('/mnt/scratch/reproduction/provisional_frame')
meta=json.loads((FRAME/'meta.json').read_text()); pre=np.load(OUT/'BTC_preprocessing.npz'); features=pre['features'].astype(str).tolist(); assert features==meta['features']
N=meta['N']; X=np.memmap(FRAME/'X.dat',mode='r',dtype='float32',shape=(N,len(features))); mid=np.memmap(FRAME/'mid.dat',mode='r',dtype='float64',shape=(N,))
test=np.load(OUT/'BTC_predictions.npz')['test_row_indices']; split=json.loads((OUT/'BTC_split.json').read_text()); rng=np.random.default_rng(20261117); selected=[]; counts={}
for ticker in split['test']:
 off,n=meta['offsets'][ticker]; left=np.searchsorted(test,off);right=np.searchsorted(test,off+n); eligible=test[left:right]; k=min(150,len(eligible))
 if k: selected.extend(rng.choice(eligible,k,replace=False));counts[ticker]=k
selected=np.asarray(selected,dtype='int64'); assert len(selected)>0
# Original training code's clipping/normalization; gradients are at the resulting input.
A=np.asarray(X[selected],dtype='float32').copy();np.clip(A,pre['lo'],pre['hi'],out=A);A-=pre['mu'];A/=pre['sg'];np.nan_to_num(A,copy=False)
tree=ast.parse((ROOT/'provisional_tc.py').read_text()); node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Net');env={'nn':nn,'W':640,'L':5,'DROP':0.2};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'model','exec'),env)
models=[]
for seed in (11,23):
 model=env['Net'](len(features));model.load_state_dict(torch.load(OUT/f'BTC_seed{seed}.pt',map_location='cpu',weights_only=True));model.eval();models.append(model)
sum_abs=np.zeros(len(features));sum_signed=np.zeros(len(features));sum_fair=np.zeros(len(features)); ys=float(pre['ys'])
for lo in range(0,len(A),1024):
 b=torch.from_numpy(A[lo:lo+1024]).clone().requires_grad_(True); pred=ys*(models[0](b).flatten()+models[1](b).flatten())/2
 grad=torch.autograd.grad(pred.sum(),b)[0].detach().numpy();sum_abs+=np.abs(grad).sum(axis=0);sum_signed+=grad.sum(axis=0);sum_fair+=(np.abs(grad)*mid[selected[lo:lo+len(b)],None]*100).sum(axis=0)
constant=set(meta['inactive_assumption']);sg=pre['sg']; low=pre['lo'];high=pre['hi'];rows=[]
for i,f in enumerate(features):
 inactive=(f in constant or not np.isfinite(sg[i]) or not np.isfinite(low[i]) or not np.isfinite(high[i]) or high[i]<=low[i])
 rows.append({'feature':f,'mean_abs_return_gradient_per_std':float(sum_abs[i]/len(selected)), 'mean_signed_return_gradient_per_std':float(sum_signed[i]/len(selected)),'mean_abs_fair_cents_per_std':float(sum_fair[i]/len(selected)),'constant_or_assumed_inactive':bool(inactive)})
rows.sort(key=lambda x:x['mean_abs_return_gradient_per_std'],reverse=True); active=[r for r in rows if not r['constant_or_assumed_inactive']]
result={'model':'provisional BTC two-seed ensemble; not final full-replay reproduction','sample':{'rows':len(selected),'test_markets_with_samples':len(counts),'max_rows_per_market':150,'selection_seed':20261117},'units':'derivative with respect to standardized, clipped input; fair cents = |return gradient| * current mid * 100; local sensitivity, not causal importance','warning':'Eight unobserved inputs are assumed inactive. Exclude constant or assumed-inactive columns from meaningful standard-deviation rankings. Full raw replay remains in progress.','ranked_active':active,'all_features':rows}
(OUT/'BTC_gradient.json').write_text(json.dumps(result,indent=2))
with (OUT/'BTC_gradient.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['feature','mean_abs_return_gradient_per_std','mean_signed_return_gradient_per_std','mean_abs_fair_cents_per_std','constant_or_assumed_inactive']);w.writeheader();w.writerows(rows)
print(json.dumps({'sample':result['sample'],'top10':active[:10]},indent=2),flush=True)
