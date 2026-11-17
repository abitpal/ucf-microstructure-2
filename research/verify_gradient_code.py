"""Run the queued gradient script on synthetic rows with duplicated seed-one weights."""
import json,shutil,subprocess
from pathlib import Path
import numpy as np
real=Path('/var/lib/kalshi-study/reproduction-btc-20261117'); root=Path('/mnt/scratch/reproduction/gradient-code-smoke'); frame=root/'frame';out=root/'provisional-output';frame.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
features=json.loads((real/'smoke_features.json').read_text());n=1000;markets=[f'synthetic_{i:02d}' for i in range(10)];offsets={m:[100*i,100] for i,m in enumerate(markets)};inactive=[f for f in features if f.startswith(('p_lqdist_','p_bslopeL_'))]
(frame/'meta.json').write_text(json.dumps({'N':n,'features':features,'markets':markets,'offsets':offsets,'inactive_assumption':inactive}))
a=np.memmap(frame/'X.dat',mode='w+',dtype='float32',shape=(n,len(features)));a[:]=np.random.default_rng(1).normal(size=a.shape);a.flush();m=np.memmap(frame/'mid.dat',mode='w+',dtype='float64',shape=(n,));m[:]=.5;m.flush()
np.savez_compressed(out/'BTC_preprocessing.npz',features=np.array(features),lo=np.full(len(features),-3,dtype='float32'),hi=np.full(len(features),3,dtype='float32'),mu=np.zeros(len(features),dtype='float32'),sg=np.ones(len(features),dtype='float32'),ys=np.float32(1))
np.savez_compressed(out/'BTC_predictions.npz',test_row_indices=np.arange(n,dtype='int64'));(out/'BTC_split.json').write_text(json.dumps({'test':markets}))
for seed in (11,23):shutil.copyfile(real/'provisional-output'/'BTC_seed11.pt',out/f'BTC_seed{seed}.pt')
shutil.copyfile(real/'provisional_tc.py',root/'provisional_tc.py')
s=(real/'gradient_recreated_model.py').read_text().replace("Path('/var/lib/kalshi-study/reproduction-btc-20261117')",f'Path({str(root)!r})').replace("Path('/mnt/scratch/reproduction/provisional_frame')",f'Path({str(frame)!r})')
p=root/'gradient_test.py';p.write_text(s)
r=subprocess.run(['/opt/kalshi/venv/bin/python',str(p)],capture_output=True,text=True,check=True);x=json.loads((out/'BTC_gradient.json').read_text());assert x['sample']['rows']==n and x['sample']['test_markets_with_samples']==10 and len(x['all_features'])==219 and len(x['ranked_active'])==211
print(json.dumps({'synthetic_rows':n,'markets':10,'features':219,'active_features':len(x['ranked_active']),'status':'passed'}))
