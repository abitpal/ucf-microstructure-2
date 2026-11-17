import os, json, re, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from torch import nn
F='/var/lib/kalshi-study/study_frame3.parquet'; OUT='/var/lib/kalshi-study/gradient_probe_20261117.json'
torch.set_num_threads(12); torch.manual_seed(11); rng=np.random.default_rng(11)
schema=pq.ParquetFile(F).schema_arrow.names
features=json.load(open('/var/lib/kalshi-study/study_frame3_meta.json'))['features']
target='ry_k_0p1s'; cols=features+[target,'market']; d=pd.read_parquet(F,columns=cols); d=d[np.isfinite(d[target])].copy(); d['market']=d['market'].astype(str)
months={k:i+1 for i,k in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])}
def key(t):
 m=re.search(r'-(\d{2})([A-Z]{3})(\d{2})(\d{4})-',t)
 if not m: raise ValueError('Unrecognized market ticker: '+t)
 y,mo,day,hm=m.groups(); return (int(y),months[mo],int(day),int(hm),t)
markets=sorted(d.market.unique(),key=key); n=len(markets); test=markets[int(.8*n):]; val=markets[int(.7*n):int(.8*n)]; train=markets[:int(.7*n)]
def sample(ms,cap):
 ix=np.flatnonzero(d.market.isin(ms).to_numpy()); return rng.choice(ix,min(len(ix),cap),replace=False)
tr=sample(train,300000); va=sample(val,50000)
# Equal number of observations per test market, with a bounded global sample.
test_indices=[]
for m in test:
 ix=np.flatnonzero(d.market.to_numpy()==m); test_indices.extend(rng.choice(ix,min(1500,len(ix)),replace=False))
te=np.array(test_indices,dtype=int)
X=d[features].to_numpy(dtype=np.float32); y=d[target].to_numpy(dtype=np.float32)
lo=np.nanquantile(X[tr],.01,axis=0); hi=np.nanquantile(X[tr],.99,axis=0); A=np.clip(X[tr],lo,hi); mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)|(sd<1e-8)]=1; mu[~np.isfinite(mu)]=0; lo[~np.isfinite(lo)]=0; hi[~np.isfinite(hi)]=0
def prep(ix): return np.nan_to_num((np.clip(X[ix],lo,hi)-mu)/sd,nan=0,posinf=0,neginf=0).astype('float32')
xt=torch.from_numpy(prep(tr)); xv=torch.from_numpy(prep(va)); xe=torch.from_numpy(prep(te)); yl,yh=np.quantile(y[tr],[.01,.99]); yt=np.clip(y[tr],yl,yh); ym=float(yt.mean()); ys=float(yt.std()); ty=torch.from_numpy(((yt-ym)/ys).astype('float32')).view(-1,1)
class Net(nn.Module):
 def __init__(self,n):
  super().__init__(); self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.4),nn.Linear(64,1))
 def forward(self,x):return self.net(x)
model=Net(len(features)); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1.0); best=float('inf'); state=None; bad=0
for epoch in range(20):
 model.train(); order=torch.randperm(len(tr))
 for ix in order.split(1024):
  opt.zero_grad(); loss=((model(xt[ix])-ty[ix])**2).mean(); loss.backward(); opt.step()
 model.eval()
 with torch.no_grad(): pred=model(xv).numpy().ravel()*ys+ym
 v=float(np.mean((pred-y[va])**2)); print('epoch',epoch+1,'validation_mse',v,flush=True)
 if v<best: best=v; state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
 else: bad+=1
 if bad>=4: break
model.load_state_dict(state); model.eval(); sums=np.zeros(len(features)); signed=np.zeros(len(features)); predictions=[]
for batch in xe.split(2048):
 batch=batch.clone().requires_grad_(True); out=model(batch).view(-1)*ys+ym; grad=torch.autograd.grad(out.sum(),batch)[0].detach().numpy(); sums+=np.abs(grad).sum(axis=0); signed+=grad.sum(axis=0); predictions.append(out.detach().numpy())
p=np.concatenate(predictions); ytst=y[te]; r2=1-float(np.sum((ytst-p)**2)/np.sum(ytst**2)); ranking=[{'feature':f,'mean_abs_gradient':float(a/len(te)),'mean_signed_gradient':float(b/len(te))} for f,a,b in zip(features,sums,signed)]; ranking.sort(key=lambda z:z['mean_abs_gradient'],reverse=True)
result={'model':'new one-hidden-layer width-64 ReLU MLP, dropout 0.4, AdamW decay 1.0, seed 11; not original saved model','dataset':F,'target':target,'feature_count':len(features),'split':'chronological market-disjoint 70/10/20','market_counts':{'train':len(train),'validation':len(val),'test':len(test)},'market_range':{'train':[train[0],train[-1]],'validation':[val[0],val[-1]],'test':[test[0],test[-1]]},'rows':{'train':len(tr),'validation':len(va),'test':len(te)},'sampling':'uniform training/validation rows; up to 1500 random rows per held-out market','gradient_units':'absolute derivative of predicted relative 0.1s move per one training-standard-deviation change in clipped feature; output unstandardized; mean over test sample','test_r2_zero':r2,'validation_mse':best,'epochs':epoch+1,'ranking':ranking}
with open(OUT,'w') as f: json.dump(result,f,indent=2)
print('RESULT '+json.dumps(result),flush=True)
