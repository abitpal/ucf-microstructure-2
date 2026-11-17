"""Small deterministic numerical check of original vs reproduction fit functions."""
import ast, hashlib, json
import numpy as np
import torch
from tools.panel_model import r2_zero

torch.set_num_threads(2)
def load_defs(path):
    tree=ast.parse(open(path).read(), filename=path)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'Net','fit','pred'}]
    env={'np':np,'torch':torch,'nn':torch.nn,'r2_zero':r2_zero,'W':640,'L':5,'DROP':0.2,'BS':1024}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),path,'exec'),env)
    return env
orig=load_defs('/var/lib/kalshi-study/scripts/tc.py')
new=load_defs('/var/lib/kalshi-study/reproduction-btc-20261117/reproduce_tc.py')
rng=np.random.default_rng(24); X=rng.normal(size=(2048,8)).astype('float32'); y=rng.normal(size=2048).astype('float32'); w=np.ones(2048)/2048; xv=X[:256]; yv=y[:256]; wv=w[:256]
a=orig['fit'](X,y,w,xv,yv,wv,11,steps=4);b=new['fit'](X,y,w,xv,yv,wv,11,steps=4)
maxdiff=max(float(torch.max(torch.abs(x.detach()-y.detach()))) for x,y in zip(a.parameters(),b.parameters()))
print(json.dumps({'test':'synthetic deterministic 4-step fit','original_vs_reproduction_max_parameter_difference':maxdiff,'exact_match':maxdiff==0.0}))
assert maxdiff==0
