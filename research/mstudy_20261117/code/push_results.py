"""tar report + offline wandb runs + logs -> blob research/mstudy_20261117/"""
import os, sys, tarfile, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
tag = time.strftime('%H%M')
out = f'{ROOT}/push_{tag}.tgz'
with tarfile.open(out, 'w:gz') as t:
    for d in ('report', 'runs', 'logs'):
        p = f'{ROOT}/{d}'
        if os.path.exists(p):
            t.add(p, arcname=d)
put_file(out, 'latest.tgz')
print('pushed', os.path.getsize(out) // 1024, 'KB')
os.remove(out)
