"""copy model-ready datasets + trained models to blob tape/research/mstudy_20261117/ (scratch is wiped on deallocate)"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from concurrent.futures import ThreadPoolExecutor
files = sorted(glob.glob(f'{ROOT}/models_*.pt')) + sorted(p for p in glob.glob(f'{DATA}/feat/**/*', recursive=True) if os.path.isfile(p))
def up(p):
    rel = os.path.relpath(p, ROOT)
    with open(p, 'rb') as f:
        container().upload_blob(f'{BLOB_OUT}/{rel}', f, overwrite=True, max_concurrency=4)
    return rel
with ThreadPoolExecutor(6) as ex:
    for i, r in enumerate(ex.map(up, files)):
        if i % 10 == 0:
            log(f'backup {i}/{len(files)} {r}', 'backup.log')
log(f'BACKUP DONE {len(files)} files', 'backup.log')
