"""Rebuild later dates concurrently using the original per-market function.

Only completed parquet files are hard-linked into the main output, and the
original --resume loop skips them. Both directories reside on the same scratch
filesystem. It never overwrites a file in the main output.
"""
import datetime as dt
import multiprocessing as mp
import os
from tools.build_panel import slots
from tools.panel_parallel import one_market

SECONDARY='/mnt/scratch/reproduction/accelerated-middle'
PRIMARY='/mnt/scratch/reproduction/panels/rich_BTC'
os.makedirs(SECONDARY, exist_ok=True)
dates=['2026-09-21','2026-09-22','2026-09-23','2026-09-24']
hours=[f'{h:02d}' for h in range(24)]
jobs=[]
for day in dates:
    for ticker, opened, closed in slots('BTC',day,hours):
        jobs.append(('BTC',ticker,opened,closed,'rich',25,SECONDARY,False,True))
print('Parallel replay:',len(jobs),'markets',flush=True)
ok=fail=linked=0
with mp.Pool(8) as pool:
    for result in pool.imap_unordered(one_market,jobs):
        if result['error']:
            fail+=1
            if fail==1: print('FIRST ERROR',result,flush=True)
        else:
            ok+=1
            name='panel_'+result['ticker']+'.parquet'
            try:
                os.link(os.path.join(SECONDARY,name),os.path.join(PRIMARY,name))
                linked+=1
            except FileExistsError:
                pass
        if (ok+fail)%20==0: print('done',ok+fail,'ok',ok,'linked',linked,'failed',fail,'utc',dt.datetime.now(dt.timezone.utc).isoformat(),flush=True)
print('COMPLETE',ok,'linked',linked,'failed',fail,flush=True)
