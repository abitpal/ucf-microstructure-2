"""Inventory date-partitioned raw crypto blobs without downloading contents."""
import concurrent.futures, datetime, json, subprocess, sys
prefix=sys.argv[1] if len(sys.argv)>1 else 'raw/regime=crypto'
start=datetime.date(2026,9,15); days=[str(start+datetime.timedelta(days=i)) for i in range(25)]
def inspect(day):
 cmd=['az','storage','blob','list','--account-name','kalshiobfdfaed53','--auth-mode','login','-c','tape','--prefix',f'{prefix}/dt={day}/','--query','{count:length(@),bytes:sum([].properties.contentLength)}','-o','json']
 p=subprocess.run(cmd,capture_output=True,text=True,check=True); result=json.loads(p.stdout);return {'day':day,**result,'truncated':'Next Marker' in p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: result=list(pool.map(inspect,days))
print(json.dumps({'days':result,'blob_count':sum(x['count'] for x in result),'bytes':sum(x['bytes'] or 0 for x in result),'truncated':any(x['truncated'] for x in result)},indent=2))
