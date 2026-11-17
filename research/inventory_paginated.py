"""Count all dated blobs, following Azure continuation tokens automatically.

Azure credentials stay in memory; the generated inventory contains no secrets.
"""
import concurrent.futures, datetime, json, os, subprocess, sys, time
from azure.core.credentials import AccessToken
from azure.storage.blob import BlobServiceClient

account='kalshiobfdfaed53'
token=json.loads(subprocess.run(['az','account','get-access-token','--resource','https://storage.azure.com/','-o','json'],capture_output=True,text=True,check=True).stdout)
class Credential:
    def get_token(self,*scopes,**kwargs):
        return AccessToken(token['accessToken'],int(token.get('expires_on', time.time()+1800)))
container=BlobServiceClient(f'https://{account}.blob.core.windows.net',credential=Credential()).get_container_client('tape')
prefixes=['raw/regime=football','polymarket/regime=sports','trades/regime=football']
start=datetime.date(2026,9,15); days=[str(start+datetime.timedelta(days=i)) for i in range(25)]
def inspect(job):
    prefix,day=job; count=size=0
    for blob in container.list_blobs(name_starts_with=f'{prefix}/dt={day}/',results_per_page=5000):
        count+=1; size+=blob.size or 0
    print(f'{prefix} {day}: {count} objects',file=sys.stderr,flush=True)
    return {'prefix':prefix,'day':day,'count':count,'bytes':size}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    rows=list(pool.map(inspect,((p,d) for p in prefixes for d in days)))
out={p:{'dates':[r for r in rows if r['prefix']==p], 'blob_count':sum(r['count'] for r in rows if r['prefix']==p),'bytes':sum(r['bytes'] for r in rows if r['prefix']==p)} for p in prefixes}
print(json.dumps(out,indent=2))
