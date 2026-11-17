"""Retrieve held-out gradients after the provisional model finishes."""
from pathlib import Path
import json,os,subprocess,sys,time
out=Path(__file__).resolve().parent/'provisional_btc_20261117';out.mkdir(exist_ok=True)
status=out/'gradient_watch_status.json'
def cmd(a):return subprocess.run(a,capture_output=True,text=True,check=True)
def save(state,**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'state':state,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2));os.replace(p,status)
while True:
 try:
  x=json.loads(cmd(['az','vm','run-command','show','-g','KALSHI','--vm-name','kalshi-analysis','--name','pitch-btc-gradients','--expand','instanceView','--query','instanceView','-o','json']).stdout or '{}');state=x.get('executionState')
  if state in ('Succeeded','Failed','TimedOut','Canceled'):
   if state!='Succeeded' or x.get('exitCode') not in (None,0):save('job_failed',execution_state=state,exit_code=x.get('exitCode'));sys.exit(1)
   break
  save('running',execution_state=state)
 except Exception as e:save('monitor_retry',error=str(e))
 time.sleep(45)
try:
 for n in ('BTC_gradient.json','BTC_gradient.csv'):
  cmd(['az','storage','blob','download','--account-name','kalshiobfdfaed53','--auth-mode','login','-c','tape','-n','research/reproductions/btc-tradecheck-20261117/'+n,'-f',str(out/n),'--only-show-errors','-o','none'])
 save('retrieved',files=['BTC_gradient.json','BTC_gradient.csv'])
except Exception as e:save('retrieval_failed',error=str(e));raise
