"""Wait for the bounded Azure run, then retrieve its artifacts to this workspace.

Does not stop the VM: that requires checking for any intervening workloads.
"""
from pathlib import Path
import json, os, subprocess, sys, tarfile, time

root=Path(__file__).resolve().parent
out=root/'reproduction_btc_20261117'; out.mkdir(exist_ok=True)
marker=out/'watch_status.json'
def command(args,check=True):
    return subprocess.run(args,capture_output=True,text=True,check=check)
def save(state,**kw):
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2));os.replace(tmp,marker)
def status(name):
    p=command(['az','vm','run-command','show','-g','KALSHI','--vm-name','kalshi-analysis','--name',name,'--expand','instanceView','--query','instanceView','-o','json'])
    return json.loads(p.stdout or '{}')
save('watching')
while True:
    try:
        state=status('pitch-btc-reproduction'); current=state.get('executionState')
        if current in ('Succeeded','Failed','TimedOut','Canceled'):
            if current!='Succeeded' or state.get('exitCode') not in (None,0):
                save('job_failed',execution_state=current,exit_code=state.get('exitCode'),message=state.get('executionMessage'));sys.exit(1)
            break
        save('running',execution_state=current)
    except Exception as e:
        save('monitor_retry',error=str(e))
    time.sleep(45)
archive=out/'artifacts.tgz'
try:
    p=command(['az','storage','blob','download','--account-name','kalshiobfdfaed53','--auth-mode','login','-c','tape','-n','research/reproductions/btc-tradecheck-20261117/artifacts.tgz','-f',str(archive),'--only-show-errors','-o','none'])
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            if m.issym() or m.islnk() or m.isdev() or (out/m.name).resolve().is_relative_to(out.resolve()) is False:
                raise ValueError('Unsafe archive member: '+m.name)
        t.extractall(out,filter='data')
    result=out/'output'/'tradecheck_reproduction.json'
    comparison=command([sys.executable,str(root/'compare_reproduction.py'),str(result),'--original',str(root.parent/'TRADECHECK.json')]).stdout
    (out/'comparison.txt').write_text(comparison)
    save('retrieved',archive=str(archive),comparison=str(out/'comparison.txt'),vm_note='VM still running; inspect workloads before returning it to deallocated state.')
except Exception as e:
    save('retrieval_failed',error=str(e));raise
