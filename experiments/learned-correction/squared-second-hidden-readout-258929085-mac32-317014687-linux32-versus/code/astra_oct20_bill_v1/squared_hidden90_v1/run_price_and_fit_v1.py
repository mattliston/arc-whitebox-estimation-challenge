"""One frozen price-first squared-hidden2 screen, no candidate export/native."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0,str(A));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def guarded(script,stem,server=False):
 py='/home/arcworker/arc-envs/judge/bin/python'
 child='/home/arcworker/arc-envs/server/bin/python' if server else py
 log=RUNS/(stem+'.log');receipt=RUNS/(stem+'_GUARD.json')
 assert not log.exists() and not receipt.exists()
 cmd=[py,'-B',str(A/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
  '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',child,'-B',str(script)]
 with log.open('x') as stream:
  proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1'))
  row=dict(utc=utc(),guard_pid=proc.pid,command=cmd,script_sha256=sha(script),guard_sha256=GUARD,
   pipeline_sha256=sha(__file__),log=str(log),receipt=str(receipt),blas_threads=1)
  write(HERE/(stem+'_LAUNCH_v1.json'),row);print(json.dumps(row),flush=True);code=proc.wait()
 g=read(receipt)
 assert code==0 and g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['max_owned_swap_bytes']==0
 return dict(path=str(receipt),sha256=sha(receipt))
def main():
 recipe=read(HERE/'PIPELINE_RECIPE_v1.json');assert recipe['script_sha256']==sha(__file__)
 assert sha(A/'guard_linux_v7.py')==GUARD
 for name,pin in recipe['pins'].items():assert sha(HERE/name)==pin
 pg=guarded(HERE/'price_squared_hidden_v1.py','squared_hidden90_price_v1',server=True)
 pricep=RUNS/'squared_hidden90_price_v1/PRICE_v1.json';price=read(pricep)
 if not price['passes_price_gate']:
  write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status='CLOSED_PRICE_GATE',
   price_sha256=sha(pricep),price_guard=pg,no_target_arrays_read=True,no_native_launch=True));return
 fg=guarded(HERE/'fit_squared_hidden_v1.py','squared_hidden90_fit_v1')
 fitp=RUNS/'squared_hidden90_fit_v1/FITTED_v1.json';fit=read(fitp)
 write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status=fit['status'],passed=fit['passed'],
  fitted_sha256=sha(fitp),fit_guard=fg,price_sha256=sha(pricep),price_guard=pg,
  source_qualification_required_if_passed=True,no_candidate_exported=True,no_native_launch=True,no_adjusted_gain_claim=True))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  final=HERE/'PIPELINE_FINAL_v1.json'
  if not final.exists():write(final,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__)))
  raise

