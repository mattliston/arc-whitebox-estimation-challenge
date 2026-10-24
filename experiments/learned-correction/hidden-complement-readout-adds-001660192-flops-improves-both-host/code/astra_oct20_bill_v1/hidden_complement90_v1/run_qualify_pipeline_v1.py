"""Pass-only full source qualification waiter; never starts native evaluation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs');A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(A/'cloud_linux_markline_v1'));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 assert sys.platform=='linux'
 pins=read(HERE/'QUALIFY_RECIPE_v1.json')
 for name,key in [('qualify_passed_v1.py','qualifier_sha256'),('check_hidden_complement_tiny_v1.py','tiny_sha256'),('check_hidden_complement_import_v1.py','import_sha256')]:assert sha(HERE/name)==pins[key]
 assert sha(HERE/'PREREG_v1.json')=='eaf6dbc3c1a4cef5d1f6a81aea47394659dc4ff8b22c35d606d8dfa9b6ccebc9'
 before=HERE/'FIT_PIPELINE_FINAL_v1.json';deadline=time.monotonic()+21600
 while not before.exists():
  assert time.monotonic()<deadline,'Fixed complement fit timeout';time.sleep(15)
 finish=read(before);assert finish['successful']
 if not finish.get('passed',False):
  write(HERE/'QUALIFY_PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status='FROZEN_FIT_CLOSED_NO_EXPORT',fit_pipeline_sha256=sha(before),no_native_launch=True));return
 guard=A/'cloud_linux_capacity_v7/guard_linux_v7.py';assert sha(guard)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
 receipt=RUNS/'hidden_complement90_qualify_v1_GUARD.json';log=RUNS/'hidden_complement90_qualify_v1.log'
 assert not receipt.exists() and not log.exists()
 cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(guard),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),'--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--','/home/arcworker/arc-envs/server/bin/python','-B',str(HERE/'qualify_passed_v1.py')]
 with log.open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1'))
 launch=dict(utc=utc(),pid=p.pid,command=cmd,receipt=str(receipt),log=str(log),recipe_sha256=sha(HERE/'QUALIFY_RECIPE_v1.json'),fit_pipeline_sha256=sha(before))
 write(HERE/'QUALIFY_LAUNCH_v1.json',launch);print(json.dumps(launch),flush=True);code=p.wait();g=read(receipt)
 assert code==0 and g['successful'] and g['child_exit']==0 and g['whole_group_retired'] and not g['cleanup_errors']
 q=HERE/'native_v1/QUALIFIED_v1.json';r=read(q);assert r['successful'] and r['qualified']
 final=dict(utc=utc(),successful=True,status='SOURCE_QUALIFIED_NATIVE_REQUIRES_PARENT',qualified_sha256=sha(q),guard_sha256=sha(receipt),no_native_launch=True)
 write(HERE/'QUALIFY_PIPELINE_FINAL_v1.json',final);print(json.dumps(final),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  p=HERE/'QUALIFY_PIPELINE_FINAL_v1.json'
  if not p.exists():write(p,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__)))
  raise

