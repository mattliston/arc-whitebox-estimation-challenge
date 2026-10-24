"""One original guarded first-stage source build; never launches native captures."""
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
def main():
 recipe=read(HERE/'PARENT_BUILD_RECIPE_v1.json');assert recipe['pipeline_sha256']==sha(__file__)
 assert sha(A/'guard_linux_v7.py')==GUARD
 for name,pin in recipe['pins'].items():assert sha(HERE/name)==pin
 script=HERE/'build_parent_sources_v1.py';stem='on_policy_head90_parent_build_v1'
 log=RUNS/(stem+'.log');receipt=RUNS/(stem+'_GUARD.json');py='/home/arcworker/arc-envs/judge/bin/python'
 assert not log.exists() and not receipt.exists()
 cmd=[py,'-B',str(A/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
  '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',py,'-B',str(script)]
 with log.open('x') as stream:
  proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1'))
  row=dict(utc=utc(),guard_pid=proc.pid,command=cmd,script_sha256=sha(script),guard_sha256=GUARD,
   pipeline_sha256=sha(__file__),log=str(log),receipt=str(receipt),blas_threads=1,no_native_launch=True)
  write(HERE/'PARENT_BUILD_LAUNCH_v1.json',row);print(json.dumps(row),flush=True);code=proc.wait()
 g=read(receipt)
 assert code==0 and g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['max_owned_swap_bytes']==0
 p=HERE/'PARENT_SOURCES_v1.json';result=read(p);assert result['successful'] and len(result['families'])==6
 write(HERE/'PARENT_BUILD_FINAL_v1.json',dict(utc=utc(),successful=True,source_manifest_sha256=sha(p),
  guard_path=str(receipt),guard_sha256=sha(receipt),pipeline_sha256=sha(__file__),prereg_sha256=sha(HERE/'PREREG_v1.json'),
  no_native_launch=True,root_review_required_before_capture=True))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  p=HERE/'PARENT_BUILD_FINAL_v1.json'
  if not p.exists():write(p,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__),no_native_launch=True))
  raise
