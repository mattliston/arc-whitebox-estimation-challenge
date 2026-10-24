"""Run the one frozen physical29 fit only after all90 source-custody checks pass."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
HD=HERE.parent/'breakthrough_v1/full_h78_train90_v1'
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0,str(A));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 recipe=read(HERE/'PIPELINE_RECIPE_v1.json')
 assert recipe['script_sha256']==sha(__file__)
 for name,pin in recipe['pins'].items():assert sha(HERE/name)==pin
 assert sha(A/'guard_linux_v7.py')==GUARD
 previous=HD/'CAPTURE_FINAL_v1.json';deadline=time.monotonic()+21600
 while not previous.exists():
  assert time.monotonic()<deadline,'fullH90 capture completion timeout';time.sleep(10)
 final=read(previous)
 assert final['successful'] and final['networks']==90 and final['all_predictions_byte_identical'] and final['no_truth_arrays_read']
 script=HERE/'fit_physical29_v1.py';log=RUNS/'physical_input29_fit_v1.log';receipt=RUNS/'physical_input29_fit_v1_GUARD.json'
 assert not log.exists() and not receipt.exists()
 py='/home/arcworker/arc-envs/judge/bin/python'
 cmd=[py,'-B',str(A/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
  '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',py,'-B',str(script)]
 with log.open('x') as stream:
  proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
   env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1'))
  launch=dict(utc=utc(),guard_pid=proc.pid,command=cmd,script_sha256=sha(script),guard_sha256=GUARD,
   prereg_sha256=sha(HERE/'PREREG_v1.json'),pipeline_sha256=sha(__file__),capture_final_sha256=sha(previous),
   log=str(log),guard_receipt=str(receipt),blas_threads=1,no_native_auto_launch=True)
  write(HERE/'FIT_LAUNCH_v1.json',launch);print(json.dumps(launch),flush=True);code=proc.wait()
 guard=read(receipt)
 assert code==0 and guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
 assert guard['guard_sha256']==GUARD and guard['max_owned_swap_bytes']==0
 fitp=RUNS/'physical_input29_fit_v1/FITTED_v1.json';fit=read(fitp)
 write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status=fit['status'],passed=fit['passed'],
  fitted_sha256=sha(fitp),fit_guard_sha256=sha(receipt),source_qualification_required_if_passed=True,
  no_candidate_exported=True,no_native_launch=True,no_adjusted_gain_claim=True))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  p=HERE/'PIPELINE_FINAL_v1.json'
  if not p.exists():write(p,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__)))
  raise

