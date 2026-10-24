"""Run fixed guarded selftest, target-free90 reconstruction, then gated one-fit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs');HD=HERE.parent/'full_h78_train90_v1'
A=HERE.parents[2]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0,str(A));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
PY='/home/arcworker/arc-envs/judge/bin/python'
ENV=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1')
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def start(tag,script,arguments=()):
 receipt=RUNS/(tag+'_GUARD_v1.json');log=RUNS/(tag+'.log')
 assert not receipt.exists() and not log.exists()
 cmd=[PY,'-B',str(A/'guard_linux_v7.py'),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
  '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',PY,'-B',str(script),*arguments]
 with log.open('x') as stream:p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=ENV)
 item=dict(utc=utc(),guard_pid=p.pid,tag=tag,command=cmd,script_sha256=sha(script),pipeline_sha256=sha(__file__),guard_sha256=GUARD,
  prereg_sha256=sha(HERE/'PREREG_FIRST128_v1.json'),log=str(log),guard_receipt=str(receipt),blas_threads=1,no_native_auto_launch=True)
 write(HERE/(tag+'_LAUNCH_v1.json'),item);print(json.dumps(item),flush=True)
 return p,receipt
def finish(pair):
 p,path=pair;code=p.wait();g=read(path)
 assert code==0 and g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'],(code,str(path))
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==1 and g['max_owned_swap_bytes']==0
 return sha(path)
def main():
 recipe=read(HERE/'PIPELINE_RECIPE_v1.json');assert recipe['script_sha256']==sha(__file__)
 for name,pin in recipe['pins'].items():assert sha(HERE/name)==pin
 assert sha(A/'guard_linux_v7.py')==GUARD
 fit=HERE/'fit_first128_local90_v1.py';recon=HERE/'reconstruct_first128_train90_v1.py'
 selfguard=finish(start('first_hidden128_fit_selftest_v1',fit,['--self-test']))
 deadline=time.monotonic()+21600;previous=HD/'CAPTURE_FINAL_v1.json'
 while not previous.exists():
  assert time.monotonic()<deadline,'fullH90 capture completion timeout';time.sleep(10)
 final=read(previous);assert final['successful'] and final['networks']==90 and final['all_predictions_byte_identical'] and final['no_truth_arrays_read']
 custodyguard=finish(start('first_hidden128_reconstruct90_custody',recon,['--prepare']))
 pairs=[start('first_hidden128_reconstruct90_shard%02d'%i,recon,['--shard',str(i)]) for i in range(1,7)]
 # All six receive originalguard admissions independently; wait all even if one exits nonzero.
 exits=[p.wait() for p,_ in pairs];assert exits==[0]*6,exits
 shardguards=[finish(pair) for pair in pairs]
 reports=[RUNS/'first_hidden128_reconstruct90_v1'/('shard%02d'%i)/'RECON_COMPLETE_v1.json' for i in range(1,7)]
 passed=all(read(p)['reconstruction_passed'] for p in reports)
 if not passed:
  write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status='CLOSED_TARGET_FREE_RECONSTRUCTION_GATE_FAIL',passed=False,
   reconstruction_reports=[dict(path=str(p),sha256=sha(p)) for p in reports],no_truth_arrays_read=True,no_response_fit=True,no_candidate_export=True,no_native_launch=True));return
 fitguard=finish(start('first_hidden128_local90_fit_v1',fit))
 fp=RUNS/'first_hidden128_local90_fit_v1/FITTED_v1.json';f=read(fp)
 write(HERE/'PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status=f['status'],passed=f['passed'],fitted_sha256=sha(fp),
  fit_guard_sha256=fitguard,custody_guard_sha256=custodyguard,shard_guard_sha256s=shardguards,selftest_guard_sha256=selfguard,
  capture_final_sha256=sha(previous),source_qualification_required_if_passed=True,no_candidate_export=True,no_native_launch=True,no_adjusted_gain_claim=True))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  path=HERE/'PIPELINE_FINAL_v1.json'
  if not path.exists():write(path,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__),no_native_launch=True))
  raise
