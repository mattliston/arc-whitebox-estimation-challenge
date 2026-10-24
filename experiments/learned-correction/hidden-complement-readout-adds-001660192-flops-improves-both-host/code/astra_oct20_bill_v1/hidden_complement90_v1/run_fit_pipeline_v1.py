"""Wait for frozen tariff PASS, then one original-guard training-only fit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(A/'cloud_linux_markline_v1'));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 assert sys.platform=='linux'
 prereg=HERE/'PREREG_v1.json';assert sha(prereg)=='eaf6dbc3c1a4cef5d1f6a81aea47394659dc4ff8b22c35d606d8dfa9b6ccebc9'
 pins=read(HERE/'FIT_RECIPE_v1.json');assert pins['prereg_sha256']==sha(prereg)
 fitter=HERE/'fit_hidden_complement_v1.py';assert sha(fitter)==pins['fitter_sha256'] and sha(__file__)==pins['pipeline_sha256']
 guard=A/'cloud_linux_capacity_v7/guard_linux_v7.py';assert sha(guard)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
 pg=RUNS/'hidden_complement90_price_v1_GUARD.json';deadline=time.monotonic()+21600
 while not pg.exists():
  assert time.monotonic()<deadline,'Frozen price qualification timeout';time.sleep(15)
 g=read(pg);assert g['successful'] and g['child_exit']==0 and g['whole_group_retired'] and not g['cleanup_errors'] and g['guard_sha256']==sha(guard)
 pp=RUNS/'hidden_complement90_price_v1/PRICE_v1.json';p=read(pp);assert p['successful'] and p['prereg_sha256']==sha(prereg)
 if not p['passes_price_gate']:
  write(HERE/'FIT_PIPELINE_FINAL_v1.json',dict(utc=utc(),successful=True,status='PRICE_GATE_CLOSED',price_sha256=sha(pp),no_truth_opened=True));return
 out=RUNS/'hidden_complement90_fit_v1';receipt=RUNS/'hidden_complement90_fit_v1_GUARD.json';log=RUNS/'hidden_complement90_fit_v1.log'
 assert not out.exists() and not receipt.exists() and not log.exists()
 cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(guard),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),'--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--','/home/arcworker/arc-envs/judge/bin/python','-B',str(fitter)]
 with log.open('x') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1'),start_new_session=True)
 launch=dict(utc=utc(),pid=proc.pid,command=cmd,price_sha256=sha(pp),price_guard_sha256=sha(pg),recipe_sha256=sha(HERE/'FIT_RECIPE_v1.json'),script_sha256=sha(fitter),receipt=str(receipt),log=str(log))
 write(HERE/'FIT_LAUNCH_v1.json',launch);print(json.dumps(launch),flush=True);code=proc.wait()
 g=read(receipt);assert code==0 and g['successful'] and g['child_exit']==0 and g['whole_group_retired'] and not g['cleanup_errors']
 fit=out/'FITTED_v1.json';r=read(fit);assert r['script_sha256']==sha(fitter) and r['prereg_sha256']==sha(prereg) and r['no_protected_truth_read']
 final=dict(utc=utc(),successful=True,exit_code=code,status=r['status'],passed=r['passed'],fit_sha256=sha(fit),guard_sha256=sha(receipt),no_native_launched=True)
 write(HERE/'FIT_PIPELINE_FINAL_v1.json',final);print(json.dumps(final),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  path=HERE/'FIT_PIPELINE_FINAL_v1.json'
  if not path.exists():write(path,dict(utc=utc(),successful=False,error=repr(error),script_sha256=sha(__file__)))
  raise

