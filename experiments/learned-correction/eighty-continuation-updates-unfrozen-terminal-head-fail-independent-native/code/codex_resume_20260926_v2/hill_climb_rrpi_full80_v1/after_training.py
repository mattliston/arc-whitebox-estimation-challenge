"""Wait for the one live controller, then export and screen once, under serial guards."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
D=Path(__file__).resolve().parent
def pin(path):
 p=Path(path).resolve(); b=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def status(state,**kw):
 value=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state=state,**kw)
 tmp=D/'after_training_status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n')
 os.replace(tmp,D/'after_training_status.json')
 print(json.dumps(value),flush=True)
def live(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return False
 # A terminated controller can remain a zombie until its tool-session parent reaps it.
 proc=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
 if proc.returncode:
  try:os.kill(pid,0)
  except ProcessLookupError:return False
  raise RuntimeError('Unable to inspect live training controller state')
 state=proc.stdout.strip()
 return bool(state) and not state.startswith('Z')
assert len(sys.argv)==3
planpath=D/'validation_plan.json';assert pin(planpath)['sha256']==sys.argv[1]
plan=json.loads(planpath.read_text());assert plan['controller']==pin(__file__)
assert plan['schema']=='full80-native-validation-plan-v1'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
pid=int(sys.argv[2]);assert pid==plan['training_controller_pid']
for r in plan['pins']:assert pin(r['path'])==r,r['path']
assert not (D/'after_training_status.json').exists()
def run_stage(name,runtime,script,args,seconds):
 guardpath=D/(name+'_guard.json');logpath=D/(name+'.log')
 assert not guardpath.exists() and not logpath.exists()
 command=[runtime,'-B',str(script)]+args
 guarded=[plan['light'],'-B',plan['guard_source']['path'],'--record',str(guardpath),
          '--seconds',str(seconds),'--rss-limit','7500000000']+command
 status(name,command=command,guard=str(guardpath))
 with logpath.open('x') as f:rc=subprocess.run(guarded,stdout=f,stderr=subprocess.STDOUT).returncode
 g=json.loads(guardpath.read_text())
 assert rc==0 and g['command']==command and g['completed'] and g['exit_code']==0 and g['error'] is None,(name,rc)
 assert g['whole_group_retired'] and not g['remaining_rows']
 assert 0<g['max_rss_bytes']<g['rss_trigger_bytes']==7500000000 and g['max_swap_growth_bytes']==0
 assert 0<g['elapsed_s']<=seconds
 return pin(guardpath)
try:
 status('waiting_for_training',training_controller_pid=pid)
 while live(pid):
  time.sleep(10)
 complete=json.loads((D/'training_complete.json').read_text())
 assert complete['completed'] and complete['updates']==complete['selected_endpoint']==80
 assert complete['protocol']==plan['protocol'] and complete['validation_read'] is False
 assert len(complete['segments'])==20
 training_status=json.loads((D/'controller_status.json').read_text())
 assert training_status['pid']==pid and training_status['state']=='completed' and training_status['completed_updates']==80
 for r in plan['pins']:assert pin(r['path'])==r,r['path']
 eg=run_stage('export',plan['torch_runtime'],plan['exporter']['path'],[plan['protocol']['sha256']],120)
 pg=run_stage('prepare',plan['light'],plan['prepare']['path'],[sys.argv[1]],120)
 cfgpath=D/'native_config.json';cfgpin=pin(cfgpath)
 ng=run_stage('native',plan['runtime'],plan['driver']['path'],[str(cfgpath),cfgpin['sha256']],300)
 ag=run_stage('assessment',plan['runtime'],plan['assessor']['path'],[str(cfgpath),cfgpin['sha256']],120)
 assessment=json.loads((D/'native_assessment.json').read_text())
 status('completed',config=cfgpin,assessment=pin(D/'native_assessment.json'),admitted=assessment['admitted'],
        export_guard=eg,prepare_guard=pg,native_guard=ng,assessment_guard=ag,submitted=False)
except BaseException as error:
 status('stopped',error=repr(error));raise
