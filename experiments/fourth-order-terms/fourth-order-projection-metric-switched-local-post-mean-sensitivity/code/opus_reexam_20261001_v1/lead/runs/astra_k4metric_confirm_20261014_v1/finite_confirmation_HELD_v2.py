"""Source-only confirmation of the unchanged16-request K4 metric winner.
One root-granted slot, original worker and arithmetic, no result values read.
"""
import argparse,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
OLD=D.parent/'astra_k4metric_20261013_v1';RUN=OLD/'run_confirm'
PY=O.parents[4]/'.venv/bin/python';MODEL=Path('/Users/matt/anaconda3/bin/python')
GUARD=O/'release/guard_footprint_v3.py';WORKER=OLD/'worker_k4metric_finite_v2.py'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def resources():
 raw=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,rss=,comm='],text=True,timeout=10);workers=[]
 for line in raw.splitlines():
  a=line.split(None,3)
  if len(a)==4 and int(a[2])>1048576 and 'python'in Path(a[3]).name.lower():workers.append(dict(pid=int(a[0]),ppid=int(a[1]),rss_kib=int(a[2])))
 free=int(re.search(r'System-wide memory free percentage: (\d+)%',subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=15)).group(1))
 return dict(utc=stamp(),free_percent=free,workers=workers,truth_reserve=1,counted=len(workers)+1)
def pins(c):
 for p,h in c['source_pins'].items():assert sha(p)==h,('source_pin',p)
 for p,h in c['preserved_results'].items():assert sha(p)==h,('preserved_result',p)
 for p,h in c['truth_pins'].items():assert sha(p)==h,('truth_pin',p)
class RequestSnapshotChanged(RuntimeError):pass
def request_snapshot(c):
 paths={sub:sorted((RUN/sub).glob('*.json'))for sub in('queue','claimed')};req={};duplicate=False
 for sub in('queue','claimed'):
  for p in paths[sub]:
   data=p.read_bytes();x=json.loads(data);ident=x['id'];digest=hashlib.sha256(data).hexdigest()
   assert ident in c['requests']and digest==c['requests'][ident]['sha256'],('request_pin',ident)
   assert x['mode']=='eval'and x['rank']==c['requests'][ident]['seed']
   duplicate=duplicate or ident in req;req[ident]=dict(path=str(p),sha256=digest)
 # One atomic queue-to-claimed rename can cross either directory scan. Retry
 # only this metadata snapshot; changed request bytes always fail immediately.
 after={sub:sorted((RUN/sub).glob('*.json'))for sub in('queue','claimed')}
 if paths!=after or duplicate or set(req)!=set(c['requests'])or len(req)!=16:raise RequestSnapshotChanged('request paths moved or inventory incomplete/duplicate')
 return req,paths
def inventory(c):
 for attempt in range(5):
  try:req,paths=request_snapshot(c);break
  except (FileNotFoundError,RequestSnapshotChanged):
   if attempt==4:raise
   time.sleep(.05)
 result={p.stem:str(p)for p in(RUN/'results').glob('*.json')}
 assert set(result)<=set(req)
 errors=sorted(str(p)for p in(RUN/'results').glob('*.error'))
 assert not errors,('worker_error_requires_root',errors)
 for p,h in c['preserved_results'].items():assert sha(p)==h
 return dict(completed=len(result),expected=16,queue=len(paths['queue']),claimed=len(paths['claimed']),result_pins={p:sha(p)for p in result.values()})
def policy(c):
 p=Path(c['gain_activation']['path']);assert sha(p)==c['gain_activation']['sha256'];a=json.loads(p.read_text())
 assert a['status']=='root_registered_prospectively' and 'metric'in a['eligible']and a['complete_at_registration']['metric']is False
 for name,h in a['legacy_decision_sha256'].items():assert (sha(O/name)if(O/name).exists()else None)==h
 assert a['addendum_sha256']==sha(c['gain_addendum']['path'])==c['gain_addendum']['sha256']
 for p in c['decision_paths']:assert not Path(p).exists(),('metric_confirmation_already_decided',p)
 rp=Path(c['root_fresh_decision_path']);assert sha(rp)==c['root_fresh_decision_sha256'];root=json.loads(rp.read_text())
 sp=Path(c['screen_decision_path']);assert sha(sp)==c['screen_decision_sha256'];selected=json.loads(sp.read_text())
 assert root['status']=='root_full48_fresh_screen_decided' and root['decision']==selected and selected['complete']is True and selected['winner']=='h'
 assert selected['criterion']=='strict_negative_no_minimum_gain' and selected['arms']['h']['mean_log_pct']<0
 assert root['root_retirement_sha256']==c['root_screen_retirement_sha256'] and sha(root['root_retirement_path'])==c['root_screen_retirement_sha256']
def clean_guard(g):
 return g.get('completed')is True and g.get('exit_code')==0 and g.get('error')is None and g.get('whole_group_retired')is True and g.get('remaining_rows')==[] and 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
def absent(ids):
 out=[]
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,comm='],text=True,timeout=10).splitlines():
  a=line.split(None,3)
  if len(a)==4 and any(int(a[i])in ids for i in range(3)):out.append(a)
 return out
def config(path,digest,gp):
 assert sha(path)==digest;c=json.loads(Path(path).read_text());g=json.loads(gp.read_text())
 assert c['status']=='REGISTERED' and c['purpose']=='unchanged_k4metric_confirm16'
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert c['worker_sha256']==sha(WORKER)=='62ccff04d9a99ed5d9ac40e0bf0cd43a8bcea80bc56ddd9b4b9c55fa421af9e7'
 assert g['approved']is True and g['reviewed_by']=='root' and g['purpose']==c['purpose']
 assert g['protocol_sha256']==digest and g['runner_sha256']==sha(__file__)and g['numerical_slots']==1
 assert g['archive_owned_STOP_authorized']is True and g['preserve_full_fresh48']is True
 assert g['resource_admission']['logical_total_after_grant']<=8 and g['resource_admission']['logical_max']==8
 assert g['resource_admission']['entry_free_percent_min']==60 and g['resource_admission']['counted_before_launch_max']==7
 assert g['run_name']==c['run_name']=='run_confirmation_owner_v1'
 assert c['confirmation_seeds']==list(range(7100592,7100608))and c['arms']=={'h':.5}
 assert c['admission_wait_policy']==g['admission_wait_policy']==dict(poll_seconds=15,status_seconds=30,max_wait_seconds=1800,no_child_while_waiting=True,no_automatic_retry=True)
 assert len(c['requests'])==16 and len(c['preserved_results'])==48
 assert all(os.environ.get(k)=='1'for k in THREADS)and os.getpriority(os.PRIO_PROCESS,0)==0
 pins(c);policy(c);i=inventory(c);assert i['completed']==0 and i['queue']==16 and i['claimed']==0
 assert(RUN/'STOP').exists()and sha(RUN/'STOP')==c['held_stop_sha256']and not(RUN/'LAUNCH_LOCK').exists()
 for p,h in c['historical_guard_pins'].items():assert sha(p)==h and clean_guard(json.loads(Path(p).read_text()))
 assert not absent(set(c['historical_owned_ids']))
 return c,g
def child_environment():
 return dict(TRAINER_CARD='M',TRAINER_PL='1',TRAINER_F4C='1',TRAINER_THREADS='2',TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0',TRAINER_RSS_LIMIT='7500000000',TRAINER_GENERATED_MIN_N='15990784',TRAINER_GENERATED_ALLOW_FILE=str(RUN/'allowed_seeds.txt'),PYTHONPATH=str(O/'trainer'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_NUM_THREADS='1')
def ensure_stop():
 try:
  with(RUN/'STOP').open('x')as f:f.write('Owned confirmation boundary hold; no automatic resume.\n')
 except FileExistsError:pass
def admit(h):
 started=time.monotonic();attempt=0;last=None
 while True:
  r=resources();elapsed=time.monotonic()-started;attempt+=1;okay=r['free_percent']>=60 and r['counted']<8 and elapsed<=1800
  event=dict(controller_pid=os.getpid(),attempt=attempt,elapsed_seconds=elapsed,admitted=okay,**r)
  with(h/'admission_wait.jsonl').open('a')as f:f.write(json.dumps(event)+'\n')
  if okay:write(h/'ADMISSION_v1.json',event);return event
  if last is None or elapsed-last>=30:
   print(json.dumps(dict(status='waiting_for_headroom_no_worker',free_percent=r['free_percent'],counted=r['counted'],elapsed_seconds=elapsed,slot_reserved=True)),flush=True);last=elapsed
  assert elapsed<1800,'Bounded admission wait expired; no automatic retry'
  time.sleep(min(15,1800-elapsed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True);p.add_argument('--run-authorized',action='store_true');a=p.parse_args();assert a.run_authorized
 c,g=config(a.protocol,a.protocol_sha256,a.grant);initial=resources()
 h=D/c['run_name'];h.mkdir(exist_ok=False);lock=RUN/'LAUNCH_LOCK';lock.mkdir()
 record=h/'guard_w0_v1.json';log=h/'worker_w0_v1.log';archive=RUN/'STOP.held_astra_confirm_20261014_v1';assert not archive.exists()
 command=[str(PY),'-B',str(GUARD),'--record',str(record),'--seconds','14400','--footprint-limit','7500000000','--rss-backstop','7500000000','/usr/bin/env']+[k+'='+v for k,v in child_environment().items()]+[str(MODEL),'-B',str(WORKER),str(RUN),'0']
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for k in THREADS:env[k]='1'
 for k in('TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):env.pop(k,None)
 child=None;retired=False;failure=None;summary=None
 try:
  write(h/'PLAN_v1.json',dict(utc=stamp(),protocol_sha256=a.protocol_sha256,root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),controller_pid=os.getpid(),initial=initial,command=command,environment=child_environment(),initial_result_pins=c['preserved_results'],expected_ids=sorted(c['requests']),stop_archive=str(archive),scientific_values_read=False))
  admit(h)
  (RUN/'STOP').rename(archive)
  with log.open('x')as stream:
   child=subprocess.Popen(command,cwd=O/'trainer',env=env,stdout=stream,stderr=subprocess.STDOUT)
   write(h/'OWNERSHIP_v1.json',dict(controller_pid=os.getpid(),guard_pid=child.pid,command=command,guard_record=str(record),controller_nice=0,worker_nice=10))
   print(json.dumps(dict(controller_pid=os.getpid(),guard_pid=child.pid,managed_foreground=True)),flush=True)
   while child.poll()is None:
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     r=resources();assert r['free_percent']>=50 and r['counted']<=10,('live_resource_bound',r)
     i=inventory(c);print(json.dumps(dict(utc=r['utc'],free_percent=r['free_percent'],counted=r['counted'],completed=i['completed'],expected=16)),flush=True)
  gr=json.loads(record.read_text());retired=gr.get('whole_group_retired')is True and gr.get('remaining_rows')==[]
  assert child.returncode==0 and clean_guard(gr),('guard_failed',str(record))
  assert not absent({child.pid,gr['pid']})
  pins(c);policy(c);summary=inventory(c)
  write(h/'OWNER_READY_v1.json',dict(status='complete16_await_actual_controller_retirement'if summary['completed']==16 else 'boundary_held_incomplete_await_actual_controller_retirement',utc=stamp(),protocol_sha256=a.protocol_sha256,controller_pid=os.getpid(),guard_pid=child.pid,worker_pid=gr['pid'],guard_path=str(record),guard_sha256=sha(record),admission_sha256=sha(h/'ADMISSION_v1.json'),admission_wait_sha256=sha(h/'admission_wait.jsonl'),whole_group_retired=True,summary=summary,scientific_values_read=False,scientific_means_computed=False,no_followup_authorization=True))
 except BaseException as e:
  failure=repr(e)
  if child is None:retired=True
  else:
   if child.poll()is None:
    try:child.send_signal(signal.SIGINT)
    except ProcessLookupError:pass
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
   try:
    gr=json.loads(record.read_text());retired=child.poll()is not None and gr.get('whole_group_retired')is True and gr.get('remaining_rows')==[] and not absent({child.pid,gr['pid']})
   except (OSError,ValueError,KeyError,subprocess.SubprocessError):retired=False
  write(h/'STOPPED_v1.json',dict(error=failure,owned_retirement_verified=retired,slot_reservation_retained=not retired,no_retry=True,scientific_values_read=False))
  raise
 finally:
  ensure_stop()
  if retired:lock.rmdir()
 return 0 if summary and summary['completed']==16 else 2
def interrupted(signum,frame):raise KeyboardInterrupt('Owned metric confirmation signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:raise SystemExit(main())
 finally:signal.signal(signal.SIGTERM,old)
