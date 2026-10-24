"""One root-approved auxiliary slot; existing guard owns only this finite worker."""
import argparse,ast,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];T=O/'trainer'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def event(p,x):
 with Path(p).open('a')as f:f.write(json.dumps(x)+'\n')
# Reuse the accepted scanner exactly, without importing any candidate or Torch.
RESOURCE_SOURCE=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(RESOURCE_SOURCE)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
resource_text=RESOURCE_SOURCE.read_text();node=next(n for n in ast.parse(resource_text).body if isinstance(n,ast.FunctionDef)and n.name=='resources')
exec(compile(ast.get_source_segment(resource_text,node),str(RESOURCE_SOURCE),'exec'))
def pins(c):
 for p,h in c['pins'].items():assert sha(p)==h,p
def pending(c):
 run=Path(c['shared_run']);return sum((run/'queue'/x['filename']).exists()for x in c['frozen_queue'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-authorized',action='store_true');ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--grant',type=Path,required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
 assert a.run_authorized and sha(a.plan)==a.plan_sha256;c=json.loads(a.plan.read_text());g=json.loads(a.grant.read_text())
 assert g['approved']is True and g['reviewed_by']=='root' and g['plan_sha256']==a.plan_sha256
 assert g['controller_sha256']==sha(__file__) and g['worker_indices']==c['worker_indices'] and g['numerical_slots']==2
 assert g['purpose']==c['purpose']=='finite_S61_existing_eval_queue_acceleration_worker17'
 assert g['UC8_seal_collectors_retired']is True and g['logical_total_reserved']<=10
 for x in g['seal_retirement_evidence']:assert sha(x['path'])==x['sha256']
 assert len(g['seal_retirement_evidence'])>=1
 assert a.index in c['worker_indices'] and all(os.environ.get(k)==v for k,v in c['environment'].items())
 assert os.getpriority(os.PRIO_PROCESS,0)==0; pins(c)
 assert pending(c)>=64,'Fewer than64 fixed existing pending evals; no auxiliary child'
 h=D/('run_s61_aux_w%d_v1'%a.index);h.mkdir(exist_ok=False)
 write(h/'PLAN_v1.json',dict(plan_sha256=a.plan_sha256,grant_path=str(a.grant.resolve()),grant_sha256=sha(a.grant),controller_pid=os.getpid(),worker_index=a.index,max_claims=24,pending_fixed_count=pending(c)))
 started=time.monotonic()
 while True:
  r=resources();okay=r['free_percent']>=60 and r['counted']<10;event(h/'admission_wait.jsonl',dict(admitted=okay,elapsed_seconds=time.monotonic()-started,**r))
  if okay:event(h/'admission.jsonl',r);break
  assert time.monotonic()-started<1800,'Bounded admission wait expired';time.sleep(15)
 assert pending(c)>=64,'Queue drained during admission; no child'
 command=[c['controller_python'],'-B',c['guard'],'--record',str(h/'GUARD_v1.json'),'--seconds','14400','--footprint-limit','7500000000','--rss-backstop','10000000000',c['torch_python'],'-B',c['finite_worker'],'--plan',str(a.plan.resolve()),'--plan-sha256',a.plan_sha256,'--index',str(a.index),'--home',str(h)]
 child=None
 try:
  with (h/'worker.log').open('x')as log:
   child=subprocess.Popen(command,cwd=T,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
   write(h/'OWNERSHIP_v1.json',dict(controller_pid=os.getpid(),owned_guard_pid=child.pid,worker_index=a.index,command=command,controller_nice=0,worker_nice=10,environment=c['environment']))
   print(json.dumps(dict(worker_index=a.index,controller_pid=os.getpid(),owned_guard_pid=child.pid,launched=True)),flush=True)
   while child.poll()is None:
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     r=resources();event(h/'resources.jsonl',r);assert r['free_percent']>=50 and r['counted']<=10,'Live resource bound; retire only owned auxiliary guard'
 except BaseException as e:
  if child is not None and child.poll()is None:
   child.send_signal(signal.SIGINT);child.wait(timeout=30)
  write(h/'STOPPED_v1.json',dict(error=repr(e),owned_guard_pid=child.pid if child else None,returncode=child.poll()if child else None,no_retry=True,incomplete_own_claims_retained=True));raise
 guard=json.loads((h/'GUARD_v1.json').read_text());assert child.returncode==0 and guard['completed']is True and guard['exit_code']==0 and guard['error']is None and guard['whole_group_retired']is True and not guard['remaining_rows']
 assert guard['max_footprint_bytes']<7500000000 and guard['max_rss_bytes']<10000000000
 complete=json.loads((h/'WORKER_COMPLETE_v1.json').read_text());assert 0<=complete['completed_claims']<=24
 pins(c);write(h/'READY_v1.json',dict(status='complete_await_actual_controller_retirement',worker_index=a.index,completed_claims=complete['completed_claims'],guard_path=str(h/'GUARD_v1.json'),guard_sha256=sha(h/'GUARD_v1.json'),worker_complete_sha256=sha(h/'WORKER_COMPLETE_v1.json'),all_owned_guard_groups_retired=True,no_followup_authority=True))
 print(json.dumps(dict(worker_index=a.index,completed_claims=complete['completed_claims'],guard_retired=True)),flush=True)
def interrupted(signum,frame):raise KeyboardInterrupt('Owned finite S61 controller signal%s'%signum)
if __name__=='__main__':
 prior=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,prior)
