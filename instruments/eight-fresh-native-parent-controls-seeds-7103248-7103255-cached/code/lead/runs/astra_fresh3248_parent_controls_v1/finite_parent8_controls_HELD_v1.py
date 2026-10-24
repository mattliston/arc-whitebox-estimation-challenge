"""Exact current-YR7 controls only: four fixed shards, two native calls each. No candidate or grade."""
import argparse,hashlib,importlib.util,json,os,signal
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DONOR=O/'lead/runs/astra_protected_difference_read256_v1/finite_prd224_science_HELD_v2.py'
assert sha(DONOR)=='5c5afdcde0b3d81eea0f6d795b53385d04fa04743b2bcd39b6ade28c139205dd'
spec=importlib.util.spec_from_file_location('_parent8_private_reader',DONOR);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=m.q;JUDGE=D/'native_parent8_judge_HELD_v1.py';q.guard.JUDGE=JUDGE
SEEDS=list(range(7103248,7103256));SHARDS=(0,1,2,3)
def read(p):return json.loads(Path(p).read_text())
def configure(protocol,digest,grant):
 assert sha(protocol)==digest;c=read(protocol);g=read(grant)
 assert c['status']=='REGISTERED'and c['purpose']=='current_YR7_fresh3248_parent_only8'
 assert c['seeds']==SEEDS and c['shards']==list(SHARDS)and c['new_parent_calls']==8 and c['new_candidate_calls']==0
 assert set(c['candidates'])=={'parent'}and c['candidates']['parent']['sha256']=='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 assert c['resource_policy']==q.RESOURCE and c['admission_wait_policy']==q.WAIT
 assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']==c['purpose']
 assert g['protocol_sha256']==digest and g['runner_sha256']==sha(__file__)
 assert g['numerical_slots']==4 and g['max_concurrent_shards']==4 and g['logical_total_reserved']<=10
 assert g['shards']==list(SHARDS)and g['new_parent_calls']==8 and g['new_candidate_calls']==0
 assert g['run_names']==c['run_names']and g['resource_policy']==c['resource_policy']and g['admission_wait_policy']==c['admission_wait_policy']
 a=Path(c['allocation_amendment_path']);assert sha(a)==c['allocation_amendment_sha256']==g['allocation_amendment_sha256']
 ar=read(a);assert ar['approved']is True and ar['reviewed_by']=='root'and ar['parent_control_seeds']==SEEDS and ar['candidate_calls_authorized']==0
 assert ar['parent_sha256']==c['candidates']['parent']['sha256']
 assert [x['seed']for x in c['truth_metadata']]==SEEDS
 for t in c['truth_metadata']:assert t['N']==15990784 and sha(t['path'])==t['sha256']
 assert all(os.environ.get(k)=='1'for k in q.guard.THREADS)and os.getpriority(os.PRIO_PROCESS,0)==0
 assert all(os.environ.get(k)is None for k in ('PB_ALL','MODE','DRY','STATS'))
 c['_protocol_path']=str(Path(protocol).resolve());c['_protocol_sha256']=digest
 return c,g

def run(protocol,digest,grant,shard):
 c,g=configure(protocol,digest,grant);assert shard in SHARDS;h=D/c['run_names'][str(shard)];assert h.parent==D
 h.mkdir(exist_ok=False);order=[('parent',s)for s in SEEDS[shard::4]];assert len(order)==2
 q.write(h/'PLAN_v1.json',dict(protocol_path=c['_protocol_path'],protocol_sha256=digest,root_grant_path=str(Path(grant).resolve()),root_grant_sha256=sha(grant),controller_pid=os.getpid(),runner_sha256=sha(__file__),shard=shard,fixed_order=order,expected_guards=[str(h/f'native_{s}_{r}_guard.json')for r,s in order],parent_only=True,no_candidate_calls=True,no_grading=True))
 evaluations=[]
 try:
  for role,seed in order:
   configure(protocol,digest,grant);q.admit(h,role,seed)
   env=dict(os.environ,PB_ALL='1',STATS=','.join(q.STATS),GOP2_MODE='1',GOP2_SCIENCE_PROTOCOL_PATH=c['_protocol_path'],GOP2_SCIENCE_PROTOCOL_SHA256=digest,PYTHONDONTWRITEBYTECODE='1')
   item=q.guard.run_native(h,dict(pinned_files=c['source_pins'],candidates=c['candidates']),role,seed,env)
   x=m.current(c,item,'parent',seed)
   assert x['cert']['science_protocol_sha256']==digest and x['cert']['science_protocol_path']==c['_protocol_path']
   evaluations.append(dict(role=role,seed=seed,**item))
   q.write(h/f'COUNT_{len(evaluations):02d}_v1.json',dict(shard=shard,completed=len(evaluations),expected=2))
   print(json.dumps(dict(shard=shard,completed=len(evaluations),expected=2,parent_only=True)),flush=True)
  admission=m.admissions(h,order,os.getpid())
  q.write(h/'NATIVE_SHARD_READY_v1.json',dict(status='two_parent_calls_complete_await_actual_controller_retirement',protocol_sha256=digest,root_grant_sha256=sha(grant),shard=shard,controller_pid=os.getpid(),candidate_sha256=c['candidates']['parent']['sha256'],evaluations=evaluations,admission_evidence=admission,all_guards_retired=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,scientific_means_computed=False,no_candidate_calls=True,no_automatic_followup=True))
 except BaseException as e:
  q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not q.guard.OWNED_RETIREMENT_CONFIRMED));raise

def interrupted(signum,frame):raise KeyboardInterrupt('Owned parent-only controller signal%s'%signum)
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True);p.add_argument('--shard',type=int,choices=SHARDS,required=True);a=p.parse_args();run(a.protocol,a.protocol_sha256,a.grant,a.shard)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
