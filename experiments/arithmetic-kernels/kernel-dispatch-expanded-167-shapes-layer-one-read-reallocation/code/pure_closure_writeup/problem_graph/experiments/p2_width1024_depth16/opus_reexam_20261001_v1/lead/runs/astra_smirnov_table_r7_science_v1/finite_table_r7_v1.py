"""Four owned paired shards; all outcomes stay unread until actual retirement."""
import hashlib,importlib.util,json,os,sys,signal,inspect,time
import numpy as np
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=O/'lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py'
assert sha(QP)=='8bd0cb93f83ddb22223b8b8f1e8f1017dcb458661a0934637107437a732c15ae'
s=importlib.util.spec_from_file_location('_owned_table_r7',QP);smoke=importlib.util.module_from_spec(s);s.loader.exec_module(smoke)
q=smoke.q;q.guard.JUDGE=D/'native_table_r7_judge_v1.py'
cs=inspect.getsource(q.current);old="assert r['stats']['_GTR_STATS']['calls']>0";assert cs.count(old)==1
exec(compile(cs.replace(old,"assert r['stats']['_GTR_STATS']['calls']==0"),str(__file__)+':private_current','exec'),q.__dict__)
STATS=q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_RTT_STATS','_RTT_ACTIVE','_RTT_CONTEXT','_ASTRA_HF_STATS','_RE22_STATS','_RE22_ACTIVE','_ASTRA_SM40_STATS','_ASTRA_SM40_BIND_STATS','_PCV_STATS')
def checked(c,item,seed,role):
 cc=dict(c,candidates=dict(c['candidates'],ON=c['candidates'][role]));a=q.current(cc,item,seed);st=a['row']['stats'];hf=st['_ASTRA_HF_STATS']
 assert a['row']['hermite_route']==dict(fusion_on=True,extra31_inactive=True)
 assert hf['calls']==28 and hf['ba']==hf['packed']==14 and hf['retired'] is True
 assert hf['layers']==[[l,persistent]for l in range(1,15)for persistent in (True,False)]
 assert hf['elements']==31195136 and a['row']['hermite_cleanup']==dict(row_absent=True)
 assert st['_RTT_ACTIVE'] is False and st['_RTT_CONTEXT']==[None]
 assert st['_YR7_ACTIVE']==[False]and st['_YR7_BIND_STATS']==dict(keys=0,retired=True)
 assert st['_YR7_STATS']['retired']and st['_YR7_STATS']['value_cache_entries']==0
 from table_r7_source_contract_v1 import gate
 gate(a['row']);assert a['row']['sm40_lifecycle_before']['prebind_count']==c['candidates'][role]['prebound']
 return a
def run(phase,shard):
 p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());digest=sha(p)
 assert c['approved_by']=='root'and c['purpose']=='smirnov_table_r7_fixed_paired24'
 assert phase in ('pilot','remaining')and shard in range(4)
 grant=json.loads((D/'ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json').read_text())
 assert grant['approved']and grant['reviewed_by']=='root'and grant['protocol_sha256']==digest and grant['runner_sha256']==sha(__file__)
 assert grant['native_calls_max']==48 and grant['research_slots']==4 and os.getpriority(os.PRIO_PROCESS,0)==0
 assert all(os.environ.get(k)=='1'for k in q.guard.THREADS)
 if phase=='remaining':
  d=json.loads((D/'PILOT_DECISION_v1.json').read_text());assert d['passed']and d['all_actual_retired']and d['candidate_sha256']==c['candidates']['ON']['sha256']
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);q.guard.pins(proto);seeds=c['phase_seeds'][phase][shard::4]
 h=D/f'run_table_r7_{phase}_s{shard}_v1';h.mkdir(exist_ok=False)
 q.write(h/'PLAN_v1.json',dict(phase=phase,shard=shard,controller_pid=os.getpid(),seeds=seeds,protocol_sha256=digest))
 evaluations=[]
 try:
  for seed in seeds:
   pair={}
   for role in (('parent','ON')if seed%2==0 else('ON','parent')):
    start=time.monotonic()
    while True:
     state=q.guard.resources();elapsed=time.monotonic()-start;admitted=state['free_percent']>=60 and state['counted']<7
     record=dict(role=role,seed=seed,controller_pid=os.getpid(),admitted=admitted,elapsed_seconds=elapsed,**state);q.guard.event(h/'admission_wait.jsonl',record)
     if admitted:q.guard.event(h/'admission.jsonl',record);break
     assert elapsed<1800,'Admission expired; no automatic retry';time.sleep(min(15,1800-elapsed))
    env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),GOP2_MODE='1',GOP2_SCIENCE_PROTOCOL_PATH=str(p),GOP2_SCIENCE_PROTOCOL_SHA256=digest,PYTHONDONTWRITEBYTECODE='1')
    item=q.guard.run_native(h,proto,role,seed,env);pair[role]=checked(c,item,seed,role);evaluations.append(dict(role=role,seed=seed,**item))
   a,b=pair['ON'],pair['parent'];assert a['pred'][:1].tobytes()==b['pred'][:1].tobytes()and a['row']['mse_layers'][:1]==b['row']['mse_layers'][:1]
   assert a['row']['weights_before_sha256']==b['row']['weights_before_sha256']and a['cert']['setup']['pb_single_sha256']==b['cert']['setup']['pb_single_sha256']
   print(json.dumps(dict(phase=phase,shard=shard,completed_pairs=len(evaluations)//2,total_pairs=len(seeds))),flush=True)
  q.write(h/'READY_v1.json',dict(controller_pid=os.getpid(),protocol_sha256=digest,evaluations=evaluations,all_guards_retired=q.guard.OWNED_RETIREMENT_CONFIRMED,means_not_read=True))
 except BaseException as e:
  q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,owned_retirement=q.guard.OWNED_RETIREMENT_CONFIRMED));raise
def interrupted(signum,frame):raise KeyboardInterrupt('Owned paired collector signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:run(sys.argv[1],int(sys.argv[2]))
 finally:signal.signal(signal.SIGTERM,old)
