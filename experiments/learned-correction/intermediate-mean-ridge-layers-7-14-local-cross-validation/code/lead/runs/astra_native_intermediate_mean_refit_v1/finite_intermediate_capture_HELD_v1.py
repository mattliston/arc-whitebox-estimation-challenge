"""Four fixed native TRAIN capture shards; no fitting or selection in collectors."""
import argparse,hashlib,importlib.util,json,os,signal,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DONOR=O/'lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py'
s=importlib.util.spec_from_file_location('_nlr_lifecycle',DONOR);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
guard=q.guard;write=q.write;guard.JUDGE=D/'native_intermediate_capture_HELD_v1.py'

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--run-authorized',action='store_true');a=p.parse_args()
 assert a.run_authorized and 0<=a.shard<4 and sha(a.protocol)==a.protocol_sha256
 c=json.loads(a.protocol.read_text());g=json.loads(a.grant.read_text())
 assert c['purpose']=='qr_native_intermediate_local_ridge_train32' and c['train_ranks']==list(range(32)) and c['new_native_calls_max']==32
 assert g['approved'] is True and g['reviewed_by']=='root' and g['protocol_sha256']==a.protocol_sha256 and g['numerical_slots']==4
 assert g['runner_sha256']==sha(__file__) and g['new_native_calls_max']==32
 assert all(os.environ.get(k)=='1' for k in guard.THREADS) and os.getpriority(os.PRIO_PROCESS,0)==0
 guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 ranks=c['shards'][str(a.shard)];assert ranks==list(range(a.shard,32,4))
 run=D/f'run_capture_s{a.shard}_v1';run.mkdir(exist_ok=False)
 write(run/'PLAN_v1.json',dict(controller_pid=os.getpid(),shard=a.shard,protocol_path=str(a.protocol),protocol_sha256=a.protocol_sha256,grant_path=str(a.grant),grant_sha256=sha(a.grant),runner_sha256=sha(__file__),fixed_order=[['capture',r]for r in ranks]))
 evaluations=[]
 try:
  for rank in ranks:
   q.admit(run,'capture',rank)
   env=dict(os.environ,PB_ALL='1',GOP2_MODE='1',STATS=','.join(q.q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_RTT_STATS','_RTP_ON')),PYTHONDONTWRITEBYTECODE='1')
   item=guard.run_native(run,dict(pinned_files=c['source_pins'],candidates=c['candidates']),'capture',rank,env);item.update(role='capture',seed=rank)
   cert=json.loads(Path(item['certificate']).read_text());record=c['network_metadata'][rank]
   assert cert['capture_complete'] and cert['intermediate94_complete'] and cert['all16_finite'] and cert['layers']==list(range(7,15))
   assert cert['weights_before_sha256']==cert['weights_after_sha256']==record['weights_raw_sha256']
   assert cert['observational_FNP_calls']==cert['observational_flops']==0
   assert cert['all16_parity_measured'] and cert['all16_byte_identical'] and cert['flops_identical']
   item.update(capture_path=cert['capture_path'],capture_sha256=cert['capture_sha256']);assert sha(item['capture_path'])==item['capture_sha256']
   evaluations.append(item)
   print(json.dumps(dict(shard=a.shard,completed=len(evaluations),total=len(ranks),rank=rank)),flush=True)
  write(run/'CAPTURE_SHARD_READY_v1.json',dict(shard=a.shard,complete_calls=len(evaluations),evaluations=evaluations,all_guards_clean=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_fit_executed=True))
 except BaseException as e:
  write(run/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED));raise

def interrupted(signum,frame):raise KeyboardInterrupt('Owned capture signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
