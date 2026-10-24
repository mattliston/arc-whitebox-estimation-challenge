"""One fixed two-seed repaired seal arm per shard; no parent rerun, grade, retry or followup."""
from pathlib import Path
import json,hashlib,importlib.util,os,signal,argparse
D=Path(__file__).resolve().parent
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');OLD=O/'lead/runs/astra_rtt2_current_seal_v1'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
DONOR=OLD/'finite_seal_four_partition_private_v2.py'
assert H(DONOR)=='e0ef84d9821c4c506ee8cdcc1ad51c31701aed54250c542b7b70c2e501064484'
s=importlib.util.spec_from_file_location('_repair_arm_donor',DONOR);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.__file__=str(Path(__file__).resolve());m.D=D;m.O=O;m.R=O/'release';m.SOURCE=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_request_lean_v1/candidate_hermite_group_lean_QR_HELD_v2.py');m.SOURCE_SHA='19fc42a4d64b6721a91719f87cd4e81eb3b6f6b54aa24aed92ae4a5facdd8f5f'
m.BASE=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_rtt2_request_repair_source_v1/candidate_rtt2_request_repair_HELD_v2.py');m.BASE_SHA='e406e64b67df590b61f887d2abfb481c7937df4ee069f230b61bed69b324b81c'
m.FIRST=list(range(7190080,7190088));m.EXTRA=list(range(7190088,7190096));m.JUDGE=OLD/'run_cert_private_truth_HELD_v1.py'
m.STATS=m.STATS+('_RTT_STATS','_RTT_ACTIVE','_RTT_CONTEXT','_ASTRA_HF_STATS','_GTR_STATS','_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS');m.ENV['STATS']=','.join(m.STATS)
def pins():
 g=read(D/'ROOT_SEAL_ARM_GRANT_v1.json')
 assert g['approved']is True and g['reviewed_by']=='root'and g['source_sha256']==m.SOURCE_SHA
 assert g['fixed_shards']==[m.seeds_for('initial8',i)for i in range(4)]
 assert g['numerical_slots']==4 and g['logical_reserved_max']==7 and g['no_extension_or_retry']is True
 for p,h in g['source_pins'].items():assert H(p)==h,('pin',p)
 for e in g['original_phase_retirements'].values():
  assert H(e['path'])==e['sha256'];r=read(e['path']);assert r['approved']is True and r['all_owned_PID_PPID_PGID_absent']is True
 return g
m.pins=pins
import inspect
for name in ('admit','admissions','numerical'):
 code=inspect.getsource(getattr(m,name)).replace("['counted'] < 10","['counted'] < 7").replace("['counted'] <= 10","['counted'] <= 7").replace("['counted']<10","['counted']<7").replace("['counted']<=10","['counted']<=7")
 exec(compile(code,str(__file__)+':cap7','exec'),m.__dict__)
old_checked=m.checked
def checked(phase,role,shard):
 rows,e,owner=old_checked(phase,role,shard)
 if role=='arm':
  for row in rows:
   st=row['stats'];hf=st['_ASTRA_HF_STATS']
   assert hf['calls']==28 and hf['ba']==hf['packed']==14 and hf['retired']is True and hf['elements']==31195136
   assert hf['layers']==[[l,p]for l in range(1,15)for p in (True,False)]
   assert st['_GTR_STATS']['calls']==0 and st['_RTT_ACTIVE']is False and st['_RTT_CONTEXT']==[None]
   assert st['_YR7_ACTIVE']==[False]and st['_YR7_BIND_STATS']==dict(keys=17,retired=True)
   assert st['_YR7_STATS']['retired']and st['_YR7_STATS']['value_cache_entries']==0
 return rows,e,owner
m.checked=checked
def run(shard):
 g=pins();m.set_phase('initial8',shard);key=m.phasekey('initial8',shard);step='arm_'+key;m.STEPS=[step]
 assert os.getpriority(os.PRIO_PROCESS,0)==0 and all(os.environ.get(k)=='1'for k in m.THREADS)
 assert all(os.environ.get(k)is None for k in('PB_ALL','MODE','DRY','STATS'))
 assert not m.N.exists();m.N.mkdir(parents=True);lock=D/f'OWNER_LOCK_{key}_v1';lock.mkdir();error=None
 m.exclusive(D/f'PHASE_OWNER_{key}_v1.json',dict(controller_pid=os.getpid(),controller_sha256=H(__file__),grant_sha256=H(D/'ROOT_SEAL_ARM_GRANT_v1.json')))
 try:
  cmd=m.command_for('initial8','arm',shard);guard=m.numerical(step,cmd,7320);out=m.N/step/'cert.json'
  m.finish(step,[out,guard,out.with_name('cert_pred.npz')],dict(seeds=m.seeds_for('initial8',shard),role='arm'))
  rows,e,owner=m.checked('initial8','arm',shard);assert len(rows)==2
  m.exclusive(D/f'PHASE_READY_{key}_v1.json',dict(controller_pid=os.getpid(),source_sha256=m.SOURCE_SHA,controller_sha256=H(__file__),grant_sha256=H(D/'ROOT_SEAL_ARM_GRANT_v1.json'),seeds=m.seeds_for('initial8',shard),evidence_sha256=e,admission_evidence=m.admissions(),expected_guards=[str(guard)],no_means_computed=True))
 except BaseException as e:
  error=e;m.exclusive(D/f'PHASE_STOPPED_{key}_v1.json',dict(error=repr(e),owned_retirement_verified=m.RETIRED,no_retry=True));raise
 finally:
  m.exclusive(D/f'PHASE_LIFECYCLE_RETIREMENT_{key}_v1.json',dict(controller_pid=os.getpid(),error=repr(error)if error else None,all_owned_guards_retired=m.RETIRED,slot_reservation_retained=not m.RETIRED))
  if m.RETIRED:lock.rmdir()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('shard',type=int,choices=range(4));a=ap.parse_args()
 previous=signal.signal(signal.SIGTERM,m.interrupted)
 try:run(a.shard)
 finally:signal.signal(signal.SIGTERM,previous)
