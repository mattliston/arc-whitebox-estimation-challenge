from pathlib import Path
import hashlib,json,ast
D=Path(__file__).resolve().parent;R=D.parent;O=D.parents[2];B=D/'boundary';B.mkdir()
S=R/'astra_smirnov_table_r7_science_v1';OLD=R/'astra_smirnov_rpe_delivery_v1';source=R/'astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py';parent=R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py';grant=S/'ROOT_SCIENCE_ACCEPTANCE_RELEASE_GRANT_v1.json';donor=R/'astra_gop2_current_boundary_v1/finite_gop2_current_boundary_HELD_v3.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x')as f:f.write(v if isinstance(v,str)else json.dumps(v,indent=2)+'\n')
assert sha(source)=='b7090c35a185309c73633d66dbaa06d48b2d173c3f6b25ff941a54c224b2ba04'
code='''"""Exact-source table+R7 two-slot production collector; unchanged donor custody/gates."""
import hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
PROTOCOL=D/'BOUNDARY_PROTOCOL_REGISTERED_v1.json'
def pins():
 p=read(PROTOCOL)
 assert p['status']=='REGISTERED' and p['purpose']=='TABLE_R7_EXACT_SOURCE_PRODUCTION'
 assert p['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 for path,h in p['source_pins'].items():assert sha(path)==h,('changedpin',path)
 return p
def qualified(g):
 p=pins();assert g['approved']is True and g['reviewed_by']=='root'and g['candidate_sha256']==p['candidate_sha256']
 assert sha(g['science'])==g['science_sha256']and sha(g['retirement'])==g['retirement_sha256']
 d=read(g['science']);r=read(g['retirement'])
 assert d['passed']is True and d['n']==24 and d['candidate_sha256']==p['candidate_sha256']and d['all_actual_retired']is True
 assert d['means']['net_log_pct']<0 and d['OOS8']['net_log_pct']<0
 assert r['all_actual_retired']is True and r['all_actual_exits_observed']is True and r['all_owned_PID_PPID_PGID_absent']is True and r['all_guards_clean']is True
 assert d['retirement_sha256']==sha(g['retirement'])
 return dict(science24_root_acceptance_path=p['root_grant'],science24_root_acceptance_sha256=p['root_grant_sha256'],scope='Direct same-b709 scientific24/OOS8; no transfer/parity-to-parent or duplicated24. Fresh production and same-source captured-setup bill/bytes parity remain required.')
def config(path):
 p=pins();assert Path(path).resolve()==Path(p['root_grant'])and sha(path)==p['root_grant_sha256']
 g=read(path);assert os.getpriority(os.PRIO_PROCESS,0)==0
 assert all(os.environ.get(k)=='1'for k in p['threads'])
 assert all(os.environ.get(k)is None for k in('PB_ALL','MODE','DRY','STATS'))
 g=dict(g,final_source_identity_root_sha256=sha(path))
 return p,g,qualified(g)
def main():
 p=pins();donor=Path(p['donor']);spec=importlib.util.spec_from_file_location('table_r7_boundary_donor',donor);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.__file__=str(Path(__file__).resolve());m.D=D;m.N=D/'numerical_v1';m.PROV=Path(p['candidate']);m.PARENT=Path(p['parent'])
 def configured(path):
  c,g,q=config(path);m.ACTIVE_PROTOCOL=PROTOCOL;return c,g,q
 m.pins=pins;m.prior_census=qualified;m.grant=configured
 for name in('admit','admissions','numerical'):
  code=inspect.getsource(getattr(m,name)).replace("['counted']<10","['counted']<7").replace("['counted'] <= 10","['counted'] <= 7")
  exec(compile(code,str(__file__)+':cap7','exec'),m.__dict__)
 old=m.exclusive
 def honest(path,data):
  if Path(path).name=='BOUNDARY_COMPLETE_HELD_v2.json':data.pop('own20_accepted',None);data['science24_accepted']=True
  return old(path,data)
 m.exclusive=honest
 m.main()
if __name__=='__main__':main()
'''
controller=B/'finite_table_r7_boundary_v1.py';compile(code,str(controller),'exec');put(controller,code)
early=(OLD/'official120_v1/early_count_gate_official120_v1.py').read_text();put(B/'early_count_gate_v1.py',early)
oldp=json.loads((OLD/'boundary/BOUNDARY_PROTOCOL_REGISTERED_v1.json').read_text());pins={p:h for p,h in oldp['source_pins'].items()if not Path(p).is_relative_to(OLD)}
paths=[source,parent,grant,donor,controller,B/'early_count_gate_v1.py',S/'FINAL24_DECISION_v1.json',S/'REMAINING_RETIREMENT_v1.json',R/'astra_smirnov_unseen_shapes_v1/LARGER_ACCURACY_STACK_PEER_v1.json',R/'astra_smirnov_unseen_shapes_v1/PREFLIGHT_TABLE_R7_v2.json',O/'lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json',OLD/'official120_v1/designation_judge_official120_v1.py',Path(__file__)]
pins.update({str(p):sha(p)for p in paths})
for p,h in pins.items():assert sha(p)==h
protocol=dict(status='REGISTERED',purpose='TABLE_R7_EXACT_SOURCE_PRODUCTION',candidate=str(source),candidate_sha256=sha(source),parent=str(parent),parent_sha256=sha(parent),root_grant=str(grant),root_grant_sha256=sha(grant),donor=str(donor),source_pins=pins,threads=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'],fixed_modes=['normal','slow'],ranks=[35,100],slots=2,logicalcap=7,entry_memory_free_pct=60,running_memory_free_pct=50,guard_bytes=7500000000,official_wall_hard=120,advisory_wall=100,all_other_gates_unchanged=True,no_SAFE=True,no_fire_authority=True,no_duplicate_science=True)
put(B/'BOUNDARY_PROTOCOL_REGISTERED_v1.json',protocol)
ready=dict(status='ROOT_AUTHORIZED_READY_TO_COLLECT',candidate=dict(path=str(source),sha256=sha(source)),parent=dict(path=str(parent),sha256=sha(parent)),science_parent=dict(path=str(source),sha256=sha(source)),tag='table_r7_20261016_astra_v1',controller=dict(path=str(controller),sha256=sha(controller)),protocol=dict(path=str(B/'BOUNDARY_PROTOCOL_REGISTERED_v1.json'),sha256=sha(B/'BOUNDARY_PROTOCOL_REGISTERED_v1.json')),grant=dict(path=str(grant),sha256=sha(grant)),no_fire_authority=True,no_SAFE=True,models=0)
put(D/'DELIVERY_READY_v1.json',ready);print(json.dumps(ready,indent=2))
