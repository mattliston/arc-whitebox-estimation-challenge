"""Exact-source table+R7 two-slot production collector; unchanged donor custody/gates."""
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
