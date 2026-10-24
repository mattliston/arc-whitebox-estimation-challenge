from pathlib import Path
import hashlib,json,os,shutil,subprocess
D=Path(__file__).resolve().parent;O=D.parents[2];R=O/'release';P=D/'official120_v1';B=D/'boundary'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(p,x):
 with p.open('x')as f:f.write(x if isinstance(x,str)else json.dumps(x,indent=2)+'\n')
seq=read(D/'FINALIZATION_SEQUENCE_HELD_v1.json');assert sha(seq['official_wall_policy']['path'])==seq['official_wall_policy']['sha256']
assert all(read(B/('OFFICIAL_'+mode+'_v1')/'early_gate.json')['passed']for mode in ('NORMAL','SLOW'))
assert read(B/'ROOT_COLLECTORS_RETIREMENT_v1.json')['all_owned_PID_PPID_PGID_absent']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[k]='1'
evidence={}
for i,step in enumerate(seq['steps']):
 if isinstance(step,dict):
  assert step['action']=='exclusive_byte_identical_copy_of_complete_pinned_count_directory'
  src=Path(step['src']);dst=Path(step['dst']);dst.parent.mkdir(parents=True,exist_ok=False);shutil.copytree(src,dst)
  files=[x for x in src.rglob('*')if x.is_file()];assert all(sha(x)==sha(dst/x.relative_to(src))for x in files)
  q=D/f'FINALIZATION_COPY_{i:02d}_v1.json';put(q,dict(source=str(src),destination=str(dst),files={str(x.relative_to(src)):sha(x)for x in files},all_bytes_exact=True));evidence[str(q)]=sha(q)
 else:
  assert '--from-run'in step or any(Path(z).name in ('fastgate_pin_v1.py','designation_judge_official120_v1.py','release_gate_official120_v1.py')for z in step)
  r=subprocess.run(step,cwd=R,env=env,capture_output=True,text=True,timeout=180)
  q=D/f'FINALIZATION_STEP_{i:02d}_v1.log';put(q,r.stdout+r.stderr);evidence[str(q)]=sha(q)
  print(i,r.returncode,r.stdout[-300:],r.stderr[-600:],flush=True);assert r.returncode==0
  if any(Path(z).name=='designation_judge_official120_v1.py'for z in step):
   assert '  FAIL 'not in r.stdout and 'no result for: worker'in r.stdout
   q=R/'runs'/('judge_'+read(D/'DELIVERY_READY_v1.json')['tag']+'_official120.txt');put(q,r.stdout);evidence[str(q)]=sha(q)
  if any(Path(z).name=='release_gate_official120_v1.py'for z in step):
   gate=json.loads(r.stdout);assert gate['passed']is True;gp=D/'RELEASE_GATE_OFFICIAL120_v1.json';put(gp,gate);evidence[str(gp)]=sha(gp)
put(D/'OFFICIAL120_METADATA_COMPLETE_v1.json',dict(candidate_sha256=read(D/'DELIVERY_READY_v1.json')['candidate']['sha256'],release_gate_passed=True,official_wall_policy=seq['official_wall_policy'],old100_FAIL_preserved=True,all_other_gates_unchanged=True,native35_100_pending=True,cost_transfer_pending=True,evidence_sha256=evidence))
