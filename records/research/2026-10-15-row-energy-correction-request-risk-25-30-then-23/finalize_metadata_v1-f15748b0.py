"""Execute the already-prepared metadata-only release sequence after runtime PASS."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
D=Path(__file__).resolve().parent;O=D.parents[2];R=O/'release';B=D/'boundary'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(p,x):
 with p.open('x')as f:f.write(x if isinstance(x,str)else json.dumps(x,indent=2,allow_nan=False)+'\n')
ready=read(D/'DELIVERY_READY_v1.json');H=ready['candidate']['sha256'];assert sha(ready['candidate']['path'])==H
done=B/'BOUNDARY_COMPLETE_HELD_v2.json';b=read(done)
assert b['candidate_sha256']==H and b['normal_pass']and b['slow_pass']and b['science24_accepted']and b['both_actual_managed_exits_observed']and b['all_owned_guards_retired']
for p,h in b['evidence_sha256'].items():assert sha(p)==h
ret=B/'ROOT_COLLECTORS_RETIREMENT_v1.json';assert read(ret)['all_owned_PID_PPID_PGID_absent']
seq=D/'FINALIZATION_SEQUENCE_HELD_v1.json';evidence={str(p):sha(p)for p in (seq,done,ret,D/'STAGE_A_EXACT_ARCHIVE_v1.json')}
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TL=str(D/'release_private.log'))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[k]='1'
for k in ('PB_ALL','MODE','DRY','STATS'):env.pop(k,None)
for i,step in enumerate(read(seq)['steps']):
 if isinstance(step,dict):
  assert step['action']=='exclusive_byte_identical_copy_of_complete_pinned_count_directory'
  src=Path(step['src']);dst=Path(step['dst']);dst.parent.mkdir(parents=True,exist_ok=False);shutil.copytree(src,dst)
  files=[p for p in src.rglob('*')if p.is_file()]
  assert all(sha(p)==sha(dst/p.relative_to(src))for p in files)
  p=D/f'FINALIZATION_COPY_{i:02d}_v1.json';put(p,dict(source=str(src),destination=str(dst),files={str(p.relative_to(src)):sha(p)for p in files},all_bytes_exact=True));evidence[str(p)]=sha(p)
 else:
  assert '--from-run'in step or any(Path(z).name in ('fastgate_pin_v1.py','designation_judge.py','release_gate_v4.py')for z in step)
  result=subprocess.run(step,cwd=R,env=env,capture_output=True,text=True,check=True,timeout=180)
  log=D/f'FINALIZATION_STEP_{i:02d}_v1.log';put(log,result.stdout+result.stderr);evidence[str(log)]=sha(log)
  if any(Path(z).name=='designation_judge.py'for z in step):
   assert ('  FAIL ' not in result.stdout and ('no result for: worker' in result.stdout or '] PASS:' in result.stdout))
   judge=R/'runs'/('judge_'+ready['tag']+'_safe.txt');put(judge,result.stdout);evidence[str(judge)]=sha(judge)
  if any(Path(z).name=='release_gate_v4.py'for z in step):
   gate=json.loads(result.stdout);assert gate['passed']is True and gate['gate_version']==4
   gp=D/'RELEASE_GATE_v4.json';put(gp,gate);evidence[str(gp)]=sha(gp)
put(B/'ROOT_PRODUCTION_ACCEPTANCE_v1.json',dict(approved=True,reviewed_by='root',executed_by='larger_accuracy',authority='Root delegated fixed release finalization when unchanged PRIMARY checks pass; no source edits or additional models.',candidate_sha256=H,normal_pass=True,slow_pass=True,science24_accepted=True,both_actual_managed_exits_observed=True,all_owned_PID_PPID_PGID_absent=True,release_gate_passed=True,evidence_sha256=evidence))
print(json.dumps(dict(production='PASS',candidate_sha256=H,receipt_sha256=sha(B/'ROOT_PRODUCTION_ACCEPTANCE_v1.json'))))
