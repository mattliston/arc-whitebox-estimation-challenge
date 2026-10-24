"""Freeze full-row target-free TRAIN4 native capture and run adapter preflight."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parent.parent/'astra_oct20_vm_frontier_v1'
PORT=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0,str(PORT));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def rec(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with p.open('x') as out:json.dump(x,out,indent=2);out.write('\n')
base=V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
assert sha(base)=='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
sci=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
assert sha(sci)=='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
teacher=dict(rec(sci),effective_sha256=sha(sci))
config=RUNS/'full_state_response_train4_config_v1';config.mkdir(exist_ok=False)
names=('client_capture_v1.py','run_group_capture_v1.py','run_capture_v1.py','fragment_mark_complete_line_v1.py')
for name in names+('full_state_hook_v1.py',):ast.parse((HERE/name).read_text())
recipe=dict(utc=utc(),kind='CURRENT777A_FULL_STATE_TRAIN4',teacher=teacher,
 capture_ranks=[35,150,200,250],train_ranks=[35,150,200,250],validation_ranks=[],
 layers=list(range(1,16)),rows_per_layer=1024,target_free=True,uses_network_truth=False,
 hook=rec(HERE/'full_state_hook_v1.py'),harness=[rec(HERE/name) for name in names],
 prereg=rec(HERE/'PREREG_v1.json'),diagnostic_only=True,guard_unchanged=True,reference_submission=337422)
plan=json.loads(base.read_text());assert plan['groups'][:2]==[[35,150,200],[250,259]]
plan.update(utc=utc(),kind=recipe['kind'],diagnostic_only=True,groups=[[35,150,200],[250]],
 first7=[35,150,200,250],held25=[],rows={str(r):{'weights':plan['rows'][str(r)]['weights']} for r in (35,150,200,250)},candidates={'teacher':teacher})
write(config/'PLAN_v1.json',plan);write(config/'RECIPE_v1.json',recipe)
env=os.environ.copy();env.update(thread_env(1));env['ARC_BLAS_THREADS']='1'
env['ARC_FULL_STATE_RECIPE']=str(config/'RECIPE_v1.json');env['ARC_FULL_STATE_RECIPE_SHA']=sha(config/'RECIPE_v1.json')
client='/home/arcworker/arc-envs/client/bin/python'
check=subprocess.run([client,'-B',str(HERE/'client_capture_v1.py'),'--help'],env=env,capture_output=True,text=True)
assert check.returncode==0,check.stderr
subprocess.run([client,'-c',"import importlib.util;assert importlib.util.find_spec('numpy') is None"],check=True,env=env)
result=dict(utc=utc(),successful=True,recipe=str(config/'RECIPE_v1.json'),recipe_sha256=sha(config/'RECIPE_v1.json'),
 plan=str(config/'PLAN_v1.json'),plan_sha256=sha(config/'PLAN_v1.json'),hook_sha256=sha(HERE/'full_state_hook_v1.py'),
 source_sha256=sha(sci),no_truth=True,client_preflight='PASS',no_native_launch=True)
write(HERE/'CAPTURE_READY_v1.json',result);print(json.dumps(result))

