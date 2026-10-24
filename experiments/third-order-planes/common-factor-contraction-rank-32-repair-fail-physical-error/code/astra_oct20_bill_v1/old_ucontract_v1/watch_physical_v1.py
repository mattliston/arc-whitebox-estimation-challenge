"""Wait for both original guards, then one separately admitted offline analysis."""
import hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs');PORT=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PORT/'guard_linux_v7.py')=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
capture=RUNS/'current_old_su_capture_v1';deadline=time.monotonic()+1800
while not(capture/'COMPLETE.json').exists():
    assert time.monotonic()<deadline,'capture completion deadline';time.sleep(2)
complete=json.loads((capture/'COMPLETE.json').read_text());assert complete['successful']and not complete['pending']
for part in('part01','part02'):
    p=capture/(part+'_GUARD.json');r=json.loads(p.read_text());assert r['successful']and r['whole_group_retired']and r['child_exit']==0
env=os.environ.copy()
for name in('ARC_BLAS_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
tag='old_su_physical_diagnostic_v1'
command=['/home/arcworker/arc-envs/client/bin/python',str(PORT/'guard_linux_v7.py'),'--receipt',str(RUNS/(tag+'_GUARD.json')),'--state-dir',str(RUNS/'admission'),'--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--','/home/arcworker/arc-envs/server/bin/python',str(HERE/'analyze_physical_v1.py'),'--capture',str(capture),'--rank35',str(RUNS/'current_old_dv_capture_v1'),'--rank150',str(RUNS/'current_old_dv_capture_rank150_v2'),'--control',str(RUNS/'msg8_loaderfix337422_control32_markline_blas4_v1'),'--out',str(RUNS/tag)]
assert not(RUNS/tag).exists()and not(RUNS/(tag+'_GUARD.json')).exists()
result=subprocess.run(command,env=env);assert result.returncode==0
guard=json.loads((RUNS/(tag+'_GUARD.json')).read_text());assert guard['successful']and guard['whole_group_retired']
print('PHYSICAL_ANALYSIS_COMPLETE',flush=True)
