"""One guarded NumPy-free full import qualification; no whole-network native."""
import hashlib,json,os,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=Path('/home/arcworker/arc-workspace');RUN=Path('/home/arcworker/arc-runs');O=H.parents[3];A=H.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1';G=A/'cloud_linux_capacity_v7/guard_linux_v7.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=json.loads((H/'IMPORT_RECIPE_v1.json').read_text())
for r in x['files']:assert sha(R/r['path'])==r['sha256']
assert not(H/'IMPORT_CHECK_v1.json').exists() and not(H/'IMPORT_GUARD_v1.json').exists()
env=dict(os.environ)
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:env[k]='1'
env['ARC_BLAS_THREADS']='1'
command=['/home/arcworker/arc-envs/judge/bin/python','-B',str(G),'--receipt',str(H/'IMPORT_GUARD_v1.json'),'--state-dir',str(RUN/'admission'),'--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--','/home/arcworker/arc-envs/client/bin/python','-B',str(H/'check_import_v1.py')]
with(H/'IMPORT_GUARD_v1.log').open('x')as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
g=json.loads((H/'IMPORT_GUARD_v1.json').read_text());assert p.returncode==0 and g['reason'] is None and g['child_exit']==0 and g['whole_group_retired'] and not g['cleanup_errors'] and not g['max_owned_swap_bytes'];assert json.loads((H/'IMPORT_CHECK_v1.json').read_text())['successful'];print('IMPORT_ONLY_COMPLETE')
