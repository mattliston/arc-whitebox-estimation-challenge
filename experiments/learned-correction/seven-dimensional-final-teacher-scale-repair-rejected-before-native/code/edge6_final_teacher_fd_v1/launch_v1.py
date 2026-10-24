"""NumPy-free import validation then explicitly requested frozen FD dispatch."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace')
RUNS=Path('/home/arcworker/arc-runs')
PORT=HERE.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def check_imports():
    assert sys.platform=='linux' and importlib.util.find_spec('numpy') is None
    b=json.loads((HERE/'BUILD_v1.json').read_text());plan=json.loads((HERE/'PLAN_v1.json').read_text())
    assert sha(HERE/'PLAN_v1.json')==b['plan_sha256']
    with tempfile.TemporaryDirectory(prefix='arc-edge6-fd-import-') as td:
        for rec in b['records']:
            assert plan['candidates'][rec['name']]['effective_sha256']==rec['effective_sha256']
            parent=ROOT/rec['path'];tail=ROOT/rec['append_tail']['path']
            assert sha(parent)==rec['sha256'] and sha(tail)==rec['append_tail']['sha256']
            code=parent.read_bytes()+b'\n'+tail.read_bytes()
            assert hashlib.sha256(code).hexdigest()==rec['effective_sha256']
            path=Path(td)/(rec['name']+'.py');path.write_bytes(code)
            spec=importlib.util.spec_from_file_location(rec['name'],path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            assert module._GN_EDGE=={int(k):v for k,v in rec['scales'].items()}
            assert module._O20_EDGE_HIDDEN==6 and module._VMFR_CM==8
            assert not module._FAST_RESCUE_FRAC and 'numpy' not in sys.modules
    write(HERE/'NUMPY_FREE_IMPORT_v1.json',dict(utc=utc(),plan_sha256=sha(HERE/'PLAN_v1.json'),
        all14_static_variants_pass=True,no_numpy=True,source_pins_verified=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-imports',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.check_imports:return check_imports()
    assert sys.platform=='linux' and HERE.is_relative_to(ROOT)
    build=json.loads((HERE/'BUILD_v1.json').read_text());assert sha(HERE/'PLAN_v1.json')==build['plan_sha256']
    pins=json.loads((PORT/'BUILD_v1.json').read_text())
    for name,digest in pins['pins'].items():assert sha(PORT/name)==digest,name
    for name,digest in pins['original_v7_pins'].items():assert sha(PORT.parent/'cloud_linux_capacity_v7'/name)==digest,name
    import_receipt=HERE/'NUMPY_FREE_IMPORT_v1.json'
    if not import_receipt.exists():
        subprocess.run(['/home/arcworker/arc-envs/client/bin/python','-B',str(Path(__file__).resolve()),'--check-imports'],check=True)
    assert json.loads(import_receipt.read_text())['plan_sha256']==build['plan_sha256']
    if not a.run:
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_NATIVE',plan_sha256=build['plan_sha256'])));return
    assert not (HERE/'LAUNCH_v1.json').exists()
    for rec in build['records']:
        assert not (RUNS/rec['name']).exists() and not (RUNS/(rec['name']+'_controller.log')).exists()
    launches=[]
    for rec in build['records']:
        cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(PORT/'run_science_linux_markline_v1.py'),
          '--root',str(ROOT),'--plan',str(HERE/'PLAN_v1.json'),'--candidate',rec['name'],'--out',str(RUNS/rec['name']),
          '--client-python','/home/arcworker/arc-envs/client/bin/python','--server-python','/home/arcworker/arc-envs/server/bin/python',
          '--state-dir',str(RUNS/'admission'),'--jobs','2','--max-active','36','--blas-threads','4',
          '--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--setup','paced']
        with (RUNS/(rec['name']+'_controller.log')).open('x') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        record=dict(utc=utc(),name=rec['name'],pid=child.pid,command=cmd,source_sha256=rec['effective_sha256'],
            plan_sha256=build['plan_sha256'],original_guard_unchanged=True,shared36=True)
        write(HERE/(rec['name']+'_LAUNCH.json'),record);launches.append(record)
    write(HERE/'LAUNCH_v1.json',dict(utc=utc(),launches=launches,new_training_predictions=56,
        no_truth=True,no_auto_fit=True,no_auto_fire=True,plan_sha256=build['plan_sha256']))
    print(json.dumps(dict(utc=utc(),pids=[r['pid'] for r in launches],arms=14,new_predictions=56)))

if __name__=='__main__':main()
