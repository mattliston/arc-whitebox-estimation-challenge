import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
OLD=D/'covariance_head_native_response_v1'
DIAG=D/'shared_sketch_roundoff_diagnostic_v1'
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NATIVE=D/'weight5_reuse_v1/native.py'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def verify(p):assert pin(p['path'])==p,p['path']
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
def guarded(label,seconds,args):
    print(dict(starting=label),flush=True)
    cmd=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{label}_watch.json'),
         '--seconds',str(seconds),*map(str,args)]
    with (HERE/f'{label}.log').open('x') as f:r=subprocess.run(cmd,cwd=D,stdout=f,stderr=subprocess.STDOUT)
    g=read(HERE/f'{label}_watch.json')
    assert r.returncode==0 and g['completed'] and g['whole_group_retired']
    assert g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000
    print(dict(completed=label,elapsed_s=g['elapsed_s'],peak_rss_bytes=g['max_rss_bytes']),flush=True)
    return g
