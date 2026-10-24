import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
PARENT=D/'full_lower_moment_panel_v1/candidate.py'
PARENT_SHA='bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12'
OLD=D/'lazy_mean_innovation_observer_v1'
MODEL=OLD/'terminal.npz'
MODEL_SHA='9aeb9adba6cef9a730dcb850c83daf62e3cad594648081bc72dbc5e5dfbd9fd5'
read=lambda p:json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def verify(p):
    assert sha(p['path'])==p['sha256'],p['path']
    assert Path(p['path']).stat().st_size==p['bytes'],p['path']

def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
    assert sha(PARENT)==PARENT_SHA and sha(MODEL)==MODEL_SHA

def guarded(name):
    g=read(HERE/f'{name}_watch.json')
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,g
    return g

def relative(a,b):
    import numpy as np
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
