"""Metadata helpers. Numerical work is launched only under run_guarded.py."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
A=P2/'astra_phase2_reexamination_v1'
PARENT=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
TRAIN=[150,200,250,300,400,500]
CHECK=[35,100]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def verify(name='sources_sealed.json'):
    s=read(HERE/name)
    for x in s['pins']:assert sha(x['path'])==x['sha256'],x['path']
    return s
