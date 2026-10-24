import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
TRAIN=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
CHECK=[37,39,42,48]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def seal_check(name='capture_sources_sealed.json'):
    s=read(HERE/name)
    for row in s['pins']:assert sha(row['path'])==row['sha256'],row['path']
    return s
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2)
def enc4_direction(np):
    return (np.sin(.17*np.arange(1,65,dtype=np.float64))[:,None]
            *np.cos(.11*np.arange(1,129,dtype=np.float64))[None,:]/np.sqrt(64*128)).astype(np.float32)
