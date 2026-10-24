import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRAIN=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def seal_check():
    s=read(HERE/'sources_sealed.json')
    for x in s['sources']+s['external_pins']:assert sha(x['path'])==x['sha256'],x['path']
    return s
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)
