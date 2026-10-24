import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2)
def seal_check():
    s=read(HERE/'sources_sealed.json')
    for r in s['pins']:assert sha(r['path'])==r['sha256'],r['path']
    return s
def guarded(name):
    g=read(HERE/(name+'_watch.json'))
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000
    return g

