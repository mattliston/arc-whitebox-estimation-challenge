import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
PARENT=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
read=lambda f:json.loads(Path(f).read_text())
def write(f,obj):
    with Path(f).open('x') as h:json.dump(obj,h,indent=2)
def seal_check():
    s=read(HERE/'sources_sealed.json')
    for r in s['pins']:assert sha(r['path'])==r['sha256'],r['path']
    return s
def guarded(name):
    g=read(HERE/(name+'_watch.json'))
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,name
    return g
