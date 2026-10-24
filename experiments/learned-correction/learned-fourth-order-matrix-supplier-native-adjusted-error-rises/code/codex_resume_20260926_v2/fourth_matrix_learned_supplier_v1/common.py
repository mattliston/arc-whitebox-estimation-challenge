from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
D19=P2/'codex_resume_20260919_v1';OLD=D19/'incoming_third_pair_head_v1'
JOINT=D19/'joint_incoming_moment_head_v1';ASTRA=P2/'astra_phase2_reexamination_v1'
PROJECTION=H2/'fourth_matrix_face_capacity_v1';CAPACITY=H2/'fourth_matrix_generalization_v1'
TRAIN=(57,68,45,59,3,12);VALID=(14,71);LAYERS=tuple(range(8,15))
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
_checked=set()
def check(p):
    key=(p['path'],p['sha256'],p['bytes'])
    if key not in _checked:
        assert pin(p['path'])=={k:p[k] for k in ('path','sha256','bytes')},p['path']
        _checked.add(key)
def once(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verify():
    s=read(HERE/'seal.json')
    for p in s['sources']:check(p)
    assert read(CAPACITY/'result.json')['passed'] and read(CAPACITY/'verification.json')['passed']
    g=read(CAPACITY/'run_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    return s
