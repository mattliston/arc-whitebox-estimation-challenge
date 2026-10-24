from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
D19=P2/'codex_resume_20260919_v1';OLD=D19/'incoming_third_pair_head_v1';JOINT=D19/'joint_incoming_moment_head_v1'
ASTRA=P2/'astra_phase2_reexamination_v1';PROJECTION=H2/'fourth_matrix_face_capacity_v1'
TRAIN=(57,68,45,59,3,12);LAYERS=tuple(range(8,14))
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
checked=set()
def check(r):
    k=(r['path'],r['sha256'],r['bytes'])
    if k not in checked:
        assert pin(r['path'])=={k:r[k] for k in ('path','sha256','bytes')};checked.add(k)
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify():
    seal=read(HERE/'seal.json')
    for p in seal['sources']:check(p)
    return seal
