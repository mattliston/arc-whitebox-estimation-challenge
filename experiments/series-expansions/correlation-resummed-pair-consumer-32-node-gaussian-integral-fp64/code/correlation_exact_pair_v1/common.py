import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
PARENT=D/'lazy_rotation_objective_v1/candidate.py'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def verify(name='sources_sealed.json'):
    s=read(HERE/name)
    for x in s['pins']:assert sha(x['path'])==x['sha256'],x['path']
    return s
def weights_path(rank):
    rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
