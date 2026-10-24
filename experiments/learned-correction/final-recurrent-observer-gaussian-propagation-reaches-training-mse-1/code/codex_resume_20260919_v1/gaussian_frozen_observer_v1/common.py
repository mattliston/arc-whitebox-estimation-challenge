"""Stdlib metadata; numerical code imports are deferred to guarded workers."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
GAUSSIAN=A/'gaussian_terminal_skew_v2'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRAIN=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
CHECK=[37,39,42,48]
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def verify(name):
    s=read(HERE/name)
    for row in s['pins']:assert sha(row['path'])==row['sha256'],row['path']
    return s
def wp(rank):
    rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
def enc4_direction(np):
    return (np.sin(.17*np.arange(1,65,dtype=np.float64))[:,None]*np.cos(.11*np.arange(1,129,dtype=np.float64))[None,:]/np.sqrt(64*128)).astype(np.float32)
