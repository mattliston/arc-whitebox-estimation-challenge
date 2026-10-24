import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
SOURCE=D/'full_lower_moment_panel_v1/candidate.py'
SOURCE_SHA='bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
TRAIN=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
CHECK=[37,39,42,48,59,63,66,67]
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TARGET_MANIFEST=P2/'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def write(p,value):
    with Path(p).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)

def verify(p):
    assert sha(p['path'])==p['sha256'],p['path']
    assert Path(p['path']).stat().st_size==p['bytes']

def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
    assert sha(SOURCE)==SOURCE_SHA

def weights_path(rank):
    rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
