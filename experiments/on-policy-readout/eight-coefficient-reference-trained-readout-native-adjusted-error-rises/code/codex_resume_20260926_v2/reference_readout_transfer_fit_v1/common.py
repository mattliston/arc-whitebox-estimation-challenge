from pathlib import Path
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parent;P2=H.parent;ROOT=P2.parents[3]
REFERENCE=H/'stable_training_reference_v1';BASIS=H/'native_innovation_readout_response_v1'
FIT=H/'native_covariance_innovation_fit_v1';BASE=H/'recurrent_covariance_observer_v1'
OLD=P2/'codex_resume_20260919_v1/incoming_third_pair_head_v1'
TRAIN=(57,68,45,59,3,12);CHECK=(14,71);NAMES=('zero',)+tuple(f'd{d}_{s}' for d in range(8) for s in ('plus','minus'))
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify():
    for e in read(HERE/'capture_seal.json')['sources']:assert pin(e['path'])==e
    assert read(REFERENCE/'result.json')['passed'] and read(REFERENCE/'verification.json')['passed']
def packet(e,keys=None):
    assert pin(e['path'])==e
    with np.load(e['path']) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def guard(name):
    g=read(HERE/(name+'_guard.json'))
    assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    return g
