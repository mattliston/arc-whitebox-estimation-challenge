from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
OLDPAIR=H2/'terminal_pair_objective_v1'
OLD=P2/'codex_resume_20260919_v1/incoming_third_pair_head_v1'
PARENT=H2/'candidate_old_both_tail_trace_bounded_v6.py'
PARENT_SHA='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
TRAIN=(57,68,45,59,3,12);CHECK=(14,71);ALL=TRAIN+CHECK
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(seal='seal.json'):
    assert sha(PARENT)==PARENT_SHA
    for e in read(HERE/seal)['sources']:assert pin(e['path'])==e
def weights(mini):
    r=read(OLD/f'capture_mini{mini:04d}.json');assert r['passed'] and r['ordinal']==mini
    e=r['weights'];assert pin(e['path'])==e
    return Path(e['path']),r['provenance']['identity']
