from pathlib import Path
import hashlib, json

HERE=Path(__file__).resolve().parent
H2=HERE.parent
P2=H2.parent
PARENT=H2/'candidate_thin_star_route_v1.py'
PARENT_SHA='25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
TRAIN=(38,40,41,43,44,46,47,49,50,51,53,54)
CHECK=()
SCREEN=(35,100,36,45,52,58,64,70)
PARITY=(35,100)
read=lambda p:json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def once(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def verify(sealname='seal.json'):
    assert sha(PARENT)==PARENT_SHA
    assert not set(TRAIN)&set(SCREEN)
    for x in read(HERE/sealname)['sources']:
        assert pin(x['path'])==x,x['path']

def packet(rank):
    assert rank in TRAIN
    return HERE/'capture'/f'row_{rank:04d}.npz'
