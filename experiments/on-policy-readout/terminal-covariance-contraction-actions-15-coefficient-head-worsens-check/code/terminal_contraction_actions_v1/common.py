from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
OLD=H2/'terminal_pair_objective_v1'
TRAIN=(35,100,36,45,52,58,64,70,38,40,41,43)
CHECK=(44,46,47,49,50,51,53,54)
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
def verify():
    for p in read(HERE/'seal.json')['sources']:assert pin(p['path'])==p
def packet(rank):return OLD/('capture_exact_00' if rank in TRAIN[:2] else 'capture_exact_01')/f'row_{rank:04d}.npz'
