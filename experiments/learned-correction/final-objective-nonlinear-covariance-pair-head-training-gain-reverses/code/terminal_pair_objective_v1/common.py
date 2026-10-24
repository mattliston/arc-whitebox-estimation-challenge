from pathlib import Path
import hashlib,json,ast
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
PARENT=H2/'candidate_old_both_tail_trace_bounded_v6.py'
PARENT_SHA='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
TRAIN=(35,100,36,45,52,58,64,70,38,40,41,43)
CHECK=(44,46,47,49,50,51,53,54)
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def once(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify():
    assert sha(PARENT)==PARENT_SHA
    for x in read(HERE/'seal.json')['sources']:
        assert pin(x['path'])==x
def source_functions(*names):
    s=PARENT.read_text();d={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    return '\n\n'.join(ast.get_source_segment(s,d[k]) for k in names)
