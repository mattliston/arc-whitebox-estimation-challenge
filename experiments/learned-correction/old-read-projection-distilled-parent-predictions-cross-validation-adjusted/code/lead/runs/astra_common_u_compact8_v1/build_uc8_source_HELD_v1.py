import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
PARENT=O/'lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transform(raw):
    text=raw.decode();lines=text.splitlines(keepends=True);nodes=ast.parse(text).body;edits=[]
    for name in ('_uvp_recon','_astra_bcc_uvrecon'):
        n=[x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name][-1]
        body=''.join(lines[n.lineno-1:n.end_lineno])
        anchor="    Pu = _ubc_product(U[:, :ru], Fu, uv, 'u')\n"
        assert body.count(anchor)==1
        edits.append((n.lineno-1,n.end_lineno,body.replace(anchor,'    Pu = _uc8_recon(U, uv)\n')))
    n=[x for x in nodes if isinstance(x,ast.FunctionDef) and x.name=='_uvp_inner'][-1]
    body=''.join(lines[n.lineno-1:n.end_lineno])
    anchor="    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')\n    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))\n    _BH_HOLD.clear()\n    del Su, Dv, inner_u\n"
    assert body.count(anchor)==1
    edits.append((n.lineno-1,n.end_lineno,body.replace(anchor,'    inner = _uc8_inner(inner, Su, uv)\n    del Su, Dv\n')))
    for lo,hi,new in sorted(edits,reverse=True):lines[lo:hi]=[new]
    return ''.join(lines).encode()+b'\n\n'+(D/'fragment_uc8_HELD_v1.py').read_bytes()


if __name__=='__main__':
    assert sha(PARENT)=='4e6e4daee89b6d7c8812e145e627896fb59afc511b3b61da4a797fe54774dc76'
    raw=transform(PARENT.read_bytes());p=D/'candidate_uc8_HELD_v1.py'
    compile(raw,str(p),'exec')
    with p.open('xb')as f:f.write(raw)
    print(json.dumps(dict(candidate=str(p),sha256=sha(p))))
