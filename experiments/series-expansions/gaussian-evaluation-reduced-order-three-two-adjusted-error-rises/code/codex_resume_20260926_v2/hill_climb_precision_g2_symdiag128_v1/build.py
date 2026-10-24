"""Source-only R2 builder; this file grants no job authorization."""
from pathlib import Path
import ast
import hashlib
import json

D=Path(__file__).resolve().parent
PARENT=D.parent/'hill_climb_precision_g3_symdiag128_v1/candidate.py'
PARENT_SHA='88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
EDITS=(
 ('Gaussian views','Fr16=Fr[1:4], Gr16=Gr[1:4], P16=PB[1:4], P6=PB[1:4][:, None]',
                   'Fr16=Fr[1:3], Gr16=Gr[1:3], P16=PB[1:3], P6=PB[1:3][:, None]'),
 ('Gaussian divisors','DIV6=fnp.arange(1, 4, dtype=f32)','DIV6=fnp.arange(1, 3, dtype=f32)'),
 ('X1 view','X1=fnp.reshape(arena[0:9 * nn], (3, 3, n, n))',
            'X1=fnp.reshape(arena[0:6 * nn], (2, 3, n, n))'),
 ('comment','# fixed Gaussian R3: rho^r/r!, r=1..3','# fixed Gaussian R2: rho^r/r!, r=1..2'),
)


def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    assert pin(PARENT)['sha256']==PARENT_SHA
    assert not (D/'candidate.py').exists() and not (D/'build.json').exists()
    original=PARENT.read_text();source=original
    for label,before,after in EDITS:
        assert source.count(before)==1,label
        source=source.replace(before,after,1)
    inverse=source
    for label,before,after in reversed(EDITS):
        assert inverse.count(after)==1,label
        inverse=inverse.replace(after,before,1)
    assert inverse==original
    old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body
         if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
         if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert old.keys()==new.keys()
    assert sorted(k for k in old if old[k]!=new[k])==['_e2_pair_arena','_e2_pair_state']
    with (D/'candidate.py').open('x') as f:f.write(source)
    record=dict(parent=pin(PARENT),candidate=pin(D/'candidate.py'),builder=pin(__file__),
        protocol=pin(D/'PROTOCOL.md'),pricing=pin(D/'pricing.json'),
        edits=[dict(label=k,before=b,after=a) for k,b,a in EDITS],source_inverse_exact=True,
        selected_for_execution=False,numerical_execution_authorized=False,model_executed=False)
    with (D/'build.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')


if __name__=='__main__':main()
