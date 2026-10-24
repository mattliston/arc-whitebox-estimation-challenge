"""One theory-selected, fully metered radial Stein birth direction."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = 'dd64dfc367fbaf154fe956f335c21aa51c1589e7f19a731be74c29ca41bc29b3'


def main():
    path = HERE.parent/'fable_wall_repair_v1/cand_c2_wallrepair_I95.py'
    source = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_SHA
    tree = ast.parse(source)
    node, = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bd_update']
    old = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    replacements = [
        ('    for j in range(4):', '    for j in range(5):'),
        ('''        else:
            direction = Cpre * gate_jet[:, None]''', '''        elif j == 3:
            direction = Cpre * gate_jet[:, None]
        else:
            excess = aux['pre_k4'] / (sigma3 * sigma)
            fourth_stein = -excess * (alpha + alpha*alpha*alpha) * aux['pdf'] / 24
            direction = Cpre * fourth_stein[:, None]'''),
        ('        change = change + direction * c[j]', '        change = change + direction * (c[j] if j < 4 else 1.0)'),
    ]
    new = old
    for a,b in replacements:
        assert new.count(a)==1
        new = new.replace(a,b)
    candidate = source.replace(old,new)
    assert candidate.replace(new,old)==source
    ast.parse(candidate)
    out = HERE/'fourth_birth_I95.py'
    with out.open('x') as f:
        f.write(candidate)
    receipt = dict(parent_sha256=PARENT_SHA,candidate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                   source_inverse_exact=True,new_coefficient=1,learned_payloads_changed=False,
                   labels_used=False,interpretation='First-order radial fourth Stein birth field; incomplete closure hypothesis.')
    with (HERE/'fourth_birth_build.json').open('x') as f:
        json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
