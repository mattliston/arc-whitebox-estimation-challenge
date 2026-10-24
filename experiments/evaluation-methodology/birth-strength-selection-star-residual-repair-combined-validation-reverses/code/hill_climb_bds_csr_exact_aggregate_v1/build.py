"""One prospective fixed conjunction; source assembly only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent

def pin(p):
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    for p in [protocol['parent'], *protocol['components']]:
        assert p == pin(Path(p['path']))
    parent, bundle, bds, csr = [Path(p['path']) for p in [protocol['parent'], *protocol['components']]]
    assert protocol['parent']['sha256'] == 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
    assert pin(bundle)['sha256'] == '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
    old = 'selected, births = select(_e2_gate_min, gate_floor)'
    new = 'selected, births = _birth_distinct_select(_e2_gate_min, gate_floor, birth, g, alive, Cpre)'
    bf = (bds.parent/'fragment.py').read_bytes()
    cf = (csr.parent/'fragment.py').read_bytes()
    original = parent.read_bytes()
    assert bds.read_bytes() == original.decode().replace(old, new).encode() + b'\n\n' + bf
    assert csr.read_bytes().endswith(cf) and csr.read_bytes().startswith(original)
    bt = bundle.read_text()
    assert bt.count(old) == 1 and new not in bt
    modified = bt.replace(old, new)
    assert modified.replace(new, old) == bt
    combined = modified.encode() + b'\n\n' + bf + b'\n\n' + cf
    ast.parse(combined)
    assert combined[:-len(cf)-2-len(bf)-2].decode().replace(new, old) == bt
    for fragment in [bf, cf]:
        assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(ast.parse(fragment)))
    with (HERE/'candidate.py').open('xb') as f:
        f.write(combined)
    report = dict(passed=True, numerical_execution=False, parent=pin(parent), candidate=pin(HERE/'candidate.py'),
                  builder=pin(Path(__file__).resolve()), protocol=pin(HERE/'protocol.json'),
                  components=[pin(p) for p in [bundle,bds,csr]],
                  fragments=[pin(bds.parent/'fragment.py'),pin(csr.parent/'fragment.py')],
                  exact_inverse_to_bundle=True, bds_callsite_count=1, standalone_failures_preserved=True)
    with (HERE/'build.json').open('x') as f:
        json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report['candidate']))

if __name__ == '__main__':
    main()
