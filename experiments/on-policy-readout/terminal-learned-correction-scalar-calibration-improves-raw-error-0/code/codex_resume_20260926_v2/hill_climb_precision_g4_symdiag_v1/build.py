"""Stdlib-only prospective composition; neither estimator is imported."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent

def pin(p):
    p = Path(p).resolve()
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def checked(r):
    b = Path(r['path']).read_bytes()
    assert len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256']
    return b

def main():
    gd = H/'hill_climb_precision_gaussian4_v1'
    sd = H/'hill_climb_precision_symdiag_v1'
    assert pin(gd/'build.json')['sha256'] == '024a42d88e86e8c70eb3c3b5a210319e39282503cfcc1e40de1c478851ec06e9'
    gb = json.loads((gd/'build.json').read_text())
    sb = json.loads((sd/'build.json').read_text())
    for b in (gb, sb):
        for v in b.values():
            if isinstance(v, dict) and set(('path', 'sha256', 'bytes')) <= v.keys():
                checked(v)
    assert gb['candidate']['sha256'] == 'd3379e664e26d4fc5b88e1c3456d756173eaeeddc9ba36017755e3d682fa82bf'
    assert sb['candidate']['sha256'] == '3f88212643535861789a051ed439e05b653a04a76653863956640369ce96e004'
    assert sb['fragment']['sha256'] == 'bbbc4bc7995d54a0484ce7cd3240ed0567c4dd0e4cbb6c21cd216ae43083146b'
    assert gb['parent'] == sb['parent']
    parent = checked(gb['parent'])
    gaussian = checked(gb['candidate'])
    suffix = b'\n' + checked(sb['fragment'])
    assert checked(sb['candidate']) == parent + suffix
    candidate = gaussian + suffix
    restored = candidate[:-len(suffix)].decode()
    for edit in reversed(gb['edits']):
        assert restored.count(edit['after']) == 1
        restored = restored.replace(edit['after'], edit['before'], 1)
    assert restored.encode() == parent
    ast.parse(candidate)
    with (D/'candidate.py').open('xb') as f:
        f.write(candidate)
    out = dict(gb)
    out.update(candidate=pin(D/'candidate.py'), builder=pin(__file__),
               protocol=pin(D/'PROTOCOL.md'), gaussian_source=gb['candidate'],
               gaussian_build=pin(gd/'build.json'), symdiag_build=pin(sd/'build.json'),
               symdiag_source=sb['candidate'], fragment=sb['fragment'], pricing=sb['pricing'],
               symdiag_review=pin(sd/'independent_accuracy_source_review.json'),
               suffix_bytes=len(suffix), append_after_gaussian_source=True,
               source_inverse_exact=True, selected_partition=[128,64,64],
               expected_setup=dict(bound=673, fused=649, sides=[200,259,190]),
               numerical_execution=False, model_executed=False,
               component_sources_unexecuted=True, qualified=False, submitted=False)
    with (D/'build.json').open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps({k: pin(D/k) for k in ('candidate.py','build.py','build.json','PROTOCOL.md')}))

if __name__ == '__main__':
    main()
