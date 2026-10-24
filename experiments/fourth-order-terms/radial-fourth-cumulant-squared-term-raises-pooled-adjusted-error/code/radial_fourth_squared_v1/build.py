"""Append a single fixed physical-map change to the exact Weight5 parent."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'weight5_reuse_v1/candidate.py'
EXPECTED = 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
source = PARENT.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
tree = ast.parse(source)


def last(name):
    return ast.get_source_segment(source, [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == name][-1])


def once(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b, 1)


edits = []
for name in ('_nonlinear_pair_batch', '_nonlinear_terminal_live', 'nonlinear_scalar'):
    original = last(name)
    new = original
    needle = '    var = r2 - m * m' if name != 'nonlinear_scalar' else '    var = r2-m*m'
    addition = ('    dm44, dr244, dr344, dr444 = _r44_marginal(s, a, pdf, g, mg, k4, v)\n'
                '    m = m + dm44\n    r2 = r2 + dr244\n'
                '    r3 = r3 + dr344\n    r4 = r4 + dr444\n')
    new = once(new, needle, addition + needle)
    if name != 'nonlinear_scalar':
        new = once(new, 'maxorder = max(gauss_order, edge_order + 4)', 'maxorder = 10')
        new = once(new, "    assert str(ans.dtype)=='float32'",
                   "    ans = ans + _r44_pair(F, G, rho, t4)\n    assert str(ans.dtype)=='float32'")
    # Source reversibility ensures only the stated lines change.
    recovered = new.replace(addition, '', 1)
    if name != 'nonlinear_scalar':
        recovered = recovered.replace('maxorder = 10', 'maxorder = max(gauss_order, edge_order + 4)', 1)
        recovered = recovered.replace('    ans = ans + _r44_pair(F, G, rho, t4)\n', '', 1)
    assert recovered == original
    edits.append(new)
extension = '\n\n# Radial fourth-squared map, fixed natural coefficient.\n'
extension += (HERE / 'kernel.py').read_text() + '\n\n' + '\n\n'.join(edits)
extension += '\n\nnonlinear = _nonlinear_pair_batch\n'
candidate = source + extension
ast.parse(candidate)
assert candidate[:-len(extension)] == source
with (HERE / 'candidate.py').open('x') as stream:
    stream.write(candidate)
receipt = dict(parent_sha256=EXPECTED,
               candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
               kernel_sha256=hashlib.sha256((HERE / 'kernel.py').read_bytes()).hexdigest(),
               builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               protocol_sha256=hashlib.sha256((HERE / 'PROTOCOL.md').read_bytes()).hexdigest(),
               parent_byte_prefix_exact=True, changed_definitions=[x for x in
                   ('_nonlinear_pair_batch', '_nonlinear_terminal_live', 'nonlinear_scalar')],
               source_inverse_exact=True, truth_accessed=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
