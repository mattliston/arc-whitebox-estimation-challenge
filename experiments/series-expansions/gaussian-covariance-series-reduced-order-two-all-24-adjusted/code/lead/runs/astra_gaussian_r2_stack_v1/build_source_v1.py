"""Fixed Gaussian-only R3 -> R2; exact QR+MLP parent, no model calls."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent / 'astra_native_terminal_mlp16_v1/candidate_QR_terminal_mlp_ON_HELD_v1.py'
SHA = 'bac3a67b29283091c59dc4e3df968105ec49fdb68be32ad7e68f95fa8129403f'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P) == SHA
source = P.read_text()
replacements = [
    ('    hold(fnp.sum(X, axis=0, out=z0))\n    # Gaussian is fully consumed',
     '    hold(fnp.sum(X[:2], axis=0, out=z0))  # GR2: Gaussian orders1,2 only; shared T22 products retained.\n    # Gaussian is fully consumed'),
    ('        _e2_hold.append(fnp.sum(X, axis=0, out=Z0))   # ans',
     '        _e2_hold.append(fnp.sum(X[:2], axis=0, out=Z0))   # GR2: Gaussian-only fallback; all non-Gaussian work retained.'),
]
candidate = source
for old, new in replacements:
    assert candidate.count(old) == 1
    candidate = candidate.replace(old, new)
inverse = candidate
for old, new in reversed(replacements):
    assert inverse.count(new) == 1
    inverse = inverse.replace(new, old)
assert inverse == source
ast.parse(candidate)
out = D / 'candidate_QR_MLP_gaussian_R2_HELD_v1.py'
with out.open('x') as f:
    f.write(candidate)
def extract(text, name):
    nodes = [x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == name]
    x = nodes[-1]
    return '\n'.join(text.splitlines()[x.lineno-1:x.end_lineno]) + '\n'
for label, text in [('parent', source), ('R2', candidate)]:
    (D / (label + '_psp_series_v1.py')).write_text(extract(text, '_psp_series'))
proof = dict(parent_path=str(P), parent_sha256=SHA, candidate_path=str(out), candidate_sha256=sha(out),
    builder_sha256=sha(__file__), exact_inverse_to_parent=True, replacements=replacements,
    modified_functions=['_psp_series (active GTR definition)', '_nonlinear_pair_batch (nonpacked fallback)'],
    unchanged=['All F/G products, including shared F2G2/F3G3 used by T22.',
        'All Z1..Z4 mixed/local cumulant computations and final ordered sum.',
        'Every scalar mean/variance/cumulant and exact diagonal overwrite.',
        'L0 first-Gaussian path, L15 scalar path, QR refinement, terminal MLP and all source captures.'],
    qualification='Source-only. Downstream states change; no first15 parity claim. No native calls authorized here.')
path = D / 'SOURCE_PROOF_v1.json'
with path.open('x') as f:
    json.dump(proof, f, indent=2); f.write('\n')
print(json.dumps(dict(candidate=str(out), candidate_sha256=sha(out), proof_sha256=sha(path))))
