"""Translate only the arithmetic expressions of the public Hopcroft-Kerr table.

No downloaded code is executed. The expression grammar is restricted to
named matrix blocks, addition/subtraction and one product for each m_i.
Source: https://fmm.univ-lille.fr/2x3x3.html (Hopcroft and Kerr, 1971).
"""
from pathlib import Path
import ast
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
source = HERE / 'rectangular_mm_reference_v1/hk233_raw.mpl'
raw = source.read_text()


def expression(text, names):
    tree = ast.parse(text, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add,
                                 ast.Sub, ast.USub, ast.Name, ast.Load)):
            raise ValueError(type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(node.id)
    return ast.unparse(tree)


names_a = {f'A_{i}_{j}' for i in (1, 2) for j in (1, 2, 3)}
names_b = {f'B_{i}_{j}' for i in (1, 2, 3) for j in (1, 2, 3)}
products = re.findall(r'(m_\d+)=\(([^\n]+)\)\*\(([^\n]+)\)', raw)
outputs = re.findall(r'(C_[12]_[123])=([^,\n]+)', raw)
assert len(products) == 15 and len(outputs) == 6
lines = [
    '# Hopcroft-Kerr <2,3,3:15>, coefficients from fmm.univ-lille.fr/2x3x3.html.',
    '# Research helper only. All model arithmetic uses metered fnp operations.',
    'def _hk233_product(a, b, mm, quantum=64):',
    '    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])',
    '    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2',
    '    M = ((m + 2 * quantum - 1) // (2 * quantum)) * 2 * quantum',
    '    K = ((k + 3 * quantum - 1) // (3 * quantum)) * 3 * quantum',
    '    N = ((n + 3 * quantum - 1) // (3 * quantum)) * 3 * quantum',
    '    h, v, w = M // 2, K // 3, N // 3',
    '    aa = fnp.zeros((M, K), dtype=a.dtype)',
    '    bb = fnp.zeros((K, N), dtype=b.dtype)',
    '    fnp.copyto(aa[:m, :k], a)',
    '    fnp.copyto(bb[:k, :n], b)',
]
for i in range(1, 3):
    for j in range(1, 4):
        lines.append(f'    A_{i}_{j} = aa[{i-1}*h:{i}*h, {j-1}*v:{j}*v]')
for i in range(1, 4):
    for j in range(1, 4):
        lines.append(f'    B_{i}_{j} = bb[{i-1}*v:{i}*v, {j-1}*w:{j}*w]')
for name, a, b in products:
    # Each product is copied before the next child may reuse its scratch arena.
    lines.append(f'    {name} = mm({expression(a, names_a)}, {expression(b, names_b)}).copy()')
lines.append('    out = fnp.empty((M, N), dtype=a.dtype)')
names_m = {name for name, _, _ in products}
for name, rhs in outputs:
    _, i, j = name.split('_')
    i, j = int(i), int(j)
    lines.append(f'    fnp.copyto(out[{i-1}*h:{i}*h, {j-1}*w:{j}*w], {expression(rhs, names_m)})')
lines.append('    return out[:m, :n]')
text = '\n'.join(lines) + '\n'
target = HERE / 'hk233_fragment_v1.py'
compile(text, str(target), 'exec')
with target.open('x') as f:
    f.write(text)
record = dict(source_url='https://fmm.univ-lille.fr/2x3x3.html',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              products=15, outputs=6, status='unverified research helper')
with (HERE / 'hk233_manifest_v1.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
