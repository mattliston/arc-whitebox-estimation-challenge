"""Translate the catalog's restricted arithmetic straight-line scheme only.

Source: https://fmm.univ-lille.fr/3x3x3.html, attributed there to Laderman.
All imported syntax is checked; no Maple procedure is executed.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
source = HERE / 'rectangular_mm_reference_v1/laderman333_raw.mpl'
raw = source.read_text()
body = raw.split('C := Matrix(3,3);', 1)[1].split('; return C', 1)[0]
defined = {f'{v}_{i}_{j}' for v in ('A', 'B') for i in range(1, 4) for j in range(1, 4)}
lines = [
    '# Catalog rank-23 3x3x3 scheme; source fmm.univ-lille.fr/3x3x3.html.',
    '# Research only: every model operation is metered by fnp.',
    'def _ld333_product(a, b, mm, quantum=64):',
    '    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])',
    '    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2',
    '    M = ((m + 3*quantum - 1) // (3*quantum)) * 3*quantum',
    '    K = ((k + 3*quantum - 1) // (3*quantum)) * 3*quantum',
    '    N = ((n + 3*quantum - 1) // (3*quantum)) * 3*quantum',
    '    h, v, w = M // 3, K // 3, N // 3',
    '    aa = fnp.zeros((M, K), dtype=a.dtype)',
    '    bb = fnp.zeros((K, N), dtype=b.dtype)',
    '    fnp.copyto(aa[:m, :k], a)',
    '    fnp.copyto(bb[:k, :n], b)',
    '    out = fnp.empty((M, N), dtype=a.dtype)',
]
for name, block, rowdim, coldim in [('A', 'aa', 'h', 'v'), ('B', 'bb', 'v', 'w')]:
    for i in range(1, 4):
        for j in range(1, 4):
            lines.append(f'    {name}_{i}_{j} = {block}[{i-1}*{rowdim}:{i}*{rowdim}, {j-1}*{coldim}:{j}*{coldim}]')
products = outputs = 0
for assignment in body.split(';'):
    left, right = (v.strip() for v in assignment.strip().split(':=', 1))
    assert re.fullmatch(r'[A-Za-z][A-Za-z0-9]*|C\[[123],[123]\]', left), left
    right = re.sub(r'([AB])\[([123]),([123])\]', r'\1_\2_\3', right)
    product = re.fullmatch(r'LinearAlgebra:-MatrixMatrixMultiply\((\w+),(\w+)\)', right)
    if product:
        x, y = product.groups()
        assert x in defined and y in defined
        rhs = f'mm({x}, {y}).copy()'
        products += 1
    else:
        tree = ast.parse(right, mode='eval')
        for node in ast.walk(tree):
            assert isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add,
                ast.Sub, ast.Mult, ast.Div, ast.USub, ast.Name, ast.Load, ast.Constant)), type(node)
            if isinstance(node, ast.Name):
                assert node.id in defined, node.id
            if isinstance(node, ast.Constant):
                assert type(node.value) in (int, float) and node.value in (1, 2)
        rhs = ast.unparse(tree)
    output = re.fullmatch(r'C\[([123]),([123])\]', left)
    if output:
        i, j = map(int, output.groups())
        lines.append(f'    fnp.copyto(out[{i-1}*h:{i}*h, {j-1}*w:{j}*w], {rhs})')
        outputs += 1
    else:
        defined.add(left)
        lines.append(f'    {left} = {rhs}')
assert products == 23 and outputs == 9
lines.append('    return out[:m, :n]')
text = '\n'.join(lines) + '\n'
target = HERE / 'laderman333_fragment_v1.py'
compile(text, str(target), 'exec')
with target.open('x') as f:
    f.write(text)
record = dict(source_url='https://fmm.univ-lille.fr/3x3x3.html',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    products=products, outputs=outputs, status='unverified research helper')
with (HERE / 'laderman333_manifest_v1.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
