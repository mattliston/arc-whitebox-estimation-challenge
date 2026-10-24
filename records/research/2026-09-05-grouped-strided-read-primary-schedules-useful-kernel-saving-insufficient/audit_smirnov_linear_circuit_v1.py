"""Exact signed/scaled-wire compiler for the whitelisted public algebra.

The compiler executes only local authored Python. Maple is parsed as data.
Signs/scales are symbolic until consumed, then ordinary scalar arithmetic
is charged. The final input scales are moved into the output linear map.
No native kernel, copy count, or runtime claim is made here.
"""
import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'smirnov_linear_circuit_v1.json'
assert not OUT.exists()
parser = (HERE / 'audit_smirnov_slp_v1.py').read_text().split('coef=json.loads', 1)[0]
parser = parser.replace("OUT=HERE/'smirnov_slp_audit_v1.json';assert not OUT.exists()", 'OUT=None')
ns = dict(__file__=str(HERE / 'audit_smirnov_slp_v1.py'), __name__='local_parser')
exec(compile(parser, str(HERE / 'audit_smirnov_slp_v1.py'), 'exec'), ns)


@dataclass(frozen=True)
class Wire:
    id: int
    scale: Fraction = Fraction(1)


nodes = []
memo = {}
values = {}
left = []
right = []
product_nodes = []


def emit(op, args, stage, vector, literal=None):
    key = (op, tuple(args), stage, str(literal))
    if key in memo:
        return memo[key]
    node = dict(op=op, args=tuple(args), stage=stage, vector=vector, literal=literal)
    result = len(nodes)
    nodes.append(node)
    memo[key] = result
    return result


def vector(wire):
    return wire.scale * nodes[wire.id]['vector']


def times(wire, factor):
    assert isinstance(wire, Wire) and isinstance(factor, Fraction)
    return Wire(wire.id, wire.scale * factor)


def actual_scale(ref, scale):
    assert scale > 0
    if scale == 1:
        return ref
    node = nodes[ref]
    return emit('scale', (ref,), node['stage'], scale * node['vector'], scale)


def plus(a, b):
    assert isinstance(a, Wire) and isinstance(b, Wire)
    assert nodes[a.id]['stage'] == nodes[b.id]['stage']
    qa, qb = a.scale, b.scale
    if not qa:
        return b
    if not qb:
        return a
    if a.id == b.id:
        return Wire(a.id, qa + qb)
    common = min(abs(qa), abs(qb))
    ia = actual_scale(a.id, abs(qa) / common)
    ib = actual_scale(b.id, abs(qb) / common)
    stage = nodes[ia]['stage']
    if (qa > 0) == (qb > 0):
        ia, ib = sorted((ia, ib))
        out = emit('add', (ia, ib), stage, nodes[ia]['vector'] + nodes[ib]['vector'])
        return Wire(out, common if qa > 0 else -common)
    if qa < 0:
        ia, ib = ib, ia
    # Canonicalize the unordered pair while keeping subtraction sign on
    # the wire; this is ordinary subtraction once the wire is consumed.
    if ia > ib:
        ia, ib = ib, ia
        common = -common
    out = emit('sub', (ia, ib), stage, nodes[ia]['vector'] - nodes[ib]['vector'])
    return Wire(out, common)


def expr(node):
    if isinstance(node, ast.Constant):
        assert type(node.value) is int
        return Fraction(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Subscript):
        assert isinstance(node.value, ast.Name) and node.value.id in ('A', 'B')
        i, j = [x.value - 1 for x in node.slice.elts]
        stage = node.value.id
        width = 3 if stage == 'A' else 6
        size = 9 if stage == 'A' else 18
        v = np.zeros(size, dtype=object)
        v[i * width + j] = Fraction(1)
        return Wire(emit('input', (), stage, v, (i, j)))
    if isinstance(node, ast.UnaryOp):
        assert isinstance(node.op, ast.USub)
        a = expr(node.operand)
        return times(a, Fraction(-1)) if isinstance(a, Wire) else -a
    if isinstance(node, ast.BinOp):
        a, b = expr(node.left), expr(node.right)
        if isinstance(node.op, ast.Add):
            return plus(a, b) if isinstance(a, Wire) else a + b
        if isinstance(node.op, ast.Sub):
            return plus(a, times(b, Fraction(-1))) if isinstance(a, Wire) else a - b
        if isinstance(node.op, ast.Mult):
            if isinstance(a, Wire):
                return times(a, b)
            if isinstance(b, Wire):
                return times(b, a)
            return a * b
        if isinstance(node.op, ast.Div):
            assert isinstance(b, Fraction)
            return times(a, 1 / b) if isinstance(a, Wire) else a / b
        raise TypeError(type(node.op))
    if isinstance(node, ast.Call):
        assert node.func.id == 'mm' and len(node.args) == 2
        a, b = (expr(x) for x in node.args)
        assert nodes[a.id]['stage'] == 'A' and nodes[b.id]['stage'] == 'B'
        # The product uses the materialized, unscaled wires. The exact
        # missing input scales are propagated through the output circuit.
        left.append(nodes[a.id]['vector'])
        right.append(nodes[b.id]['vector'])
        v = np.zeros(40, dtype=object)
        v[len(left) - 1] = Fraction(1)
        p = emit('mm', (a.id, b.id), 'C', v, len(left) - 1)
        product_nodes.append(p)
        return Wire(p, a.scale * b.scale)
    raise TypeError(type(node))


for name, node in ns['program']:
    v = expr(node)
    values[name] = v
    if name.startswith('C['):
        i, j = name[2:-1].split(',')
        values[f'output_{i}_{j}'] = v

outputs = []
for i, j in itertools.product(range(1, 4), range(1, 7)):
    w = values[f'C[{i},{j}]']
    ref = actual_scale(w.id, abs(w.scale))
    if w.scale < 0:
        ref = emit('neg', (ref,), 'C', -nodes[ref]['vector'])
    outputs.append(ref)

live = set()


def visit(ref):
    if ref in live:
        return
    live.add(ref)
    for arg in nodes[ref]['args']:
        visit(arg)


for ref in outputs:
    visit(ref)
counts = Counter(nodes[ref]['stage'] for ref in live if nodes[ref]['op'] not in ('input', 'mm'))
operations = Counter(nodes[ref]['op'] for ref in live)
assert operations['mm'] == 40
L, R = np.stack(left), np.stack(right)
P = np.stack([nodes[ref]['vector'] for ref in outputs])
tensor = np.zeros((9, 18, 18), dtype=object)
for r in range(40):
    tensor += L[r, :, None, None] * R[r, None, :, None] * P[:, r][None, None, :]
reference = np.zeros_like(tensor)
for i, q, j in itertools.product(range(3), range(3), range(6)):
    reference[i * 3 + q, q * 6 + j, i * 6 + j] = 1
error = max(abs(x) for x in (tensor - reference).flat)
assert error == 0
result = dict(
    passed=True, exact_basis_products=162, max_exact_error=str(error),
    original_arithmetic=ns['counts'], optimized_arithmetic=dict(counts),
    operation_counts=dict(operations), live_nodes=len(live), total_nodes=len(nodes),
    input_count=27, output_count=18, rank=40,
    L=[[str(x) for x in row] for row in L],
    R=[[str(x) for x in row] for row in R],
    P=[[str(x) for x in row] for row in P],
    nodes=[dict(id=r, **{k: (str(v) if k == 'literal' else v) for k, v in nodes[r].items()
                        if k != 'vector'}) for r in sorted(live)],
    output_nodes=outputs,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    qualification='Exact rational arithmetic DAG only. Every scale/negative that is actually materialized is charged; input-output gauge scales are absorbed into the output map. Copies, allocation, full live memory, numeric stability and native runtime remain unmeasured.')
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('L', 'R', 'P', 'nodes')}, indent=2))
