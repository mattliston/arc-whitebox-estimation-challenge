"""Certify six executable, regauged linear DAGs and their packed costs."""
import ast
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from executable_linear_dag_v1 import (Builder, Wire, transpose, column_transpose,
                                    row_transpose, physical_bilinear, matrix,
                                    accounting, serialize)

HERE = Path(__file__).resolve().parent
OUT = HERE/'smirnov_executable_transpose_certificate_v1.json'
assert not OUT.exists()
old = json.loads((HERE/'smirnov_linear_circuit_v1.json').read_text())
assert old['passed']
products = [x for x in old['nodes'] if x['op'] == 'mm']
products.sort(key=lambda x: int(x['literal']))
assert len(products) == 40


def extract(stage):
    count = dict(A=9, B=18, C=40)[stage]
    b = Builder(count)
    values = {}
    for node in old['nodes']:
        if node['stage'] != stage:
            continue
        op, args = node['op'], node['args']
        if op == 'input':
            i, j = ast.literal_eval(node['literal'])
            value = Wire(i*(3 if stage == 'A' else 6)+j)
        elif op == 'mm':
            value = Wire(int(node['literal']))
        elif op == 'add':
            value = b.add(values[args[0]], values[args[1]])
        elif op == 'sub':
            value = b.add(values[args[0]], b.times(values[args[1]], -1))
        elif op in ('scale', 'neg'):
            value = b.times(values[args[0]], -1 if op == 'neg' else Fraction(node['literal']))
        else:
            raise ValueError(op)
        values[node['id']] = value
    refs = ([p['args'][0 if stage == 'A' else 1] for p in products]
            if stage in ('A', 'B') else old['output_nodes'])
    return b.circuit([values[ref] for ref in refs])


start = time.time()
L, R, P = (extract(stage) for stage in ('A', 'B', 'C'))
for circuit, key in ((L, 'L'), (R, 'R'), (P, 'P')):
    assert np.array_equal(matrix(circuit), np.array([[Fraction(x) for x in row] for row in old[key]], dtype=object))
    assert np.array_equal(matrix(transpose(circuit)), matrix(circuit).T)
entries = [dict(name='signed_executable_0', shape=(3, 3, 6), L=L, R=R, P=P),
           dict(name='signed_executable_1', shape=(3, 6, 3), L=R,
                R=column_transpose(transpose(P), 3, 6),
                P=row_transpose(transpose(L), 3, 3)),
           dict(name='signed_executable_2', shape=(6, 3, 3),
                L=column_transpose(transpose(P), 3, 6), R=L,
                P=row_transpose(transpose(R), 3, 6))]
for e in entries.copy():
    m, k, n = e['shape']
    entries.append(dict(name=e['name']+'_transpose', shape=(n, k, m),
                        L=column_transpose(e['R'], k, n),
                        R=column_transpose(e['L'], m, k),
                        P=row_transpose(e['P'], m, n)))
records = []
for e in entries:
    m, k, n = e['shape']
    left, right, output = physical_bilinear(e['L'], e['R'], e['P'])
    l, r, p = (matrix(x) for x in (left, right, output))
    value = np.zeros((m*k, k*n, m*n), dtype=object)
    for i in range(40):
        value += l[i, :, None, None]*r[i, None, :, None]*p[:, i][None, None, :]
    target = np.zeros_like(value)
    for i, q, j in itertools.product(range(m), range(k), range(n)):
        target[i*k+q, q*n+j, i*n+j] = Fraction(1)
    assert np.array_equal(value, target), e['name']
    stats = [accounting(x) for x in (left, right, output)]
    record = dict(name=e['name'], shape=e['shape'], rank=40, passed=True,
                  exact_basis_products=m*k*k*n,
                  arithmetic=[s['arithmetic'] for s in stats],
                  packed_copies=[s['packed_copies'] for s in stats],
                  packed_linear_operations=[s['billed_linear_operations'] for s in stats],
                  accounting=stats,
                  circuits={key: serialize(x) for key, x in zip(('L', 'R', 'P'), (left, right, output))})
    records.append(record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('accounting', 'circuits')}), flush=True)
result = dict(passed=True, records=records, wall_s=time.time()-start,
              source_sha256=hashlib.sha256((HERE/'executable_linear_dag_v1.py').read_bytes()).hexdigest(),
              qualification='Actual exact rational linear DAGs, symbolic gauges moved into output, direct-input and duplicate packed copies explicitly counted. Computed nodes may write their final output slots directly. Intermediate storage, padding/fringes, full live memory, native operations and numerical stability not yet certified. No candidate or score.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
