"""Exact executable/copy receipts for the same six public scalar programs."""
import bz2
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from compile_catalog_linear_dag_v1 import compile_program
from executable_linear_dag_v1 import (transpose, column_transpose, row_transpose,
                                    physical_bilinear, matrix, accounting, serialize)

HERE = Path(__file__).resolve().parent
OUT = HERE/'catalog_executable_cse_certificate_v1.json'
assert not OUT.exists()
local = (HERE/'audit_rectangular_catalog_v1.py').read_text().split('start=time.time()', 1)[0]
local = local.replace('assert not OUT.exists()', '')
ns = dict(__file__=str(HERE/'audit_rectangular_catalog_v1.py'))
exec(compile(local, 'local_restricted_scalar_parser', 'exec'), ns)
old = json.loads((HERE/'rectangular_catalog_certificate_v1.json').read_text())
assert old['passed']
start = time.time()
records = []
for rec in old['records']:
    shape, rank = tuple(rec['shape']), rec['rank']
    m, k, n = shape
    name = 'x'.join(map(str, shape))
    compressed = (HERE/'public_rectangular_catalog_data_v1'/(name+'_raw.mpl.bz2')).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == rec['compressed_sha256']
    program, kind = ns['parse'](bz2.decompress(compressed).decode('ascii'), shape)
    L, R, P = compile_program(program, shape, rank)
    entries = [dict(name=name+'_primary', shape=shape, L=L, R=R, P=P),
               dict(name=name+'_cyclic1', shape=(k, n, m), L=R,
                    R=column_transpose(transpose(P), m, n),
                    P=row_transpose(transpose(L), m, k)),
               dict(name=name+'_cyclic2', shape=(n, m, k),
                    L=column_transpose(transpose(P), m, n), R=L,
                    P=row_transpose(transpose(R), k, n))]
    for e in entries.copy():
        em, ek, en = e['shape']
        entries.append(dict(name=e['name']+'_transpose', shape=(en, ek, em),
                            L=column_transpose(e['R'], ek, en),
                            R=column_transpose(e['L'], em, ek),
                            P=row_transpose(e['P'], em, en)))
    for e in entries:
        em, ek, en = e['shape']
        circuits = physical_bilinear(e['L'], e['R'], e['P'])
        l, r, p = (matrix(x) for x in circuits)
        value = np.zeros((em*ek, ek*en, em*en), dtype=object)
        for i in range(rank):
            value += l[i, :, None, None]*r[i, None, :, None]*p[:, i][None, None, :]
        target = np.zeros_like(value)
        for i, q, j in itertools.product(range(em), range(ek), range(en)):
            target[i*ek+q, q*en+j, i*en+j] = Fraction(1)
        assert np.array_equal(value, target), e['name']
        stats = [accounting(c) for c in circuits]
        record = dict(name='executable_catalog_'+e['name'], shape=e['shape'], rank=rank,
                      passed=True, exact_basis_products=em*ek*ek*en,
                      arithmetic=[s['arithmetic'] for s in stats],
                      packed_copies=[s['packed_copies'] for s in stats],
                      packed_linear_operations=[s['billed_linear_operations'] for s in stats],
                      accounting=stats,
                      circuits={key: serialize(c) for key, c in zip(('L', 'R', 'P'), circuits)})
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in ('accounting', 'circuits')}), flush=True)
result = dict(passed=True, records=records, wall_s=time.time()-start,
              compiler_sha256=hashlib.sha256((HERE/'executable_linear_dag_v1.py').read_bytes()).hexdigest(),
              parser_compiler_sha256=hashlib.sha256((HERE/'compile_catalog_linear_dag_v1.py').read_bytes()).hexdigest(),
              qualification='Same six public programs and36 orientations, actual exact signed/scaled common-subexpression DAGs, packed direct/duplicate copies included. No downloaded execution or coefficient/ordering search. Live intermediate allocation, full native cost/runtime and floating-point stability remain unproved.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
