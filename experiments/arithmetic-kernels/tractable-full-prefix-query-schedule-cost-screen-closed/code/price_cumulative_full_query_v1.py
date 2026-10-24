"""Optimistic partial cost of the declared exact/full prefix query schedule."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cumulative_full_query_cost_v1.json'
assert not OUT.exists()
coefficients = json.loads((HERE / 'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank = json.loads((HERE / 'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']


def mm(m, k, n):
    plain = 2 * m * k * n - m * n
    assert all(t % 32 == 0 for t in (m, k, n))
    prices = [sum(a * b for a, b in zip(coefficients[str(depth)], (m*k*n, m*k, k*n, m*n)))
              for depth in (3, 4, 5)]
    return float(min(plain, *prices))


def mixed(a, o):
    return 2 * mm(o, a, a) + 2 * mm(a, a, o) + 4 * mm(o, a, o)


rows = []
for rank in (35, 100):
    alive = np.load(HERE / 'rows_current_transport_shadow_v2' / f'row_{rank:04d}' / 'alive.npy')
    active = alive.sum(axis=1).astype(int)
    post_width, next_width = int(active[12]), int(active[13])
    layers = []
    for layer in range(13):
        a = int(active[layer])
        tractable = mixed(a, post_width) + mixed(a, next_width)
        triangle = 4 * (mm(a, a, a) + mm(a, a, post_width) + mm(post_width, a, post_width))
        layers.append(dict(source_layer=layer, source_active=a, tractable_dual_query_flops=tractable,
                           triangle_slices_and_transport_flops=triangle, subtotal=tractable + triangle))
    subtotal = sum(row['subtotal'] for row in layers)
    baseline = next(row['flops'] for row in bank if row['rank'] == rank)
    ratio = subtotal / baseline
    rows.append(dict(rank=rank, active_post12=post_width, active_pre13=next_width,
                     omitted_costs_included=False, layers=layers, optimistic_added_subtotal=subtotal,
                     whole_bank_first_context_measured_flops=baseline, optimistic_added_fraction=ratio,
                     passed=bool(ratio <= .50)))
result = dict(passed=all(row['passed'] for row in rows), rows=rows,
              threshold=.50, utc_finished=datetime.now(timezone.utc).isoformat(),
              qualification='Subtotal for one declared pre13 exact/full query with compact source/output supports and only triangle kept q4. Cheapest existing square Winograd float32 bills. Old-star/trace sources, final repeat removal, Jacobian products, scalar/cold outputs, source construction, projection finish, storage and remote costs omitted. A lower workload for this compiler schedule only, NOT an algorithm-independent lower bound, candidate bill, runtime or accuracy result.')
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=result['passed'], rows=[{k:v for k,v in row.items() if k != 'layers'} for row in rows]), indent=2), flush=True)
