"""Optimistic lower-cost projection for the new certified full trace.

This is NOT measured candidate FLOPs. It uses current mask/birth tapes
and already metered public-kernel polynomials. It separately shows why
pricing only the new matrix core omits a material downstream carry cost.
"""
from pathlib import Path
import json
import numpy as np

here = Path(__file__).resolve().parent
out = here/'augmented_fourth_trace_cost_v1.json'
assert not out.exists()
coef = json.loads((here/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank = json.loads((here/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']


def mm(m, k, n):
    if not m or not k or not n:
        return 0.
    if n > 8192:
        return sum(mm(m, k, min(8192, n-j)) for j in range(0, n, 8192))+m*n
    if k > 8192:
        return sum(mm(m, min(8192, k-j), n) for j in range(0, k, 8192))+(len(range(0, k, 8192))-1)*m*n
    plain = 2*m*k*n-m*n
    if any(t % 32 for t in (m, k, n)):
        return float(plain)
    depths = (3, 4, 5) if max(m, k, n) <= 1024 else (4,)
    return min(plain, *(sum(a*b for a, b in zip(coef[str(d)], (m*k*n, m*k, k*n, m*n))) for d in depths))


rows = []
for rank in (35, 100):
    folder = here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x = np.load(folder/'x.npy')
    active = np.load(folder/'alive.npy').sum(1).astype(int)
    births = ((np.count_nonzero(np.minimum(x[:, :, 5], 1-x[:, :, 5]) >= .01, axis=1)+31)//32*32).clip(max=1024)
    layers = []
    cp_columns = 0
    new_source_columns = 0
    for l in range(16):
        a = int(active[l])
        prev_a = 1024 if l == 0 else int(active[l-1])
        # Optimistic: everything may be restricted to existing active support;
        # input/output geometry remains unchanged by the new state.
        acquisition = 0 if l == 15 else 4*mm(a, a, a)+3*mm(a, cp_columns, a)
        # Full residual core two-sided push plus the new identity metric push.
        push = 0 if l == 0 else mm(1024, prev_a, prev_a)+mm(1024, prev_a, 1024)+mm(1024, prev_a, 1024)
        # ONLY the n-column mixed A/identity/B third-source branch. The two
        # vector/matrix branches, its production and repeated-plane repair,
        # star/trace third-input contractions and all scalar operations are
        # omitted. Each inherited mixed branch requires three factor pushes
        # and three repeated-plane products (or a separately certified fusion).
        source_push = 3*mm(1024, prev_a, new_source_columns)
        source_read = 0 if l == 15 else 3*mm(a, new_source_columns, a)
        layers.append(dict(layer=l, active=a, cp_columns=cp_columns,
                           new_mixed_source_columns=new_source_columns,
                           core_acquisition=acquisition, residual_metric_push=push,
                           one_mixed_source_push=source_push, one_mixed_source_read=source_read))
        cp_columns += 2*int(births[l])
        # First pre-nonlinear fourth tensor is zero, so no new branch at L0.
        if 0 < l < 15:
            new_source_columns += int(births[l])
    total = next(r['flops'] for r in bank if r['rank'] == rank)
    core_extra = sum(r['core_acquisition']+r['residual_metric_push'] for r in layers)
    source_extra = sum(r['one_mixed_source_push']+r['one_mixed_source_read'] for r in layers)
    projected = total+core_extra+source_extra
    rows.append(dict(rank=rank, current_first_context_measured_flops=total,
                     optimistic_core_only_extra_flops=core_extra,
                     optimistic_one_source_branch_extra_flops=source_extra,
                     optimistic_total_flops=projected,
                     optimistic_total_utilization=projected/2**41,
                     core_only_flop_multiplier=1+core_extra/total,
                     including_one_source_flop_multiplier=projected/total,
                     raw_needed_for_4e9=4e-9/(projected/2**41),
                     raw_needed_for_1e9=1e-9/(projected/2**41),
                     layers=layers))
receipt = dict(rows=rows, qualification='Analytical optimistic lower-cost projection for one exact mixed-factor source representation, not a universal complexity bound, measured bill, runtime or raw result. Includes only one of three fourth-to-third branches; omitted work is enumerated in source. Existing active masks are frozen.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({**receipt, 'rows': [{k: v for k, v in r.items() if k != 'layers'} for r in rows]}, indent=2))
