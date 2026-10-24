"""Prospective GEMM-only saving of half CP pairs; not an accuracy claim."""
import json
from pathlib import Path
import hashlib
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'uniform_half_cp_shape_cost_v1.json'
assert not OUT.exists()
COEF = json.loads((HERE / 'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
BANK = json.loads((HERE / 'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']


def mm(m, k, n):
    if not m or not k or not n:
        return 0.
    if n > 8192:
        return sum(mm(m, k, min(8192, n-j)) for j in range(0, n, 8192)) + m*n
    if k > 8192:
        return sum(mm(m, min(8192, k-j), n) for j in range(0, k, 8192)) + (len(range(0, k, 8192))-1)*m*n
    plain = 2*m*k*n-m*n
    if any(t % 32 for t in (m, k, n)):
        return float(plain)
    depths = (3, 4, 5) if max(m, k, n) <= 1024 else (4,)
    prices = [sum(a*b for a, b in zip(COEF[str(d)], (m*k*n, m*k, k*n, m*n))) for d in depths]
    return min(plain, *prices)


def cp_products(counts, active):
    rows = []
    for layer in range(1, 16):
        previous = int(active[layer-1])
        inherited = int(2*sum(counts[:layer-1]))
        newest = int(counts[layer-1])
        pushed_columns = inherited + 2*newest
        # Existing bank shares the newest signed pair's common input push.
        push = mm(1024, previous, inherited) + mm(1024, previous, newest)
        read = mm(int(active[layer]), pushed_columns, int(active[layer])) if layer < 15 else 0.
        rows.append(dict(layer=layer, inherited_columns=inherited, newest_pairs=newest,
                         push_flops=push, matrix_plane_flops=read))
    return rows


records = []
for rank in (35, 100):
    z = np.load(HERE / 'rows_final_cohort_source_probe_v1' / f'row_{rank:04d}' / 'data.npz')
    counts = z['birth_counts'].copy()
    active = np.load(HERE / 'rows_current_transport_shadow_v2' / f'row_{rank:04d}' / 'alive.npy').sum(axis=1).astype(int)
    # Round retained pair counts UP to 32, preserving the bank's fast-product
    # divisibility. This exceeds the half-pair accuracy-screen budget slightly.
    half = ((counts // 2 + 31) // 32 * 32).clip(max=counts)
    original = cp_products(counts, active)
    reduced = cp_products(half, active)
    original_sum = sum(r['push_flops'] + r['matrix_plane_flops'] for r in original)
    reduced_sum = sum(r['push_flops'] + r['matrix_plane_flops'] for r in reduced)
    bank = next(r['flops'] for r in BANK if r['rank'] == rank)
    records.append(dict(rank=rank, original_pair_count=int(counts.sum()),
                        retained_pair_count=int(half.sum()), retained_fraction=float(half.sum()/counts.sum()),
                        original_cp_gemm_projection=original_sum, retained_cp_gemm_projection=reduced_sum,
                        affected_gemm_saving=original_sum-reduced_sum,
                        first_context_bank_flops=bank,
                        optimistic_whole_bank_fraction_saved=(original_sum-reduced_sum)/bank,
                        original_layers=original, retained_layers=reduced))
result = dict(records=records, utc_finished=datetime.now(timezone.utc).isoformat(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Target-free shape-only arithmetic projection using known D4 rectangle and D3/4/5 square polynomial costs. Preserves newest-pair shared push. Uniform half-per-cohort rounded UP to32; no sampling, carrier, accuracy, actual runtime or measured-candidate FLOPs. All acquisition, changed non-GEMM costs, branch feedback and head changes omitted. Not the optimum allocation used in the separate information gate.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({**result, 'records': [{k:v for k,v in r.items() if not k.endswith('_layers')} for r in records]}, indent=2))
