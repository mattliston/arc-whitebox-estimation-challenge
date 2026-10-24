"""Target-free geometry freeze and necessary, deliberately incomplete cost screen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE/'two_high_fourth_geometry_cost_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert = json.loads((HERE/'two_high_fourth_certificate_v1.json').read_text())
assert cert['passed'] and cert['source_sha256'] == hashlib.sha256((HERE/'two_high_fourth_v1.py').read_bytes()).hexdigest()
path = HERE/'fourth_trace_oracle_inputs_r35_v1.npz'
input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
assert input_hash == json.loads(path.with_suffix('.json').read_text())['output_sha256']
f = dict(np.load(path))
coef = json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank = json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']


def mm(m, k, n):
    plain = 2*m*k*n-m*n
    if any(t % 32 for t in (m, k, n)):
        return float(plain)
    depths = (3, 4, 5) if max(m, k, n) <= 1024 else (4,)
    return min(plain, *(sum(a*b for a, b in zip(coef[str(d)], (m*k*n, m*k, k*n, m*n))) for d in depths))


with threadpool_limits(limits=3):
    C = (f['post12_covariance']+f['post12_covariance'].T)/2
    values, vectors = np.linalg.eigh(C)
    Q = vectors[:, -4:][:, ::-1].copy()
    for j in range(4):
        Q[:, j] *= np.sign(Q[np.argmax(np.abs(Q[:, j])), j])
    orthogonality = float(np.max(np.abs(Q.T@Q-np.eye(4))))
    eigen_residual = float(np.linalg.norm(C@Q-Q*values[-4:][::-1])/np.linalg.norm(C@Q))
    assert max(orthogonality, eigen_residual) <= 1e-10
    rows = []
    for rank in (35, 100):
        active = np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'alive.npy').sum(1).astype(int)
        layers = []
        for layer in range(1, 16):
            a = int(active[layer-1])
            # Ten unique symmetric q=4 matrices. At final only a diagonal read.
            core = 10*(mm(a, a, 1024)+(1024*a if layer == 15 else mm(1024, a, 1024)))
            triples = mm(1024, a, 64)
            high_push = mm(1024, a, 4)
            layers.append(dict(layer=layer, previous_active=a, affine_core=core,
                               affine_triples=triples, affine_high_frame=high_push))
        extra = sum(sum(r[k] for k in ('affine_core', 'affine_triples', 'affine_high_frame')) for r in layers)
        bill = next(r['flops'] for r in bank if r['rank'] == rank)
        rows.append(dict(rank=rank, current_measured_flops=bill, optimistic_affine_extra_flops=extra,
                         affine_extra_fraction=extra/bill, passed=extra/bill <= .5, layers=layers))
    record = dict(passed=all(r['passed'] for r in rows), q=4, rank=35, source_post_layer=12,
                  orthogonality_max=orthogonality, eigen_relative=eigen_residual,
                  retained_eigenvalues=values[-4:][::-1].tolist(), fifth_eigenvalue=float(values[-5]),
                  fourth_fifth_gap=float(values[-4]-values[-5]),
                  positive_trace_fraction=float(np.sum(values[-4:])/np.sum(np.maximum(values, 0))),
                  rows=rows, target_access=False, frozen_input_sha256=input_hash,
                  helper_sha256=cert['source_sha256'], utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Necessary optimistic affine-only cost screen, not a complete estimator, budget claim, runtime or score. Nonlinear source acquisition, eigenspace updates, contractions and third-source carry are not included. Actual masks remain frozen.')
    np.savez(OUT.with_suffix('.npz'), Q=Q, C=C, eigenvalues=values)
    record['output_sha256'] = hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({**record, 'rows':[{k:v for k,v in r.items() if k != 'layers'} for r in rows]}, indent=2), flush=True)
