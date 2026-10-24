"""One fixed leading256 interpolated / trailing128 exact QR coefficient test."""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
CAP = D.parent / 'opus_sketchgb_v1' / 'cap'
RANK, LEAD = 384, 256
t0 = time.perf_counter()
rows = []


def metric(cov, gram):
    return {
        'weighted_residual_energy': float(np.trace(cov)),
        'energy_fraction': float(np.trace(cov) / np.trace(gram)),
        'residual_covariance_fro_fraction': float(np.linalg.norm(cov) / np.linalg.norm(gram)),
    }


for path in sorted(CAP.glob('join_*.npz')):
    with np.load(path) as z:
        J = z['J'].astype(np.float64)
        w = z['w'].astype(np.float64)
        Qp = z['Qp'].astype(np.float64)
        core = z['core'].astype(np.float64)
        saved_rank = int(z['rank'])
    assert np.all(w >= 0)
    n, width = J.shape
    Gj = (J * w[None, :]) @ J.T
    Go = Qp @ core @ Qp.T if Qp.size else np.zeros_like(Gj)
    G = Gj + Go
    G = .5 * (G + G.T)
    S = np.sort(np.argsort(-np.diag(G), kind='stable')[:RANK])
    Q, _ = np.linalg.qr(G[:, S], mode='reduced')
    P = Q @ Q.T
    # Both row-interpolation operators use the same complete384 square solve.
    T = np.linalg.solve(Q[S, :], np.eye(RANK))
    Bfull = np.zeros((n, n))
    Bfull[:, S] = Q @ T
    Bhybrid = P - Q[:, :LEAD] @ Q[:, :LEAD].T
    Bhybrid[:, S] += Q[:, :LEAD] @ T[:LEAD, :]
    Rorth = np.eye(n) - P
    old_residual_cov = Rorth @ Go @ Rorth.T
    methods = {}
    for name, B in [('orthogonal', P), ('full_interpolation', Bfull), ('hybrid256_128', Bhybrid)]:
        A = np.eye(n) - B
        young_cov = A @ Gj @ A.T
        methods[name] = {
            'J_only_old_overlap_exact': metric(young_cov + old_residual_cov, G),
            'J_only': metric(young_cov, Gj),
            'all_source_interpolated': metric(A @ G @ A.T, G),
        }
    # Explicit coefficients independently verify the residual-energy decomposition.
    F = Q.T @ J
    Fi = T @ J[S, :]
    E = Fi - F
    extra_lead = float(np.sum(E[:LEAD] ** 2 * w[None, :]))
    extra_tail = float(np.sum(E[LEAD:] ** 2 * w[None, :]))
    base = methods['orthogonal']['J_only_old_overlap_exact']['weighted_residual_energy']
    hyb = methods['hybrid256_128']['J_only_old_overlap_exact']['weighted_residual_energy']
    full = methods['full_interpolation']['J_only_old_overlap_exact']['weighted_residual_energy']
    assert abs(hyb - base - extra_lead) <= 1e-8 * max(base, extra_lead, 1e-20)
    assert abs(full - base - extra_lead - extra_tail) <= 1e-8 * max(base, extra_lead + extra_tail, 1e-20)
    result = {
        'capture': path.name,
        'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'J_shape': list(J.shape), 'old_rank': int(Qp.shape[1]) if Qp.size else 0,
        'saved_rank': saved_rank, 'reconstructed_rank': RANK,
        'leading_interpolated_rank': LEAD,
        'condition_Q_pivot_rows': float(np.linalg.cond(Q[S, :])),
        'methods': methods,
        'hybrid_vs_orth_energy_ratio': hyb / base,
        'full_interp_vs_orth_energy_ratio': full / base,
        'interpolation_damage_retained_by_leading256': extra_lead / (extra_lead + extra_tail),
        'interpolation_damage_removed_by_exact_tail128': extra_tail / (extra_lead + extra_tail),
        'leading_extra_energy': extra_lead, 'trailing_extra_energy': extra_tail,
    }
    rows.append(result)
    print(path.name, 'hybrid/orth', round(hyb/base, 5), 'full/orth', round(full/base, 5),
          'damage_retained', round(result['interpolation_damage_retained_by_leading256'], 5), flush=True)

out = {
    'status': 'SAVED_GEOMETRY_ONLY',
    'rows': rows,
    'mean_hybrid_vs_orth_energy_ratio': float(np.mean([r['hybrid_vs_orth_energy_ratio'] for r in rows])),
    'mean_full_interp_vs_orth_energy_ratio': float(np.mean([r['full_interp_vs_orth_energy_ratio'] for r in rows])),
    'mean_fraction_interpolation_damage_retained': float(np.mean([r['interpolation_damage_retained_by_leading256'] for r in rows])),
    'seconds': time.perf_counter() - t0,
    'max_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'notes': [
        'Main deployable comparison changes young J projection only; old overlap remains exact.',
        'All-source diagnostic changes both blocks and reproduces full Gram interpolation semantics.',
        'Residual covariance Frobenius is ||(I-B)G(I-B)^T||/||G||, not ||G-PGP||/||G||.',
        'No targets, final predictions, public outcomes, rank/threshold grid, or native models were used.',
    ],
}
(D / 'RESULT_v1.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows', 'notes')}, indent=2))
