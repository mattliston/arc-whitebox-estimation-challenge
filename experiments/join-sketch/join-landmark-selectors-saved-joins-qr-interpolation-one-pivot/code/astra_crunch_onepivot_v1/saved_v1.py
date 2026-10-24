import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
CAP = D.parent / 'opus_sketchgb_v1' / 'cap'
t0 = time.perf_counter()
rows = []
for path in sorted(CAP.glob('join_*.npz')):
    with np.load(path) as z:
        J, w, Qp, core = [z[k].astype(np.float64) for k in ('J', 'w', 'Qp', 'core')]
    G = (J * w[None, :]) @ J.T
    if Qp.size:
        G += Qp @ core @ Qp.T
    G = .5 * (G + G.T)
    n = G.shape[0]
    diagonal = np.diag(G)
    p = int(np.argmax(diagonal))
    assert diagonal[p] > 0
    score = diagonal - G[:, p] ** 2 / diagonal[p]
    score[p] = -np.inf
    current = np.sort(np.argsort(-diagonal, kind='stable')[:384])
    proposed = np.sort(np.r_[p, np.argsort(-score, kind='stable')[:383]])
    assert len(np.unique(proposed)) == 384
    methods = {}
    for name, selected in [('top_diagonal', current), ('one_pivot_residual', proposed)]:
        Q, _ = np.linalg.qr(G[:, selected], mode='reduced')
        A = np.eye(n) - Q @ Q.T
        C = A @ G @ A.T
        methods[name] = {
            'residual_energy': float(np.trace(C)),
            'residual_energy_fraction': float(np.trace(C) / np.trace(G)),
            'residual_covariance_fro_fraction': float(np.linalg.norm(C) / np.linalg.norm(G)),
            'pivot_Q_condition': float(np.linalg.cond(Q[selected, :])),
        }
    a, b = methods['top_diagonal'], methods['one_pivot_residual']
    row = {
        'capture': path.name,
        'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'J_shape': list(J.shape), 'old_rank': int(Qp.shape[1]) if Qp.size else 0,
        'first_pivot': p,
        'number_replaced': int(384 - len(np.intersect1d(current, proposed))),
        'selected_rows': proposed.tolist(),
        'methods': methods,
        'energy_change_pct': 100 * (b['residual_energy'] / a['residual_energy'] - 1),
        'covariance_fro_change_pct': 100 * (b['residual_covariance_fro_fraction'] / a['residual_covariance_fro_fraction'] - 1),
    }
    rows.append(row)
    print(path.name, row['number_replaced'], 'energy', round(row['energy_change_pct'], 5),
          'covfro', round(row['covariance_fro_change_pct'], 5), flush=True)

out = {
    'status': 'SAVED_GEOMETRY_ONLY', 'rows': rows,
    'mean_energy_change_pct': float(np.mean([x['energy_change_pct'] for x in rows])),
    'mean_covariance_fro_change_pct': float(np.mean([x['covariance_fro_change_pct'] for x in rows])),
    'energy_improved_count': sum(x['energy_change_pct'] < 0 for x in rows),
    'seconds': time.perf_counter() - t0,
    'max_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(D / 'RESULT_v1.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k:v for k,v in out.items() if k != 'rows'}, indent=2))
