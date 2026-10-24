"""One fixed historical-factor oracle; no Estimator, truths, or native calls."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
ARCHIVE = O / 'engineer/a35_join/state/s39_rank0651.npz'
OUT = D / 'SAVED_PAIR_CORESET_ORACLE_v1.json'
assert not OUT.exists()
t0 = time.monotonic()
with np.load(ARCHIVE) as z:
    J = z['J_10'].astype(np.float64)
    lam = z['lam_10'].astype(np.float64)
    Qp = z['Qp_10'].astype(np.float64)
    core = z['core_10'].astype(np.float64)
    old = z['old_09'][z['prev_10']].astype(np.float64)
    lo = z['lamold_09'].astype(np.float64)
full = np.concatenate([old, J], axis=1)
coeff = np.r_[lo, lam]
breaks = np.r_[0, np.flatnonzero(np.sign(coeff[1:]) != np.sign(coeff[:-1])) + 1, len(coeff)]
blocks = []
for k in range(0, len(breaks) - 1, 2):
    a, b, c = map(int, breaks[k:k + 3])
    assert b-a == c-b and np.array_equal(coeff[a:b], -coeff[b:c])
    blocks.append((a, b, c))
diag = (J*J) @ (lam*lam) + np.sum((Qp @ core)*Qp, axis=1)
rows = np.sort(np.argsort(-diag, kind='stable')[:320])
Y = J @ ((lam*lam)[:, None] * J[rows].T) + Qp @ (core @ Qp[rows].T)
Q = np.linalg.qr(Y, mode='reduced')[0]
F = Q.T @ full
Fu = np.concatenate([.5*(F[:, a:b]+F[:, b:c]) for a,b,c in blocks], axis=1)
Fv = np.concatenate([.5*(F[:, a:b]-F[:, b:c]) for a,b,c in blocks], axis=1)
Fu[256:] = 0.0
A = np.ascontiguousarray(Fu + Fv, dtype=np.float32)
B = np.ascontiguousarray(Fu - Fv, dtype=np.float32)
w = np.concatenate([coeff[a:b] for a,b,c in blocks])
assert np.all(w > 0) and A.shape == B.shape == (320, 5888)
del J, lam, Qp, core, old, full, Q, F, Fu, Fv, Y

def gram(ai, bi, aj, bj):
    # Cross products are F32, subsequent cubes/cancellation/solve are F64.
    aa = (ai.T @ aj).astype(np.float64)
    ab = (ai.T @ bj).astype(np.float64)
    ba = (bi.T @ aj).astype(np.float64)
    bb = (bi.T @ bj).astype(np.float64)
    return aa**3 - ab**3 - ba**3 + bb**3

a2 = np.sum(A.astype(np.float64)**2, axis=0)
b2 = np.sum(B.astype(np.float64)**2, axis=0)
ab = np.sum(A.astype(np.float64)*B, axis=0)
diagonal = a2**3 + b2**3 - 2*ab**3
priority = w*w*diagonal
selected = np.sort(np.argsort(-priority, kind='stable')[:256])
cross = gram(A[:, selected], B[:, selected], A, B)
K = cross[:, selected]
K = .5*(K + K.T)
rhs = cross @ w
ridge = 1e-6 * np.mean(np.diag(K))
beta = np.linalg.solve(K + ridge*np.eye(256), rhs)

# Stream the full cubic target norm. No full P-by-P Gram is retained.
target_norm = 0.0
absolute_terms = 0.0
for start in range(0, len(w), 256):
    stop = min(start + 256, len(w))
    tile = gram(A[:, start:stop], B[:, start:stop], A, B)
    weighted = tile @ w
    target_norm += float(w[start:stop] @ weighted)
    absolute_terms += float(np.abs(w[start:stop]) @ (np.abs(tile) @ np.abs(w)))
    del tile, weighted
residual_norm = target_norm - 2*float(beta @ rhs) + float(beta @ K @ beta)

# Only a numerical diagnostic: F64 cross products of the already-selected set.
# Does not change the frozen F32-product fit or selection.
As, Bs = A[:, selected].astype(np.float64), B[:, selected].astype(np.float64)
K64 = (As.T@As)**3 - (As.T@Bs)**3 - (Bs.T@As)**3 + (Bs.T@Bs)**3
ev = np.linalg.eigvalsh(.5*(K64+K64.T))
out = {
    'status': 'HISTORICAL_TENSOR_PROXY_ONLY_NO_MODEL_OR_TRUTH',
    'archive': str(ARCHIVE),
    'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'frame': {'rank':320, 'common_rank':256, 'pairs':5888,
              'pair_blocks':[b-a for a,b,c in blocks], 'source_generation':'S39 native651 L10'},
    'fixed_rule': {'selected_pairs':256, 'priority':'lambda^2 * ||fplus^3-fminus^3||_F^2',
                   'tie':'stable source pair index', 'ridge_relative_mean_diagonal':1e-6,
                   'cross_product_dtype':'float32', 'cubic_difference_and_solve_dtype':'float64'},
    'target_tensor_norm_squared':target_norm,
    'residual_tensor_norm_squared':residual_norm,
    'relative_squared_tensor_error':residual_norm/target_norm,
    'retained_tensor_energy_fraction':1-residual_norm/target_norm,
    'target_signed_cancellation_ratio':target_norm/absolute_terms,
    'ridge_absolute':float(ridge),
    'selected_gram_eigenvalue_min_f64':float(ev[0]),
    'selected_gram_eigenvalue_max_f64':float(ev[-1]),
    'selected_gram_relative_f32_error':float(np.linalg.norm(K-K64)/np.linalg.norm(K64)),
    'fit_weight_norm_ratio':float(np.linalg.norm(beta)/np.linalg.norm(w[selected])),
    'negative_fit_weights':int(np.count_nonzero(beta < 0)),
    'elapsed_seconds':time.monotonic()-t0,
    'limitations':[
       'Historical incoming L10 factors, not current native predictions or true moments.',
       'Ftilde=pad(Fu256)+/-Fv320 matches common-coordinate truncation in this reconstructed frame.',
       'Tensor Frobenius error does not measure actual folded pair-law read error.',
       'Current old read also subtracts fm*(q*P+xq*partner). Full carried state alone does not preserve that term after read-only reweighting.',
       'No subsequent gate/weight transport, future join, or learned-head response measured.',
       'One fixed selection, ridge, and subset size; no sweep or native-call authority.'
    ]
}
OUT.write_text(json.dumps(out, indent=2)+'\n')
np.savez_compressed(D/'FIXED_FIT_v1.npz', selected=selected, beta=beta)
print(json.dumps(out, indent=2))
print('receipt_sha256', hashlib.sha256(OUT.read_bytes()).hexdigest())
