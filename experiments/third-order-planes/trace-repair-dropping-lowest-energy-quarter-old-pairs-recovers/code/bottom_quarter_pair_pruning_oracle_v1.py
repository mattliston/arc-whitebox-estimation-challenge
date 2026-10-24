"""One fixed historical-factor oracle; no Estimator, truths, or native calls."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
ARCHIVE = O / 'engineer/a35_join/state/s39_rank0651.npz'
OUT = D / 'BOTTOM_QUARTER_PAIR_PRUNING_ORACLE_v1.json'
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

dropped = np.sort(np.argsort(priority, kind='stable')[:len(w)//4])
assert len(dropped) == 1472
Ad, Bd, wd = A[:,dropped], B[:,dropped], w[dropped]
omitted_norm = 0.0
for start in range(0,len(wd),256):
    stop = min(start+256,len(wd))
    tile = gram(Ad[:,start:stop],Bd[:,start:stop],Ad,Bd)
    omitted_norm += float(wd[start:stop] @ (tile @ wd))
reference = D/'SAVED_PAIR_CORESET_ORACLE_v1.json'
ref = json.loads(reference.read_text())
target_norm = ref['target_tensor_norm_squared']
assert ref['archive_sha256'] == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
out = dict(status='FIXED_BOTTOM_QUARTER_HISTORICAL_TENSOR_ORACLE_NO_MODELS',
    archive=str(ARCHIVE), archive_sha256=ref['archive_sha256'],
    full_tensor_reference=str(reference),full_tensor_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rule='Drop bottom25% pairs by lambda^2*||fplus^3-fminus^3||_F^2; original signed lambdas unchanged; stable index ties',
    frame=ref['frame'], dropped_pairs=len(dropped),retained_pairs=len(w)-len(dropped),
    omitted_tensor_squared_norm=omitted_norm,full_tensor_squared_norm=target_norm,
    relative_squared_tensor_error=omitted_norm/target_norm,
    relative_tensor_frobenius_error=float(np.sqrt(omitted_norm/target_norm)),
    fraction_sum_individual_squared_norm=float(priority[dropped].sum()/priority.sum()),
    original_signed_weights_preserved=True,elapsed_seconds=time.monotonic()-t0,
    history='Nearby engineer/a61_prune closed YOUNG birth/read pruning from flat/cancelling energies; no exact OLD read-only pair-pruning arm found in bounded history.',
    limitations=['Historical S39 native651 L10 effective common256/difference320 frame only; not current output/truth.',
      'Exact paired cubic-Gram identity with F32 cross products and F64 cubes, not sum-of-independent-energy approximation.',
      'Folded old NP q/xq term and later U/gate transport not tested.',
      'One fixed fraction; no fitting, Gram solve, parameter sweep, or native model.'])
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
print('receipt_sha256',hashlib.sha256(OUT.read_bytes()).hexdigest())
