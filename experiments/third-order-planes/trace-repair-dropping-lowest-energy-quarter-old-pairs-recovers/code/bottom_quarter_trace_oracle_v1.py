"""One fixed historical-factor oracle; no Estimator, truths, or native calls."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
ARCHIVE = O / 'engineer/a35_join/state/s39_rank0651.npz'
OUT = D / 'BOTTOM_QUARTER_TRACE_ORACLE_v1.json'
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
Ad, Bd, wd = A[:,dropped].astype(np.float64), B[:,dropped].astype(np.float64), w[dropped]
v = Ad @ (wd*np.sum(Ad*Ad,axis=0)) - Bd @ (wd*np.sum(Bd*Bd,axis=0))
r = A.shape[0]
trace_energy = 3*float(v@v)/(r+2)
p = D/'BOTTOM_QUARTER_PAIR_PRUNING_ORACLE_v1.json'
a = json.loads(p.read_text())
omitted = a['omitted_tensor_squared_norm'];fullnorm=a['full_tensor_squared_norm']
out=dict(status='FIXED_OMITTED_TENSOR_TRACE_PROJECTION_ONLY',
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 parent_oracle_path=str(p),parent_oracle_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 rank=r,dropped_pairs=len(dropped),trace_vector_squared_norm=float(v@v),
 omitted_tensor_squared_norm=omitted,trace_projection_squared_norm=trace_energy,
 omitted_energy_fraction_removable=trace_energy/omitted,
 corrected_relative_squared_tensor_error=(omitted-trace_energy)/fullnorm,
 corrected_relative_frobenius_error=float(np.sqrt((omitted-trace_energy)/fullnorm)),
 formula='v=sum_drop lambda*(||fplus||²fplus-||fminus||²fminus); ||Tiso||²=3||v||²/(r+2)',
 read_action='[2u(u dot v)+||u||²v]/(r+2)',
 actual_models=0,elapsed_seconds=time.monotonic()-t0,
 caveats=['Effective Fu256/Fv320 historical native651 orthonormal join frame, not current output/truth.',
 'Omitted tensor norm uses prior exact paired cubic-Gram identity evaluated with F32 dot products; trace vector uses F64 accumulation.',
 'Trace compensation applies raw cubic old-read term only. Folded fm*(qP+xqPswap) remains separately changed by pair pruning.',
 'TRACE321 changed carried join basis/rank; this test leaves carry intact and changes only the pruned read, so it is not the same mechanism.'])
OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('receipt_sha256',hashlib.sha256(OUT.read_bytes()).hexdigest())
