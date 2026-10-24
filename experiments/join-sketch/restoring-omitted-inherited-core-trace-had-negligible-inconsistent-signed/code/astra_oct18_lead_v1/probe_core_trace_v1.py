"""Saved-array audit of the JSK-lite v6 inherited trace omission.

No endpoint labels, no fitted scalar, no SDK. The counterfactual L13 comparison
only alters inherited Gram scale; it does not claim full native propagation.
"""
import gc
import json
import resource
import time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
LAB = D.parent / 'opus_fix336687_v1/joinsketch'
start = time.monotonic()
rows = []
for rank in (35, 200):
    z = np.load(LAB / f'join{rank}_L10.npz')
    J, lam, Qp, C = (z[k].astype('f') for k in ('J', 'lam', 'Qp', 'core'))
    w = lam * lam
    r = z['Qn'].shape[1]
    old_diag = np.sum((Qp @ C) * Qp, axis=1)
    diag = (J * J) @ w + old_diag
    ri = np.sort(np.argsort(-diag.astype('d'), kind='stable')[:r])
    ci = np.argsort(-(np.sum(J * J, axis=0) * w).astype('d'), kind='stable')[:1536]
    Y = J[:, ci] @ (w[ci, None] * J[ri][:, ci].T) + (Qp @ C) @ Qp[ri].T
    Q, R = np.linalg.qr(Y)
    rawcore = np.linalg.solve(Q[ri], R.T)
    rawcore = (rawcore + rawcore.T) * .5
    Fj = Q.T @ J
    new = (Fj * w) @ Fj.T
    overlap = Q.T @ Qp
    old = (overlap @ C) @ overlap.T
    full = new + old
    newtrace, oldtrace = float(np.trace(new)), float(np.trace(old))
    norm = float(np.linalg.norm(full.astype('d')))
    scales = dict(current=newtrace, corrected_projected=newtrace + oldtrace,
                  corrected_input=newtrace + float(old_diag.sum()))
    errors = {name: float(np.linalg.norm((rawcore * (trace / np.trace(rawcore)) - full).astype('d'))) / norm
              for name, trace in scales.items()}
    row = dict(rank=rank, L10_rank=int(r), inherited_fraction=oldtrace / (newtrace + oldtrace),
               input_vs_projected_old_trace_ratio=float(old_diag.sum()) / oldtrace,
               core_relative_errors=errors)
    missing_fraction = newtrace / (newtrace + oldtrace)
    del z, J, lam, Qp, C, w, old_diag, diag, ri, ci, Y, Q, R, rawcore, Fj, new, overlap, old, full
    gc.collect()
    z = np.load(LAB / f'join{rank}_L13.npz')
    J, lam, Qp, C, F = (z[k].astype('f') for k in ('J', 'lam', 'Qp', 'core', 'F'))
    w = lam * lam
    X = np.concatenate((Qp @ F, J), axis=1)
    L = np.concatenate((z['lam_old'].astype('f'), lam))
    Ws = z['w32'][z['previous']].T[np.arange(0, 1024, 16)].astype('f')
    actual = Ws @ X
    truth = (actual * actual * L) @ actual.T
    np.fill_diagonal(truth, 0)
    ci = np.argsort(-(np.sum(J * J, axis=0) * w).astype('d'), kind='stable')[:1536]
    arms = {}
    for name, scale in [('omitted_trace_proxy', missing_fraction), ('full_trace', 1.)]:
        oldgram = (Qp @ C) * np.float32(scale)
        diag = (J * J) @ w + np.sum(oldgram * Qp, axis=1)
        ri = np.sort(np.argsort(-diag.astype('d'), kind='stable')[:384])
        Y = J[:, ci] @ (w[ci, None] * J[ri][:, ci].T) + oldgram @ Qp[ri].T
        Q = np.linalg.qr(Y)[0]
        projected = (Ws @ Q) @ (Q.T @ X)
        plane = (projected * projected * L) @ projected.T
        np.fill_diagonal(plane, 0)
        arms[name] = float(np.sum((plane.astype('d') - truth) ** 2))
    row['L13_saved_CP_errors'] = arms
    row['L13_CP_error_ratio'] = arms['full_trace'] / arms['omitted_trace_proxy']
    rows.append(row)
    del z, J, lam, Qp, C, F, X, L, Ws, actual, truth, ci, oldgram, diag, ri, Y, Q, projected, plane
    gc.collect()
result = dict(rows=rows, no_truths=True, no_sdk=True,
              caveat='Historical pre-JSK saved states; L13 scale counterfactual does not propagate a new basis.',
              elapsed_s=time.monotonic()-start, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (D / 'CORE_TRACE_SAVED_RESULT_v1.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
