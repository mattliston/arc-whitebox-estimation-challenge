"""[opus-lead] can the LIGHT columns' factors use fewer frame coordinates? Fj = Qn^T J costs 2 r m K; computing the light columns' factors
only on the leading h coordinates (Fj_light = Qn[:, :h]^T J_light) saves (1-h/r)*(K_light/K) of it. Plane error of the content with
light factors truncated to h coordinates: frame = today's (index-sorted rows) or energy-ordered rows (same span, different order).
usage: join_lightfac_v19.py <join_L.npz> kc"""
import sys, numpy as np
d = np.load(sys.argv[1]); kc = int(sys.argv[2]); r = 384
J = d['J'].astype(np.float32); lam = d['lam'].astype(np.float32); w = lam * lam; m, K = J.shape; old = int(d['old_columns'])
Qp = d['Qp'].astype(np.float32) if old else None; core = d['core'].astype(np.float32) if old else None; F = d['F'].astype(np.float32) if old else None
lam_old = d['lam_old'].astype(np.float32); w32 = d['w32'].astype(np.float64); prev = d['previous']
X = np.hstack([(Qp @ F) if old else np.zeros((m, 0), np.float32), J]).astype(np.float64); L = np.concatenate([lam_old, lam]).astype(np.float64); WJ = w32[prev, :].T
def plane(Y): return (Y * Y * L[None, :]) @ Y.T
P0 = plane(X); PP0 = plane(WJ @ X); n0 = np.linalg.norm(P0); np0 = np.linalg.norm(PP0)
dg = (J * J) @ w; Qc = Qp @ core if old else None
if old: dg = dg + (Qc * Qp).sum(1)
energy = np.argsort(-dg.astype(np.float64), kind='stable')
cn = ((J * J).sum(0) * w).astype(np.float64); order = np.argsort(-cn, kind='stable'); heavy = np.sort(order[:kc]); light = np.sort(order[kc:])
def frame(rows):
    J1 = J[:, heavy]; Y = J1 @ (w[heavy][:, None] * J1[rows].T)
    if old: Y = Y + Qc @ Qp[rows].T
    return np.linalg.qr(Y)[0].astype(np.float64)
def err_trunc(Q, h):
    Fj = Q.T @ X                                   # old columns + joined columns in the frame
    Fj[h:, old + light] = 0.0                      # light joined columns: only the leading h coordinates
    R = Q @ Fj; return np.linalg.norm(plane(R) - P0) / n0, np.linalg.norm(plane(WJ @ R) - PP0) / np0
print('layer', int(d['layer']), 'K', K, 'light cols', len(light), 'light energy share %.3f' % (cn[light].sum() / cn.sum()))
for name, rows in (('index-sorted rows (today)', np.sort(energy[:r])), ('energy-ordered rows', energy[:r])):
    Q = frame(rows); line = name + ':'
    for h in (384, 320, 256, 192, 128):
        line += ' | h=%d %.4f/%.4f' % ((h,) + err_trunc(Q, h))
    print(line, flush=True)
