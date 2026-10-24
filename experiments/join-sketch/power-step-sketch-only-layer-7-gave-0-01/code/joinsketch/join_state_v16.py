"""[opus-lead] why does the L7-only JSK (v51j) help rank 35 but hurt rank 200 after L10? Compare the L10 join under the two STATES:
today's (join<r>_L10.npz) and v51j's (j51_<r>_L10.npz): the content entering the join X = [Qp F, J], its exact rank-384 ceiling,
today's Nystrom basis on it (with the state's own core, with the exact core, with the core-free variant), and the plane error of the
OLD tier alone vs the NEW columns alone. f32 pipeline. usage: join_state_v16.py <today_L10.npz> <v51j_L10.npz>"""
import sys, numpy as np
def load(p):
    d = np.load(p); J = d['J'].astype(np.float32); lam = d['lam'].astype(np.float32); old = int(d['old_columns'])
    Qp = d['Qp'].astype(np.float32); core = d['core'].astype(np.float32); F = d['F'].astype(np.float32); lam_old = d['lam_old'].astype(np.float32)
    w32 = d['w32'].astype(np.float64); prev = d['previous']; return dict(J=J, lam=lam, old=old, Qp=Qp, core=core, F=F, lam_old=lam_old, WJ=w32[prev, :].T, Qn=d['Qn'].astype(np.float32))
r = 384
for name, p in (('today', sys.argv[1]), ('v51j (JSK L7)', sys.argv[2])):
    s = load(p); J, lam, Qp, core, F, lam_old, WJ = s['J'], s['lam'], s['Qp'], s['core'], s['F'], s['lam_old'], s['WJ']
    m, K = J.shape; old = s['old']; w = lam * lam
    Xold = (Qp @ F).astype(np.float64); X = np.hstack([Xold, J.astype(np.float64)]); L = np.concatenate([lam_old, lam]).astype(np.float64)
    def plane(Y, cols=None):
        Y = Y if cols is None else Y[:, cols]; l = L if cols is None else L[cols]; return (Y * Y * l[None, :]) @ Y.T
    P0 = plane(X); n0 = np.linalg.norm(P0); PP0 = plane(WJ @ X); np0 = np.linalg.norm(PP0)
    oc = np.arange(old); nc = np.arange(old, old + K); Po = plane(X, oc); Pn = plane(X, nc)
    def err(Q):
        Q = Q.astype(np.float64); R = Q @ (Q.T @ X)
        return (np.linalg.norm(plane(R) - P0) / n0, np.linalg.norm(plane(WJ @ R) - PP0) / np0, np.linalg.norm(plane(R, oc) - Po) / np.linalg.norm(Po), np.linalg.norm(plane(R, nc) - Pn) / np.linalg.norm(Pn))
    # exact ceiling
    U, sv, _ = np.linalg.svd(X * np.abs(L)[None, :], full_matrices=False)
    print('== %s: live %d, joined %d, old %d | old-tier plane share %.3f | Qp orthonormality |Qp^T Qp - I| = %.3f | Qp cond %.1f' % (
        name, m, K, old, np.linalg.norm(Po) / n0, np.linalg.norm(Qp.T.astype(np.float64) @ Qp - np.eye(r)), np.linalg.cond(Qp.astype(np.float64))), flush=True)
    print('   exact ceiling r=384: plane %.4f pushed %.4f | old %.4f new %.4f' % err(U[:, :r]))
    print('   dumped Qn (native):  plane %.4f pushed %.4f | old %.4f new %.4f' % err(s['Qn']))
    core_exact = ((F * (lam_old * lam_old)[None, :]) @ F.T).astype(np.float32)
    print('   core: trace ratio stored/exact %.3f, rel err %.3f' % (np.trace(core) / np.trace(core_exact), np.linalg.norm(core.astype(np.float64) - core_exact) / np.linalg.norm(core_exact)))
    for cname, c in (('stored core', core), ('exact core', core_exact)):
        dg = (J * J) @ w + ((Qp @ c) * Qp).sum(1); rows = np.sort(np.argsort(-dg.astype(np.float64), kind='stable')[:r])
        Y = J @ (w[:, None] * J[rows].T) + (Qp @ c) @ Qp[rows].T
        print('   Nystrom recomputed with %-11s: plane %.4f pushed %.4f | old %.4f new %.4f' % ((cname,) + err(np.linalg.qr(Y)[0])))
