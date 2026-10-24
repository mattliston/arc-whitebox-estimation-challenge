"""TAG fable-physicist (RF6-18): VERSIONED COPY of fable_engineer_v1/lrstar_torch_hook.py (E-62c) with ONE switch in star():
S1 = 'zero' (default) drops the s1 term, sub = (((wt*wt) @ Br.T) @ P1.T) / 3 (engineer2: s1 = 0 beats the rank-32 s1; lead: loopR2 trains
with it); S1 = 'exact' restores the original line (bit-exact vs the engineer's hook). star_diagonal() and the diagonal are unchanged.
Original header follows:
TAG fable-engineer (E-62c): torch drop-ins for the S18 low-rank star plane, for refits that should train on the S18 physics.
S18's native fragment (lrstar2_k48_q1_r32.py) replaces the star plane A by its rank-32 part from a randomized SVD: Omega = the fixed
1024 x 48 float32 Gaussian drawn by numpy's default_rng(20260924) (the same draw the native flopscope server makes), one power step
with QR re-orthonormalization, then the leading 32 singular directions of Q^T A. Here the basis Q_r is computed on a DETACHED copy of
A (no gradient through QR or SVD, which is unstable on this flat spectrum) and the plane enters as Q_r (Q_r^T A): the forward value
equals the native projection, and gradients flow through A along the kept subspace.
usage: import lrstar_torch_hook as lh; lh.install(card, torch)   (after the card is built; replaces env['_flms_star'] and
env['_flms_star_diagonal'])"""
import numpy as np

K, Q_STEPS, R, SEED = 48, 1, 32, 20260924
S1 = 'zero'


def install(card, torch, k=K, q=Q_STEPS, r=R, seed=SEED):
    t = torch
    om_np = np.random.default_rng(seed).standard_normal((1024, k), dtype=np.float32)
    om = {}

    def basis(A32):
        with t.no_grad():
            Ad = A32.detach()
            o = om.get(Ad.dtype)
            if o is None:
                o = om[Ad.dtype] = t.from_numpy(om_np).to(Ad.dtype)
            Q = t.linalg.qr(Ad @ o)[0]
            for _ in range(q):
                Q = t.linalg.qr(Ad @ t.linalg.qr(Ad.T @ Q)[0])[0]
            U = t.linalg.svd(Q.T @ Ad, full_matrices=False)[0]
            return Q @ U[:, :r]

    def star(w, A):
        wt = w.T
        A32 = A if A.dtype == t.float32 else A.to(t.float32)
        Qr = basis(A32)
        Br = Qr.T @ A32
        P1 = wt @ Qr
        pw = (P1 @ Br) * wt
        sub = (2 * (pw @ w) + ((wt * wt) @ Br.T) @ P1.T) / 3 if S1 == 'exact' else (((wt * wt) @ Br.T) @ P1.T) / 3   # S1 = 0: engineer2 / lead
        diagonal = (pw * wt).sum(dim=1, dtype=t.float64)
        result = sub.to(t.float64).clone()
        result.diagonal().copy_(diagonal)
        return result

    def star_diagonal(w, A):
        wt = w.T
        A32 = A.to(t.float32)
        Qr = basis(A32)
        pushed = (wt @ Qr) @ (Qr.T @ A32)
        return ((pushed * wt) * wt).sum(dim=1, dtype=t.float64)

    card.env['_flms_star'] = star
    card.env['_flms_star_diagonal'] = star_diagonal
    return card
