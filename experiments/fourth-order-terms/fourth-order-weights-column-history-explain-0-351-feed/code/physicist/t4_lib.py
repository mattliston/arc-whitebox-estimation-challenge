"""TAG opus-physicist T4 library: the in-place column feed operator (certified in t4_cp4_ceiling.py cert: forward against brute force and
adjoint identity to 1e-15). T = sum_r lam_r Sym(u_r, u_r, om_r); fwd(lam) = its distinct-index content pushed to the next (2,1) plane."""
import numpy as np, torch
T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))


def Pi_t(W, P):
    A = 3 * P.T - 2 * torch.diag(torch.diagonal(P)); wt = W.T; pushed = wt @ A
    res = (2 * ((pushed * wt) @ W) + (wt * wt) @ pushed.T) / 3
    return torch.diagonal_scatter(res, (pushed * wt * wt).sum(1))


class Op:
    def __init__(self, W, u, om):
        self.W, self.u, self.om = W, u, om; self.U = W.T @ u; self.Om = W.T @ om
    def plane(self, lam):
        return (self.u * self.u) @ (lam[:, None] * self.om.T) + 2 * (self.u * self.om) @ (lam[:, None] * self.u.T)
    def fwd(self, lam):
        X = (self.U * self.U) @ (lam[:, None] * self.Om.T) + 2 * (self.U * self.Om) @ (lam[:, None] * self.U.T)
        return X - Pi_t(self.W, self.plane(lam))
