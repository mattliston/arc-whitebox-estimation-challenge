"""TAG opus-physicist NTOR: ORACLE injection of the network-specific part R_l of the second-chaos matrix Ntilde (t19_oracle_rank*.npz) into
the torch card, for the closed-loop ceiling of a matrix carrier of the fourth order (lead's order of 09:01Z). Three uses, each with its
own strength, at layers 1-15 (R_l belongs to the pre-activations of layer l):
  k4    per-unit fourth cumulant  k4_i += (6/n) R_ii                              (pair map of layers 1-14 and ctx['k4']; the final layer's mean)
  k31   K31 extra plane (3/n) R_ij, i != j, through the pair map's remainder31     (layers 1-14; standardised as T31: / (sigma_i^3 sigma_j))
  feed  fourth-cumulant feed into births Q(i,j,k,k) = R_ij v_k / tr C at layer l, pushed with the weights of layer l+1 and ADDED TO K21 of
        layer l+1 (pair-map input only: one-layer effect, nothing is carried by the bank); at the final layer its diagonal goes to k3
rank = r > 0: R_l is replaced by its r eigen-directions of largest |eigenvalue| (what a rank-r carrier would hold); 0 = the full matrix.
install(cd, torch, R, k4=, k31=, feed=, rank=)   R: dict layer -> (n, n) array"""
import numpy as np


def install(cd, torch, R, k4=0.0, k31=0.0, feed=0.0, rank=0):
    t = torch; f64 = t.float64; n = 1024; st = dict(plane={}); Rr = {}; info = {}; cd.ntor_info = info
    for l, M in R.items():
        M = t.as_tensor(np.asarray(M, dtype=np.float64)); M = (M + M.T) / 2
        if rank:
            lam, V = t.linalg.eigh(M); o = lam.abs().argsort(descending=True)[:rank]; M = (V[:, o] * lam[o][None, :]) @ V[:, o].T
        Rr[l] = M
    old_pin = cd.hooks.get('pair_in')
    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        mu, Cpre, k21, k4v, k22, rem = pin
        if feed and l in st['plane']:
            pl_ = st['plane'].pop(l); info.setdefault(l, {})['feed_plane_over_k21'] = float(pl_.norm() / k21.norm()); k21 = k21 + feed * pl_
        if l in Rr:
            M = Rr[l]; s = t.diagonal(Cpre).clamp_min(1e-10).sqrt()
            if k4:
                dk = k4 * (6.0 / n) * t.diagonal(M); info.setdefault(l, {})['dk4_over_k4'] = float(dk.norm() / (k4v * t.ones(n, dtype=f64)).norm()); k4v = k4v + dk; ctx['k4'] = ctx['k4'] + dk
            if k31:
                rm = k31 * (3.0 / n) * (M - t.diag(t.diagonal(M))) / (s ** 3)[:, None] / s[None, :]
                info.setdefault(l, {})['remainder31_over_T31'] = float(rm.norm() / (((k4v * t.ones(n, dtype=f64)) / s ** 4)[:, None] * (Cpre - t.diag(t.diagonal(Cpre))) / t.outer(s, s)).norm())
                rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook
    if k4 and 15 in Rr:
        old4 = cd.hooks.get('k4_15')
        cd.hooks['k4_15'] = lambda l, v, ctx: (v if old4 is None else old4(l, v, ctx)) + k4 * (6.0 / n) * t.diagonal(Rr[15])
    if feed:
        old_out = cd.hooks.get('state_out')
        def out_hook(l, out, ctx):
            if old_out is not None: out = old_out(l, out, ctx)
            if l in Rr and l <= 14:
                aux = ctx['aux']; g = aux['gate'].to(f64); d = (aux['pdf'] / aux['sigma']).to(f64); v = (aux['sigma'] * aux['sigma']).to(f64); om = v / v.sum()
                W = cd.ws[l + 1]; G = g[:, None] * W; MG = G.T @ Rr[l] @ G
                st['plane'][l + 1] = t.outer(t.diagonal(MG), (om * d / 2) @ W) + MG * ((om * d) @ W)[:, None]      # full push of the symmetrised feed tensor (t2_lib.feed_push)
            return out
        cd.hooks['state_out'] = out_hook
        old3 = cd.hooks.get('k3_15')
        def k3_hook(l, k3, ctx):
            if old3 is not None: k3 = old3(l, k3, ctx)
            return k3 + feed * t.diagonal(st['plane'].pop(15)) if 15 in st['plane'] else k3
        cd.hooks['k3_15'] = k3_hook
    return cd
