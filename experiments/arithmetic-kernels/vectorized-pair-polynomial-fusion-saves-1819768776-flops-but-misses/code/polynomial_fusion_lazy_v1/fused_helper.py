def _pair_series_fused(F, G, rho, T21, T22, T31, t3, t4, n):
    """Vectorized linear polynomial fusion; symmetric-rho real identity."""
    if T31 is not None:
        raise RuntimeError('Polynomial fusion does not take a pre-rounded T31')
    P = fnp.cumprod(fnp.broadcast_to(rho[None, :, :], (6, n, n)) / fnp.arange(1, 7, dtype=fnp.float32)[:, None, None], axis=0)
    FF = F.transpose(1, 0, 2)
    GG = G.transpose(1, 0, 2)
    kk = fnp.asarray([4, 8], dtype=fnp.float32)[:, None, None]
    df12 = t3[None, None, :] * FF[4:6] / 6 + t4[None, None, :] * (kk * FF[3:5] + FF[5:7]) / 24
    dg12 = t3[None, None, :] * GG[4:6] / 6 + t4[None, None, :] * (kk * GG[3:5] + GG[5:7]) / 24
    DF = fnp.concatenate((df12, (t4[None, None, :] * FF[5:6] / 2)), axis=0)
    DG = fnp.concatenate((dg12, (t4[None, None, :] * GG[5:6] / 2)), axis=0)
    ans = fnp.sum(P[:3, None] * ((FF[1:4] + DF)[:, :, :, None] * GG[1:4, :, None, :] + FF[1:4, :, :, None] * DG[:, :, None, :]), axis=0)
    ans = ans + fnp.sum(P[3:, None] * (FF[4:7, :, :, None] * GG[4:7, :, None, :]), axis=0)
    Fr = FF[:, :, :, None]
    Gr = GG[:, :, None, :]
    RR = fnp.stack((fnp.ones_like(rho), P[0], P[1]), axis=0)[:, None]
    def tri(A, B, R):
        return fnp.sum(R * (A * B), axis=0)
    e21 = T21[None] * tri(Fr[2:5], Gr[1:4], RR) + T21.T[None] * tri(Fr[1:4], Gr[2:5], RR)
    e22 = T22[None] * tri(Fr[2:5], Gr[2:5], RR)
    return ans + e21 / 2 + e22 / 4
