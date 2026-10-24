
# Fixed local common-factor coordinate selection; all carried factors and
# difference coordinates remain unchanged. This is a coefficient-energy
# surrogate, not a claim of optimal cubic or final-mean accuracy.
_UCE_ORIGINAL_SPLIT = _uvp_split
_UCE_ORIGINAL_RECON = _uvp_recon
_UCE_ORIGINAL_INNER = _uvp_inner


def _uvp_split(F, blocks):
    rank = int(F.shape[0])
    ru = min(_UVP_RU, rank)
    if rank <= ru:
        return _UCE_ORIGINAL_SPLIT(F, blocks) + (None,)
    P = sum(h for _, h in blocks)
    fullFu = fnp.empty((rank, P), dtype=fnp.float32)
    Fv2 = fnp.empty((rank, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(F[:, a:a+h], F[:, a+h:a+2*h], out=fullFu[:, b:b+h]))
        h_(fnp.subtract(F[:, a:a+h], F[:, a+h:a+2*h], out=Fv2[:, b:b+h]))
        b += h
    h_(fnp.multiply(fullFu, 0.5, out=fullFu))
    h_(fnp.multiply(Fv2, 0.5, out=Fv2))
    energy = fnp.sum(fullFu * fullFu, axis=1)
    indices = fnp.sort(fnp.argsort(-energy, kind='stable')[:ru])
    Fu = fullFu[indices, :]
    _BH_HOLD.clear()
    _UVP_STATS['split'] += 1
    return (Fu, Fv2, blocks, ru, indices)


def _uvp_recon(U, uv):
    Fu, Fv2, blocks, ru, indices = uv
    if indices is None:
        return _UCE_ORIGINAL_RECON(U, uv[:4])
    rows = int(U.shape[0])
    Pu = _shared_old384_mm(U[:, indices], Fu)
    Pv = _shared_old384_mm(U, Fv2)
    out = fnp.empty((rows, 2*int(Fu.shape[1])), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(Pu[:, b:b+h], Pv[:, b:b+h], out=out[:, a:a+h]))
        h_(fnp.subtract(Pu[:, b:b+h], Pv[:, b:b+h], out=out[:, a+h:a+2*h]))
        b += h
    _BH_HOLD.clear()
    return out


def _uvp_inner(left_sq, lam_cols, uv):
    Fu, Fv2, blocks, ru, indices = uv
    if indices is None:
        return _UCE_ORIGINAL_INNER(left_sq, lam_cols, uv[:4])
    rows = int(left_sq.shape[0])
    rank = int(Fv2.shape[0])
    P = int(Fu.shape[1])
    weighted = left_sq * lam_cols[None, :]
    Su = fnp.empty((rows, P), dtype=fnp.float32)
    Dv = fnp.empty((rows, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(weighted[:, a:a+h], weighted[:, a+h:a+2*h], out=Su[:, b:b+h]))
        h_(fnp.subtract(weighted[:, a:a+h], weighted[:, a+h:a+2*h], out=Dv[:, b:b+h]))
        b += h
    _BH_HOLD.clear()
    del weighted
    inner = _shared_old384_mm(Dv, Fv2.T)
    inner_u = _shared_old384_mm(Su, Fu.T)
    positions = (fnp.arange(rows, dtype=fnp.int64)[:, None]*rank + indices[None, :]).reshape(-1)
    values = (inner[:, indices] + inner_u).reshape(-1)
    fnp.put(inner, positions, values)
    return inner
