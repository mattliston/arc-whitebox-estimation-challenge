"""Inert two-block basis helpers; executed inside the paid prediction only."""


def _krylov384_apply(J, weight, Qp, core, X):
    Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, X))
    if Qp is not None:
        Y = Y + _shared_old384_mm(
            Qp, _shared_old384_mm(core, _shared_old384_mm(Qp.T, X)))
    return Y


def _krylov384_basis(J, weight, Qp, core, Omega):
    if (len(J.shape) != 2 or len(weight.shape) != 1
            or len(Omega.shape) != 2 or int(J.shape[0]) < 384
            or int(J.shape[0]) > 1024 or int(J.shape[1]) > 2048
            or int(J.shape[1]) <= 0 or int(J.shape[1]) != int(weight.shape[0])
            or tuple(Omega.shape) != (int(J.shape[0]), 192)
            or str(J.dtype) != 'float32' or str(weight.dtype) != 'float32'
            or str(Omega.dtype) != 'float32'):
        raise RuntimeError('Krylov384 input contract')
    if (Qp is None) != (core is None):
        raise RuntimeError('Krylov384 old-core ownership')
    if Qp is not None:
        if (tuple(Qp.shape) != (int(J.shape[0]), 384)
                or tuple(core.shape) != (384, 384)
                or str(Qp.dtype) != 'float32' or str(core.dtype) != 'float32'):
            raise RuntimeError('Krylov384 old-core contract')
    Y1 = _krylov384_apply(J, weight, Qp, core, Omega)
    Q1, unused_R = fnp.linalg.qr(Y1, mode='reduced')
    del Y1, unused_R
    _shared_old384_finite(Q1, 'Krylov first block')
    Y2 = _krylov384_apply(J, weight, Qp, core, Q1)
    for _ in range(2):
        Y2 = Y2 - _shared_old384_mm(Q1, _shared_old384_mm(Q1.T, Y2))
    joined = fnp.concatenate([Q1, Y2], axis=1)
    del Q1, Y2
    Qn, unused_R = fnp.linalg.qr(joined, mode='reduced')
    del joined, unused_R
    return _shared_old384_finite(Qn, 'Krylov joint basis')
