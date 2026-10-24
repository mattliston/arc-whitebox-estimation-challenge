"""Fixed rank8 correlated-message energy; only public metered operations."""


def correlated_energy(fnp, covariance, state, weights, parameters, pdf):
    n = int(weights.shape[0])
    channels, rank = 16, 8
    assert int(parameters['msg_w'].shape[0]) // 3 == channels
    c = covariance.astype(fnp.float32)
    scale = fnp.sqrt(fnp.maximum(fnp.diag(c), 1e-10))
    corr = (c + c.T) * .5
    fnp.fill_diagonal(corr, 0.)
    corr = corr / scale[:, None] / scale[None, :]
    fnp.fill_diagonal(corr, 1.)
    w = weights.astype(fnp.float32)
    q, unused = fnp.linalg.qr(corr @ w[:, :rank], mode='reduced')
    del unused
    q, unused = fnp.linalg.qr(corr @ q, mode='reduced')
    del unused
    core = q.T @ (corr @ q)
    core = (core + core.T) * .5
    source = state @ parameters['msg_w'][:channels].T + parameters['msg_b'][:channels]
    projected = w.T @ (q[:, :, None] * source[:, None, :]).reshape(n, rank*channels)
    projected = projected.reshape(n, rank, channels)
    full_energy = fnp.sum(projected * (core @ projected), axis=1)
    diagonal = fnp.sum((q @ core) * q, axis=1)
    source2 = source * source
    w2 = w * w
    independent = w2.T @ source2
    low_diagonal = w2.T @ (diagonal[:, None] * source2)
    off_ratio = (full_energy - low_diagonal) / fnp.maximum(independent, 1e-12)
    return (.5 * pdf.astype(fnp.float32)[:, None] * fnp.clip(off_ratio, -8, 8)).astype(fnp.float32)
