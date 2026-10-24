def _jrt_reduce(Qwide, core, rank):
    """Keep the common256 prefix and a rank64 Ritz subspace in its tail."""
    if rank != 320 or int(Qwide.shape[1]) != int(core.shape[0]):
        raise RuntimeError('JRT middle-join rank contract')
    head = 256
    tail_rank = rank - head
    tail_core = core[head:, head:].astype(fnp.float64)
    values, vectors = fnp.linalg.eigh(tail_core)
    width = int(tail_core.shape[0])
    order = fnp.arange(width - 1, width - tail_rank - 1, -1)
    V64 = vectors[:, order]
    V = V64.astype(fnp.float32)
    Qtail = _shared_old384_mm(Qwide[:, head:], V)
    Qn = fnp.concatenate([Qwide[:, :head], Qtail], axis=1)
    cross = core[:head, head:].astype(fnp.float64) @ V64
    diagonal = fnp.diag(values[order])
    reduced = fnp.concatenate([
        fnp.concatenate([core[:head, :head].astype(fnp.float64), cross], axis=1),
        fnp.concatenate([cross.T, diagonal], axis=1)
    ], axis=0)
    return Qn, reduced.astype(fnp.float32)
