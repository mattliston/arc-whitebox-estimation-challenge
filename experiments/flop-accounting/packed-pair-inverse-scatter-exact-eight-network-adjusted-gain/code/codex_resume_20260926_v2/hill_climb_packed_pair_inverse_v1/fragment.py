# One paid, shape-only packed-pair inverse per prediction; no response cache.
_PPI_RUN = [None]


def _ppi_make_inverse(idx, n, block):
    q = int(idx['count']) * block * block
    inverse = fnp.full((n, n), q, dtype=fnp.int32)
    order = fnp.arange(q, dtype=fnp.int32)
    fnp.put(inverse, idx['upper'], order)
    del order
    return inverse, inverse.T


def _ppi_get(st, idx, n, block):
    run = _PPI_RUN[0]
    if run is None or (n, block) != (1024, 128):
        raise RuntimeError('packed inverse prediction scope')
    if not run:
        inverse, transposed = _ppi_make_inverse(idx, n, block)
        run.update(st=st, idx=idx, inverse=inverse, transposed=transposed)
    if run['st'] is not st or run['idx'] is not idx:
        raise RuntimeError('packed inverse owner mismatch')
    return run['inverse'], run['transposed']


def _ppi_take(source, inverse, count, block):
    if str(source.dtype) != 'float32' or source.shape != inverse.shape:
        raise RuntimeError('packed inverse source shape/dtype')
    out = fnp.empty((count + 1, block, block), dtype=fnp.float32)
    fnp.put(out, inverse, source)
    return out[:count]


def _ppi_copy_p16(source, inverse, out, count, block, hold):
    scratch = hold(fnp.empty((count + 1, block, block), dtype=fnp.float32))
    valid = scratch[:count]
    for channel in range(3):
        hold(fnp.put(scratch, inverse, source[channel]))
        hold(fnp.copyto(out[channel], valid))


_PPI_PARENT_PREDICT = Estimator.predict


def _ppi_predict(self, mlp, budget):
    if _PPI_RUN[0] is not None:
        raise RuntimeError('nested packed inverse prediction')
    _PPI_RUN[0] = {}
    try:
        return _PPI_PARENT_PREDICT(self, mlp, budget)
    finally:
        _PPI_RUN[0].clear()
        _PPI_RUN[0] = None


Estimator.predict = _ppi_predict
