# BEGIN OPUS-ENGINEER REQUEST DIET OF THE PAIR SERIES (a56 rqd v1, 2026-10-11; lead 03:5xZ request census; predictions byte-identical,
# bill identical). The pair map's packed series (_psp_series, _psp_pair_series) and its packed-inverse helpers (_ppi_take,
# _ppi_copy_p16) ask the same views of the same PERSISTENT arrays (the pair arena's views from _psp_views, the pair state st) on every
# layer, and allocate the same scratch shapes: about 500 requests per network (a56_req/probe_repeat.py on S46: 489 repeated views, 84
# fixed-shape allocations in this family). Here those views are made once per owner object (a cache keyed by the owner, which it
# holds) and the scratch arrays are persistent per (source, index) pair: every element of a persistent output is written by the same
# put before it is read, exactly as the fresh one was. The arithmetic, its operands and its order are unchanged.
_RQD_CACHE = {}
_RQD_CAP = 256
_RQD_STATS = dict(hits=0, misses=0)


def _rqd_get(tag, owner, make):
    key = (tag, id(owner))
    got = _RQD_CACHE.get(key)
    if got is not None and got[0] is owner:
        _RQD_STATS['hits'] += 1
        return got[1]
    value = make()
    if len(_RQD_CACHE) >= _RQD_CAP:
        _RQD_CACHE.pop(next(iter(_RQD_CACHE)))
    _RQD_CACHE[key] = (owner, value)
    _RQD_STATS['misses'] += 1
    return value


def _psp_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
    Z = a['Z']
    z0, z1, z2, z3, z4 = _rqd_get('Z5', Z, lambda: fnp.unstack(Z))
    X = a['X1']
    if persistent_fg:
        F13, G13 = _rqd_get('F13', F, lambda: F[1:4]), _rqd_get('G13', G, lambda: G[1:4])
    else:
        F13, G13 = F[1:4], G[1:4]
    hold(fnp.multiply(F13, G13, out=X))
    hold(fnp.multiply(X, P['P6'], out=X))
    hold(fnp.sum(X, axis=0, out=z0))
    X = a['X2']
    hold(fnp.multiply(F[[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]],
                      G[[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
    hold(fnp.multiply(a['X2r1'], a['RR1'], out=a['X2r1']))
    T = a['T']
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3, t4, t5 = _rqd_get('T5', T, lambda: fnp.unstack(T))
    X = a['X3']
    hold(fnp.multiply(F5[[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]],
                      G5[[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]], out=X))
    hold(fnp.multiply(X, P['P12'], out=X))
    U = a['U']
    hold(fnp.sum(X, axis=1, out=U))
    t6, t7, t8, t9 = _rqd_get('U4', U, lambda: fnp.unstack(U))
    hold(fnp.multiply(t21, t1, out=z1)); hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    hold(fnp.multiply(t31, t4, out=z3)); hold(fnp.add(z3, hold(t13 * t5), out=z3))
    hold(fnp.add(z3, hold(t6 + t7), out=z3))
    hold(fnp.add(t8, t9, out=z4))
    Z15 = _rqd_get('Z15', Z, lambda: Z[1:5])
    hold(fnp.divide(Z15, a['div'], out=Z15))
    return fnp.sum(Z, axis=0)


def _psp_pair_series(st, n, hold, block=128):
    v = _psp_views(st, n, block)
    idx = v['idx']
    # BA remains full, with the original reduction axes and arithmetic order.
    ba = hold(_psp_series(v['Fba'], v['Gba'], v['F5ba'], v['G5ba'], st,
                         st['T21'], st['T21T'], st['T22'], st['T31'], st['T31T'], v['ba'], hold, persistent_fg=True))
    Zv = st['Zv']
    raw = _rqd_get('Zv0', Zv, lambda: Zv[0])
    raw0, raw1, raw2 = _rqd_get('raw3', raw, lambda: (raw[0], raw[1], raw[2]))
    hold(fnp.copyto(raw1, ba[0]))
    # Gather current values every invocation; only shape views/indices persist.
    inverse, transposed = _ppi_get(st, idx, n, block)
    _ppi_copy_p16(st['P16'], inverse, v['p16out'], idx['count'], block, hold)
    f5 = hold(fnp.take(v['Fsrc'], idx['rows'], axis=3))[:, :, :, :, :, None]
    g5 = hold(fnp.take(v['Gsrc'], idx['cols'], axis=3))[:, :, :, :, None, :]
    t21 = hold(_ppi_take_slot(st['T21'], inverse, idx['count'], block, 't21'))
    t12 = hold(_ppi_take_slot(st['T21'], transposed, idx['count'], block, 't12'))
    t22 = hold(_ppi_take_slot(st['T22'], inverse, idx['count'], block, 't22'))
    t31 = hold(_ppi_take_slot(st['T31'], inverse, idx['count'], block, 't31'))
    t13 = hold(_ppi_take_slot(st['T31'], transposed, idx['count'], block, 't13'))
    packed = hold(_psp_series(f5[0], g5[0], f5, g5, v['P'], t21, t12, t22, t31, t13, v['packed'], hold))
    for target, values in zip((raw0, raw2), fnp.unstack(packed)):
        fnp.put(target, idx['upper'], values)
        fnp.put(target, idx['lower'], values[idx['diagonal_blocks']:])
    return raw


def _ppi_take_slot(source, inverse, count, block, slot):
    # = _ppi_take into a persistent output per (source, call slot): the put writes every element of out[:count] each call
    if str(source.dtype) != 'float32' or source.shape != inverse.shape:
        raise RuntimeError('packed inverse source shape/dtype')
    out, valid = _rqd_get(('take', slot, count, block), source,
                          lambda: (lambda o: (o, o[:count]))(fnp.empty((count + 1, block, block), dtype=fnp.float32)))
    fnp.put(out, inverse, source)
    return valid


def _ppi_copy_p16(source, inverse, out, count, block, hold):
    scratch, valid = _rqd_get(('p16', count, block), source,
                              lambda: (lambda s: (s, s[:count]))(fnp.empty((count + 1, block, block), dtype=fnp.float32)))
    sources = _rqd_get('p16src', source, lambda: tuple(source[c] for c in range(3)))
    outs = _rqd_get('p16out', out, lambda: tuple(out[c] for c in range(3)))
    for channel in range(3):
        hold(fnp.put(scratch, inverse, sources[channel]))
        hold(fnp.copyto(outs[channel], valid))


_PB_CACHES = tuple(_PB_CACHES) + ('_RQD_CACHE',)
# END OPUS-ENGINEER REQUEST DIET OF THE PAIR SERIES
