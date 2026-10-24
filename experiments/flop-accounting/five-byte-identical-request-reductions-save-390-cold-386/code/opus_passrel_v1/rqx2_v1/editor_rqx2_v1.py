"""[opus-rqx2] RQX extension v1 = identity reuse of views that the clean base re-makes from the SAME live object with the SAME key
(view probe 2026-10-16 on 517fb1d0, client-server: 248 within-network + 46 cross-network exact repeats on the warm net; this takes the
safe ones). A reused view aliases exactly the memory a fresh one would, so every value read or written through it is unchanged: no
arithmetic, operand, order or dtype changes (billing-safe). Sites (LR2 body line numbers; the line count is unchanged):
  14844 _d96_views         the depth-7 strip view cache keeps 256 entries instead of 96 (it evicted live keys: 24 repeats a network)
    933 _nonlinear_pair_batch the unstack of the pair series' persistent result buffer (13-14 a network)
   7632 _gn_dcov           P16[0] of the persistent pair state (13)
   7643 _gn_dcov           X.T of the GN contraction's persistent output (13)
  14435 omitted_radial_source / 14436: the same row_energy[:columns] / column_sum[:columns] view as input and out= (8 + 8)
  18227 / 18239 _rtt_read  live_indices[:c15] made once per call (8)
Identity caches (_rqx2_*) hold the base object (so its id stays its own), are bounded, and are cleared at every predict.
usage (module): apply(text) -> text"""

EDITS = [
    (14844, "    if len(_D96_VIEWS) >= 96:", "    if len(_D96_VIEWS) >= 256:"),
    (933, "        _dm2_x0, _dm2_x1, _dm2_x2 = fnp.unstack(_e2_H(_psp_pair_series(st, n, _e2_H)))   # [opus-exact] dm2: no stack of the factors",
          "        _dm2_x0, _dm2_x1, _dm2_x2 = _rqx2_unstack(_e2_H(_psp_pair_series(st, n, _e2_H)))   # [opus-exact] dm2: no stack of the factors; [opus-rqx2] views reused"),
    (7632, "    rho = _g5st['P16'][0]   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1",
           "    rho = _rqx2_item(_g5st['P16'], 0)   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1 ([opus-rqx2] view reused)"),
    (7643, "    dcov = _e2_H(_e2_H(X + _e2_H(X.T)) * pr['ss32'])",
           "    dcov = _e2_H(_e2_H(X + _rqx2_T(X)) * pr['ss32'])"),
    (14435, "        _ott_h0 = fnp.add(row_energy[:columns], delta_energy, out=row_energy[:columns])",
            "        _rqx2_re = row_energy[:columns]; _ott_h0 = fnp.add(_rqx2_re, delta_energy, out=_rqx2_re)"),
    (14436, "        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])",
            "        _rqx2_cs = column_sum[:columns]; _ott_h1 = fnp.add(_rqx2_cs, delta_sum, out=_rqx2_cs)"),
    (18227, "        prefix=hold(_RTT_PARENT_READ(left,hold(young_right[:c15]),lam,hold(live_indices[:c15]),state))",
            "        _rqx2_li=hold(live_indices[:c15]);prefix=hold(_RTT_PARENT_READ(left,hold(young_right[:c15]),lam,_rqx2_li,state))"),
    (18239, "                    prefix_old=hold(_xsg_rows(5119,state['pre_basis'],hold(live_indices[:c15])))",
            "                    prefix_old=hold(_xsg_rows(5119,state['pre_basis'],_rqx2_li))"),
]

HELPERS = '''

# BEGIN [opus-rqx2] RQX extension v1 helpers (2026-10-16; append LAST): identity-checked reuse of views of the same live object.
_RQX2_C = {}
_RQX2_CAP = 48
_RQX2_STATS = dict(hits=0, misses=0)


def _rqx2_cached(owner, key, make):
    k = (id(owner), key)
    got = _RQX2_C.get(k)
    if got is not None and got[0] is owner:
        _RQX2_STATS['hits'] += 1
        return got[1]
    value = make()
    if len(_RQX2_C) >= _RQX2_CAP:
        _RQX2_C.pop(next(iter(_RQX2_C)))
    _RQX2_C[k] = (owner, value)
    _RQX2_STATS['misses'] += 1
    return value


def _rqx2_unstack(x):
    return _rqx2_cached(x, 'unstack', lambda: fnp.unstack(x))


def _rqx2_item(x, i):
    return _rqx2_cached(x, ('item', i), lambda: x[i])


def _rqx2_T(x):
    return _rqx2_cached(x, 'T', lambda: x.T)


_RQX2_PARENT_PREDICT = Estimator.predict


def _rqx2_predict(self, *args, **kwargs):
    _RQX2_C.clear()
    try:
        return _RQX2_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _RQX2_C.clear()


Estimator.predict = _rqx2_predict
# END [opus-rqx2] RQX extension v1 helpers
'''


def apply(text):
    L = text.split('\n')
    for n, old, new in EDITS:
        assert L[n - 1] == old, (n, L[n - 1][:100])
        L[n - 1] = new
    return '\n'.join(L) + HELPERS


if __name__ == '__main__':
    import hashlib, sys
    from pathlib import Path
    s = apply(Path(sys.argv[1]).read_text())
    compile(s, 'rqx2', 'exec')
    with open(sys.argv[2], 'x') as f:
        f.write(s)
    print(hashlib.sha256(s.encode()).hexdigest()[:8])
