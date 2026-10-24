_RE23_ON = True
_RE23_ACTIVE = [False]
_RE23_PENDING = [None]
_RE23_STATS = dict(reads=[], applied=[], retired=True)
_RE23_PARENT_PREDICT = Estimator.predict

def _re23_publish(layer, left, young_right, lam, live_indices, state, rows, Q, scores, weighted, c15, hold):
    if not (_RE23_ON and _RE23_ACTIVE[0] and 7 <= layer <= 14):
        return
    assert _RE23_PENDING[0] is None
    old = int(state['columns']); k = int(young_right.shape[1])
    blocks = _uvp_blocks(state, old, old+k)
    A = hold(left[:,old:])
    if weighted is None:
        weighted = hold(hold(young_right[c15:]) * hold(lam[old:]))
    cols = hold(live_indices[c15:])
    sigma = _E2_LAYER['s']
    variance = hold(sigma*sigma)
    energy = _re23_energy(A,Q,scores,weighted,rows,cols,variance,blocks,hold)
    _RE23_PENDING[0] = (layer,cols,energy)
    _RE23_STATS['reads'].append(dict(layer=layer,m=int(A.shape[0]),k=k,tail=int(cols.shape[0]),blocks=len(blocks)))

def _re23_add(layer, squared_norm, hold):
    found = _RE23_PENDING[0]
    if not (_RE23_ON and _RE23_ACTIVE[0]) or found is None:
        return squared_norm
    expected, cols, energy = found
    assert int(layer)==expected and 7<=int(layer)<=14
    # squared_norm is the fresh parent reduction, exclusively owned here.
    # Keep its dtype and all non-tail entries exactly unchanged.
    squared_norm[cols] = hold(hold(squared_norm[cols]) + energy)
    _RE23_PENDING[0] = None
    _RE23_STATS['applied'].append(int(layer))
    return squared_norm

def _re23_predict(self,*args,**kwargs):
    assert not _RE23_ACTIVE[0] and _RE23_PENDING[0] is None
    _RE23_STATS.clear();_RE23_STATS.update(reads=[],applied=[],retired=False)
    _RE23_ACTIVE[0]=bool(_RE23_ON)
    try:
        answer=_RE23_PARENT_PREDICT(self,*args,**kwargs)
        assert _RE23_PENDING[0] is None
        return answer
    finally:
        _RE23_ACTIVE[0]=False;_RE23_PENDING[0]=None;_RE23_STATS['retired']=True

Estimator.predict=_re23_predict
