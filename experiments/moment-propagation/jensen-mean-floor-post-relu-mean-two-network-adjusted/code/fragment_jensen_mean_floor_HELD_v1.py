# HELD single Jensen mean-floor arm on current4e. No strength, fit, or alternative.
assert _ASTRA_FHM_ON is True and _ASTRA_IOP_ON is False
assert _astra_fhm_effective.__name__ == '_astra_fhm_effective'
_JMF_ORIGINAL_EFFECTIVE = _astra_fhm_effective

def _astra_fhm_effective(layer, mean, delta):
    assert _ASTRA_FHM_ACTIVE[0] and 1 <= int(layer) <= 14
    assert len(mean.shape) == 1 and mean.shape == delta.shape
    assert str(mean.dtype) == str(delta.dtype) == 'float64'
    hold = _E2Hold()
    # Jensen: E[ReLU(X)] >= max(E[X],0). Same active-layer mu already paid.
    mu = _E2_LAYER['mu']
    assert _PHYS_LAYER[0] == int(layer) and mu.shape == mean.shape and str(mu.dtype) == 'float64'
    lower = hold(hold(fnp.maximum(mu, 0.0)) - mean)
    effective = fnp.maximum(delta, lower)
    # Diagnostic is a charged O(n) comparison/reduction, not an unbilled host read.
    changed = int(hold(fnp.sum(hold(fnp.greater(effective, delta)))))
    _ASTRA_FHM_STATS['layers'].append(int(layer))
    _ASTRA_FHM_STATS['changed_rows'].append([int(layer), changed])
    return effective
