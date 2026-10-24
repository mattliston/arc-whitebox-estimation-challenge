# BEGIN OPUS-ENGINEER PER-BAND HEAD PARAMETERS (a49 band v1, 2026-10-10; OFF unless given: then bit-identical). The trainer's jmE arms
# (lead 09:07Z): the learned mean correction's first-layer bias enc0_b by band of layers, and the head columns' first-layer weights u by
# band. A band is a range of layers (_BAND_EDGES); a layer outside every band keeps the shared arrays.
#   _ENC0_B_BAND  None, or one 128-vector per band (the bias the first layer uses there, in place of the shared enc0_b)
#   _HC_U_BAND    None, or one K x 128 matrix per band for the columns of _HC_COLUMNS (fragment hc v2), in place of the shared _HC_U;
#                 None for a band = the shared weights there
# Native: a different constant array per layer, picked in Python and made once per process: no operation and no request per layer.
# The final layer's own columns (fragment hcf) add to whatever bias layer 15 gets here.
_BAND_EDGES = ((1, 9), (10, 14), (15, 15))
_ENC0_B_BAND = None
_HC_U_BAND = None
_BAND_CONST = {}


def _band_of(l):
    for i, (lo, hi) in enumerate(_BAND_EDGES):
        if lo <= l <= hi:
            return i
    return None


_BAND_PARENT_HC_BIAS = _hc_bias


def _hc_bias(l, bias):
    i = _band_of(l)
    if i is not None and _ENC0_B_BAND is not None and _ENC0_B_BAND[i] is not None:
        key = ('b', i)
        if key not in _BAND_CONST:
            _BAND_CONST[key] = fnp.asarray(_ENC0_B_BAND[i], dtype=fnp.float32)
        bias = _BAND_CONST[key]
    if _HC_U_BAND is not None and _HC_U is not None:
        if 'shared' not in _BAND_CONST:
            _BAND_CONST['shared'] = fnp.asarray(_HC_U if len(_HC_COLUMNS) > 1 else _HC_U[0], dtype=fnp.float32)   # several columns (fragment hc v2) only
        if i is not None and _HC_U_BAND[i] is not None:
            key = ('u', i)
            if key not in _BAND_CONST:
                _BAND_CONST[key] = fnp.asarray(_HC_U_BAND[i] if len(_HC_COLUMNS) > 1 else _HC_U_BAND[i][0], dtype=fnp.float32)
            _HC_CONST['u'] = _BAND_CONST[key]
        else:
            _HC_CONST['u'] = _BAND_CONST['shared']
    return _BAND_PARENT_HC_BIAS(l, bias)
# END OPUS-ENGINEER PER-BAND HEAD PARAMETERS
