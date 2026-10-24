# BEGIN ENGINEER EDBAND v1: conditional 16-channel edge bias/output constants.
# Select only AFTER _e2_edge_consts(shared_t,n): its R/S cache keys dictionary identity.
# Constants are fnp.asarray once at CS setup (or lazy if CS absent), never numpy.
# Full exported values, no runtime base+delta arithmetic. No new einsum signatures.
_EDBAND_EDGES = ()
_EDBAND_BIAS = ()
_EDBAND_OUTPUT = ()
_EDBAND_CONST = {}


def _edband_bind(shared):
    if not _EDBAND_EDGES:
        return
    if _EDBAND_CONST.get('shared') is shared:
        return
    _EDBAND_CONST.clear()
    for i, _edge in enumerate(_EDBAND_EDGES):
        values = dict(shared)
        if _EDBAND_BIAS[i] is not None:
            values['bias'] = fnp.asarray(_EDBAND_BIAS[i], dtype=fnp.float32)
        if _EDBAND_OUTPUT[i] is not None:
            values['output'] = fnp.asarray(_EDBAND_OUTPUT[i], dtype=fnp.float32)
        _EDBAND_CONST[i] = values
    _EDBAND_CONST['shared'] = shared


def _edge_band_parameters(layer, shared):
    for i, (lo, hi) in enumerate(_EDBAND_EDGES):
        if lo <= layer <= hi:
            _edband_bind(shared)
            return _EDBAND_CONST[i]
    return shared


if '_cs_build' in globals():
    _EDBAND_PARENT_CS = _cs_build
    _CS_DICTS = _CS_DICTS + ('_EDBAND_CONST',)

    def _cs_build(n=1024, depth=16):
        _EDBAND_PARENT_CS(n, depth)
        if _EDGE_PARAMETERS is not None:
            _edband_bind(_EDGE_PARAMETERS)
# END ENGINEER EDBAND v1
