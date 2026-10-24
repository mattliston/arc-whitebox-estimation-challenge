

# BEGIN FABLE-ENGINEER2 DEPTH-7 1024^3 (d7sq v1, 2026-09-25; rounding-level, NOT bit-exact; OPTION on the lead's frontier, NOT gated yet;
# compose AFTER d7cp_fragment.py, which defines _IPT_REGION[7] and the depth-7 A key). The 1024^3 products (the covariance's C w in every
# layer, _flms_cov_star / _flms_covariance_diagonal, and the newborn common pushes of layers 1-3; 18 per net on nets 35/100) run at depth 7
# as ONE product (A/B/C trees 52.7M floats each, inside d7cp's depth-7 region). Priced on S27 (bench_frontier.py): bill -51.7 MF per call
# (-4.88%) = -0.305% of the net's bill; backend +0.100 s per call = +1.8 s/net = about +3.7 s on the slowest MLP (WALL v3); +11 requests
# per call; rounding error x2.06-2.2 on these products.
_D7SQ_ON = True
_PB_SINGLE = ((1, 7, 1024, 1024), (2, 7, 1024, 1024)) + tuple(_PB_SINGLE)
_D7SQ_STATS = {'products': 0}
_D7SQ_OLD_MM = _mm


def _mm(a, b):
    if (_D7SQ_ON and _AB_ON and _WINOGRAD and _SCRATCH is not None and len(a.shape) == 2 and len(b.shape) == 2
            and str(a.dtype) == 'float32' and str(b.dtype) == 'float32' and tuple(a.shape) == (1024, 1024) and tuple(b.shape) == (1024, 1024)
            and 7 in _IPT_REGION):
        out = fnp.empty((1024, 1024), dtype=fnp.float32)
        _D7SQ_STATS['products'] += 1
        return _ipt_product_ab(a, b, out, 7)
    return _D7SQ_OLD_MM(a, b)
# END FABLE-ENGINEER2 DEPTH-7 1024^3
