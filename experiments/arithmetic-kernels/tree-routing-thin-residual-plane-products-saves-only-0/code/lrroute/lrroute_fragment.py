

# BEGIN FABLE-ENGINEER2 --lrroute (E2-TR, 2026-09-24; rounding-level, not bit-exact): the low-rank star's three range-finder LEFT
# products (A Omega, A^T Q, A Q'; 1024x1024x48, 45 per net) run on the alt-basis tree kernel at depth 2 instead of plain @.
# In-situ bench (lowrank/thin_route_bench.py on S21): 100.6 -> 81.9 Mflop per product (-18.6%), 45 op calls instead of 1, ~+4.5 ms
# local CPU, max relative difference ~4e-6. Chosen inside the lead's ~2k-request residual budget: 45 products x ~+45 requests =
# ~+2.0k requests per net for about -0.84 Gflop (-0.25% of the bill). Q^T A and the 32-wide star products stay plain (they save only
# ~0.25 Mflop per request). The depth-2 A region grows to 7^2 x 256 x 256 floats so a 1024x1024 left tree fits (set here at import,
# before setup's prebinding), and setup prebinds the three depth-2 keys first.
_LRR_L = 2
_IPT_REGION[2] = (max(_IPT_REGION[2][0], 49 * 256 * 256), _IPT_REGION[2][1], _IPT_REGION[2][2])
_LRR_OLD_LR_FACTOR = _lr_factor


def _lrr_mm(a, b):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if (_AB_ON and len(a.shape) == len(b.shape) == 2 and int(b.shape[0]) == k
            and str(a.dtype) == str(b.dtype) == 'float32' and _ipt_fits(_LRR_L, m, k, n)):
        return _ipt_product_ab(a, b, fnp.empty((m, n), dtype=fnp.float32), _LRR_L)
    return a @ b


def _lr_factor(A32):
    # = the E-62 _lr_factor with its three left products routed through _lrr_mm; everything else unchanged
    n = int(A32.shape[0])
    om = _LR_OMEGA[0]
    if om is None or int(om.shape[0]) != n:
        om = fnp.random.default_rng(_LR_SEED).standard_normal((n, _LR_K), dtype=fnp.float32)
        _LR_OMEGA[0] = om
    Q = fnp.linalg.qr(_lrr_mm(A32, om))[0]
    for _ in range(_LR_Q):
        Q = fnp.linalg.qr(_lrr_mm(A32, fnp.linalg.qr(_lrr_mm(A32.T, Q))[0]))[0]
    B = Q.T @ A32
    if _LR_R is None:
        return Q, B
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :_LR_R], S[:_LR_R, None] * Vt[:_LR_R]


_LRR_PB = ((0, 2, 1024, 1024), (1, 2, 1024, 48), (2, 2, 1024, 48))
_PB_SINGLE = _LRR_PB + tuple(x for x in _PB_SINGLE if x not in _LRR_PB)
# END FABLE-ENGINEER2 --lrroute
