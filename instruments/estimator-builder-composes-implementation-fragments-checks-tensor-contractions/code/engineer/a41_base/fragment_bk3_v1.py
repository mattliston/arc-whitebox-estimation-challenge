# BEGIN OPUS-ENGINEER COLLECTIVE MODE TIMES THE SPECIFIC THIRD CUMULANT (a41 bk3 v1, 2026-10-02; a physics switch, OFF at strength 0:
# the plane and the vector come back as the same objects). Native form of the physicist's O/physicist/bk3.py: the fourth cumulant
# contains kappa Sym4(b (x) kappa3_specific); in the (3,1) plane
#     K31_ij  contains  x1 k3s_i b_j + x2 b_i Ks_ij,    Ks = the bank + star planes, k3s = their diagonal, b = the pushed vector bpre.
# Level 'pair' (the reference's native form): the K21 plane given to the pair map loses (mu_i / (3 v_i)) times that form off the
# diagonal: one row factor on the bank + star planes and a change of the vector in the outer product the K21 plane already has,
#     Ks_ij (1 - k x2 mu_i b_i / (3 v_i))  +  (bv_i - k x1 mu_i k3s_i / (3 v_i)) b_j  + ..
# NOT USABLE AS IT STANDS: _bk3_fold changes the loop's K21 plane itself, which the learned correction's features, the edge, the
# covariance knob and the birth update read as well, while the reference's hook changes the plane for the pair map's call only.
# The builder refuses BK3p until the pair map gets its own plane (a second n x n plane and two cache lines); _bk3_fold and its
# unit check are kept for that.
# Level 'r31' (the reference's research form, the one its table measured): the same form as the pair map's extra (3,1) plane,
# through the loop's own hook for it (_T31_LAYERS / _t31_extra): three float32 passes and the pair map's normalisation.
_BK3_K31 = 0.0           # k: overall strength (0 = off)
_BK3_X = (3.2, 10.0)     # x1, x2
_BK3_LEVEL = 'pair'
_BK3_LO, _BK3_HI = 1, 14
_BK3_CONST = {}
_BK3_STATS = dict(layers=0)


def _bk3_rows(mu, vv, bpre, cpdiag, star_pre, hold):
    """(k x2 mu b / (3 v), k x1 mu k3s / (3 v)) per unit, float64."""
    S = _BK3_CONST
    n = int(mu.shape[0])
    if S.get('n') != n:   # rows (mu, 1 / v, b, k3s): the products mu b / v and mu k3s / v
        S.update(n=n, index=fnp.asarray([[0, 0], [1, 1], [2, 3]]),
                 coefficient=fnp.asarray([[_BK3_K31 * _BK3_X[1] / 3.0], [_BK3_K31 * _BK3_X[0] / 3.0]], dtype=fnp.float64))
    k3s = hold(cpdiag + hold(fnp.diag(star_pre)))
    rows = hold(fnp.stack([mu, hold(fnp.power(hold(fnp.maximum(vv, 1e-12)), -1.0)), bpre, k3s]))
    return hold(fnp.unstack(hold(hold(fnp.prod(hold(rows[S['index']]), axis=0)) * S['coefficient'])))


def _bk3_fold(l, Ks32, bv, mu, vv, bpre, cpdiag, star_pre):
    """Level 'pair'. Ks32: bank + star planes (float32); bv: the vector of the K21 plane's outer product with bpre (float64).
    Returns the plane with its row factor and the changed vector; the same objects when the switch is off or the layer is out of range."""
    if not _BK3_K31 or _BK3_LEVEL != 'pair' or not (_BK3_LO <= l <= _BK3_HI):
        return (Ks32, bv)
    hold = _E2Hold()   # temporaries of this call, released together
    row, shift = _bk3_rows(mu, vv, bpre, cpdiag, star_pre, hold)
    _BK3_STATS['layers'] += 1
    return (fnp.einsum('ij,i->ij', Ks32, hold(hold(1.0 - row).astype(fnp.float32)), optimize=False), bv - shift)


def _bk3_extra(l, ctx):
    """Level 'r31': the raw extra (3,1) plane k (x1 k3s_i b_j + x2 b_i Ks_ij), zero diagonal, float32."""
    hold = _E2Hold()   # temporaries of this call, released together
    bpre, star32 = ctx['bpre'], ctx['_plane_star32']
    Ks32 = hold(ctx['_ps_cp32'] + (star32 if star32 is not None else hold(ctx['star_pre'].astype(fnp.float32))))
    k3s = hold(ctx['_plane_cpdiag64'] + hold(fnp.diag(ctx['star_pre'])))
    b32 = hold(bpre.astype(fnp.float32))
    plane = fnp.einsum('ij,i->ij', Ks32, hold(b32 * (_BK3_K31 * _BK3_X[1])), optimize=False)
    hold(fnp.add(plane, hold(fnp.outer(hold(hold(k3s * (_BK3_K31 * _BK3_X[0])).astype(fnp.float32)), b32)), out=plane))
    fnp.fill_diagonal(plane, 0)
    _BK3_STATS['layers'] += 1
    return plane


if _BK3_K31 and _BK3_LEVEL == 'r31':   # the loop's hook for the pair map's extra plane (empty in the parent)
    if _T31_LAYERS:
        raise RuntimeError('bk3 at level r31: the extra plane already has a formula')
    _T31_LAYERS = tuple(range(_BK3_LO, _BK3_HI + 1))
    _t31_extra = _bk3_extra
# END OPUS-ENGINEER COLLECTIVE MODE TIMES THE SPECIFIC THIRD CUMULANT
