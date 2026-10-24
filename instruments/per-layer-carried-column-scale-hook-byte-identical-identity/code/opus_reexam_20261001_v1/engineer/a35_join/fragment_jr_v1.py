# BEGIN OPUS-ENGINEER JOIN REPAIR (a35 JR v1, 2026-10-02; a physics change, NOT rounding-level). A join replaces the columns J of the
# joined cohorts by their projection Qn Fj on the join's rank-r frame. For third-order content T the next layer's plane holds the
# coherent part m_i (W' t)_j off the diagonal and three times that on it, with t_c = sum_a T_aac (how strongly unit c is tied to the
# squared norm of its layer; m = each unit's mean squared weight; a33_starsum). The projection changes t of the joined columns by
#     miss = J (lam o |J_r|^2) - Qn (Fj (lam o |Fj_r|^2)),
# 1.0-2.3% of it on network 651 (a35_join/probe_join.py): 1.6-2.6 times what the fired rank-32 star missed of its row sums at the
# same layers, and the old tier never gets it back. Here the miss is recorded at the join and its coherent part is added to the
# transported star of the join layer (the plane the pair map reads, and its diagonal); the star channel carries it on from there.
# Per join: two passes over the joined columns and one vector through W, about 35 Mflop (0.04% of the bill for the three joins).
# Not included: what the OLD columns lose when they are re-projected on the new frame at the second and third join.
_JR_SCALE = 1.0      # 1 = the miss as computed
_JR_STATE = {}
_JR_STATS = dict(joins=0, applied=0)


def _jr_record(layer, previous, J, coefficient, Qn, Fj):
    hold = _E2Hold()
    before = hold(J @ hold(coefficient * hold(fnp.sum(hold(J * J), axis=0))))
    after = hold(Qn @ hold(Fj @ hold(coefficient * hold(fnp.sum(hold(Fj * Fj), axis=0)))))
    _JR_STATE[int(layer)] = (previous, before - after)
    _JR_STATS['joins'] += 1


def _jr_apply(l, w32, star64, star32):
    hold = _E2Hold()
    previous, miss = _JR_STATE.pop(int(l))
    c = hold(hold(hold(w32[previous, :]).T) @ miss) * _JR_SCALE
    wt = w32.T
    m = hold(hold(fnp.sum(hold(wt * wt), axis=1)) / int(w32.shape[0]))
    mc = hold(m * c)
    hold(fnp.add(star32, hold(fnp.outer(m, c)), out=star32))
    fnp.fill_diagonal(star32, hold(hold(fnp.diag(star32)) + hold(2 * mc)))
    fnp.fill_diagonal(star64, hold(hold(fnp.diag(star64)) + hold(hold(3 * mc).astype(fnp.float64))))
    _JR_STATS['applied'] += 1
# END OPUS-ENGINEER JOIN REPAIR
