# HELD, independent of LG18: restore only mixed first-order Edgeworth r=2.
# No Gaussian-order change, second Edgeworth product, marginal cumulant tail,
# coefficient fit, layer sweep, or model execution is part of this fragment.
_MR2_LAYERS = (12, 13, 14)
_MR2_PARENT_HF = _astra_hf_grouped
_MR2_STATS = dict(ba=0, packed=0)


def _astra_hf_grouped(F, G, F5, G5, P, t21, t12, t22, a, hold, qr, qc, persistent_fg=False):
    result = hold(_MR2_PARENT_HF(F, G, F5, G5, P, t21, t12, t22,
                                a, hold, qr, qc, persistent_fg=persistent_fg))
    if int(_PHYS_LAYER[0]) not in _MR2_LAYERS:
        return result
    # For every AA/BA/BB channel, P2=rho^2/2 and P3=rho^3/6.
    # T31=q_i*rho already holds in this HF path, so its r=2 continuation
    # equals P3*(q_i F5 G3 + q_j F3 G5)/2, without forming a T31 plane.
    f3 = hold(F[3]); f4 = hold(F[4]); f5 = hold(F[5])
    g3 = hold(G[3]); g4 = hold(G[4]); g5 = hold(G[5])
    hf4 = hold(f4 * 0.5); hg4 = hold(g4 * 0.5)
    qf4 = hold(hf4 * 0.5)
    left21 = hold(t21 * hold(hf4 * g3))
    right21 = hold(t12 * hold(f3 * hg4))
    mixed21 = hold(left21 + right21)
    mixed22 = hold(t22 * hold(qf4 * g4))
    tail2 = hold(hold(mixed21 + mixed22) * hold(P['P6'][1]))
    qf5 = hold(hold(qr * f5) * 0.5)
    qg5 = hold(hold(qc * g5) * 0.5)
    tail31 = hold(hold(hold(qf5 * g3) + hold(f3 * qg5)) * hold(P['P6'][2]))
    _MR2_STATS['ba' if persistent_fg else 'packed'] += 1
    return result + hold(tail2 + tail31)
