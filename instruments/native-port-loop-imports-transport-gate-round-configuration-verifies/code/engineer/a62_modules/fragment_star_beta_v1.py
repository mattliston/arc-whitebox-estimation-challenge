
# BEGIN OPUS-ENGINEER a62 STAR STRENGTH (fragment_star_beta_v1, 2026-10-11; the trainer's 'ssb' family, otr_m_v4.py comment STAR STRENGTH):
# with the one-direction star (_SS_RANK = -1, fragment_star_dir_v1) a strength beta on the part beyond the row-sum form,
#     s2 = s2_K0 + beta (s2_K-1 - s2_K0),   i.e.   T2 = T2_K0 + beta (T2_K-1 - T2_K0),   P1 = W' (A 1) common,
# T2_K0 = each unit's mean squared weight (the row-sum form), T2_K-1 = (W o W)' (A' a) / a'a. beta = 1 takes T2_K-1 as SS v3 computes it
# (the parent's arithmetic bit for bit); the layer index is _SS_RANK's: entry i = the star moved into layer i + 1 (_PHYS_LAYER[0] = i),
# so the trainer's beta at the star's layer l (1-15) is _SS_BETA[l - 1]. Appended after fragment_star_dir_v1.
_SS_BETA = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_SSB_PARENT_FACTORS = _ss_factors


def _ss_factors(wt, wt2, A32, hold):
    index = int(_PHYS_LAYER[0])
    beta = float(_SS_BETA[index])
    if _SS_RANK[index] >= 0 or beta == 1.0:
        return _SSB_PARENT_FACTORS(wt, wt2, A32, hold)
    P1, T2 = _SSB_PARENT_FACTORS(wt, wt2, A32, hold)                      # the one-direction form: P1 = W' a, T2 = T2_K-1
    mean = hold(hold(hold(fnp.sum(wt2, axis=1)) / int(A32.shape[0]))[:, None])   # T2_K0, as the row-sum form computes it
    return P1, hold(mean + hold(hold(T2 - mean) * beta))
# END OPUS-ENGINEER a62 STAR STRENGTH
