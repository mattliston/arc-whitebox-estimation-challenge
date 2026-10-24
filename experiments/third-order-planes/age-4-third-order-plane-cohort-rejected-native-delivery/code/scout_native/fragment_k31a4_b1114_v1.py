
# BEGIN SCOUT-NATIVE K31L4 AGE 4 B1114 (fragment_k31a4_b1114_v1, 2026-10-13, lead 19:0xZ: the physicist's native form of T147 (h) (passed at 24: K31L4 age 4 at 0.6 at read layers 1-14 on top of C, all-24 -0.157%): the BAND L11-14, since full-layer age 4 delivered poorly natively (D 0.28 / 0.25 on 10-12)) = fragment_k31a4_v2 with _K31A4_LAYERS = (11, 12, 13, 14).
# v2's text: (2026-10-12; = v1 + a read-layer band, for scout-card's run_k6 band arms: the age-4 instance on a zero-shift
# leaf with strength _K31A4 only on the band's read layers and nkl3 lowered by the same amount there, so ages 1-3 stay exact everywhere).
# v1's text: the native route for scout-card's arm a4k31s6 = O/scout_card/make_k3_v3.py k31l_age4(0.6): the pair law already reads one
# strength per cohort age (_np_left: cohort b at ages[l - b - 1] for 1 <= l - b <= len(ages); _np_update keeps K4L's four coefficients of
# the last len(ages) cohorts), so the arm is a fourth entry on _NP_K3L and on every read layer's tuple of _NP_K3L_BY_LAYER; the law's tables
# are rebuilt for the new key (their content does not depend on the strengths). The joins (layer 7: cohorts 0-2, 10: 3-5, 13: 6-8) never take
# a cohort younger than 5, so the age-4 cohort is always in the bank's exact part, as in torch (O/trainer/frozen/k31l_t1508y.py: cohort j at
# l - j <= age). v2: every read layer 1-14 gets its own four-entry tuple, the fourth _K31A4 on the layers in _K31A4_LAYERS and 0 elsewhere (a
# 0 there reads exactly the three-age plane: zero test bit for bit on s51k1tw_x 7100136); _K31A4_LAYERS = 1..14 is v1's arithmetic.
# _K31A4 = 0 = the parent bit for bit (nothing below runs).
_K31A4 = 0.6
_K31A4_LAYERS = (11, 12, 13, 14)
if _K31A4:
    if len(_K3LB_PARENT_AGES()) != 3 or any(len(v) != 3 for v in _NP_K3L_BY_LAYER.values()):
        raise RuntimeError('K31A4: the parent must carry exactly three K3L ages')
    _NP_K3L = _K3LB_PARENT_AGES() + (float(_K31A4),)
    _NP_K3L_BY_LAYER = {l: tuple(float(x) for x in _NP_K3L_BY_LAYER.get(l, _NP_K3L[:3])) + (float(_K31A4) if l in _K31A4_LAYERS else 0.0,)
                        for l in range(1, 15)}
    _NP_TABLES = _np_tables(1024)
# END SCOUT-NATIVE K31L4 AGE 4 v2
