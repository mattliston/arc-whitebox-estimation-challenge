
# BEGIN SCOUT-NATIVE K31L4 AGE 4 (fragment_k31a4_v1, 2026-10-12, lead 21:3xZ: native route for scout-card's arm a4k31s6 = O/scout_card/make_k3_v3.py
# k31l_age4(0.6): a fourth k31ly instance with age limit 4 and the age-3 instance's base lowered by the same s, so the young pairs' fourth-chaos
# plane (K31L4) gets strength s on the cohorts aged 4 while ages 1-3 keep theirs). Native: the pair law already reads one strength per cohort
# age (_np_left: cohort b at ages[l - b - 1] for 1 <= l - b <= len(ages); _np_update keeps K4L's four coefficients of the last len(ages)
# cohorts), so the arm is a fourth entry _K31A4 on _NP_K3L and on every per-layer tuple of _NP_K3L_BY_LAYER; the law's tables are rebuilt
# for the new key (their content does not depend on the strengths). The joins (layer 7: cohorts 0-2, 10: 3-5, 13: 6-8) never take a cohort
# younger than 5, so the age-4 cohort is always in the bank's exact part, as in torch (O/trainer/frozen/k31l_t1508y.py: cohort j at l - j <= age).
# _K31A4 = 0 = the parent bit for bit (nothing below runs).
_K31A4 = 0.6
if _K31A4:
    if len(_K3LB_PARENT_AGES()) != 3 or any(len(v) != 3 for v in _NP_K3L_BY_LAYER.values()):
        raise RuntimeError('K31A4: the parent must carry exactly three K3L ages')
    _NP_K3L = _K3LB_PARENT_AGES() + (float(_K31A4),)
    _NP_K3L_BY_LAYER = {l: tuple(float(x) for x in v) + (float(_K31A4),) for l, v in _NP_K3L_BY_LAYER.items()}
    _NP_TABLES = _np_tables(1024)
# END SCOUT-NATIVE K31L4 AGE 4
