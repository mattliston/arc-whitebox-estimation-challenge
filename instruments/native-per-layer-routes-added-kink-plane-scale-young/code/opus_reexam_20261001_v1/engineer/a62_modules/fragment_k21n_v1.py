# BEGIN OPUS-ENGINEER a62 K21N BY LAYER v1 (file written 2026-10-12 07:11Z; for S51's band dial dk21n_7_10: k21n had one value for all layers natively).
# Torch (trainer/otr_pl.py cov_pm, layers 2-14): k21n = V3['k21n'] + eps['k21n'][l], the factor on K21's non-kink rows in the GN covariance
# correction. Native: the correction (_gn_dcov, both the G5 and the older form) reads its knobs once per call through _pld_knobs(l)
# (fragment_pld_v3: per-layer iso); this wraps it so that k21n takes _K21N_BY_LAYER[l] at the layers listed (the knobs dict is copied once
# per layer and kept), every other knob as before. _K21N_BY_LAYER = {} = the parent bit for bit. Needs the per-layer dials fragment.
assert '_pld_knobs' in globals(), 'fragment_k21n_v1 needs fragment_pld (DLY@) in the parent'
_K21N_BY_LAYER = {}
_K21N_CACHE = {}
_K21N_PARENT_KNOBS = _pld_knobs


def _pld_knobs(l):
    knobs = _K21N_PARENT_KNOBS(l)
    v = _K21N_BY_LAYER.get(l)
    if v is None:
        return knobs
    got = _K21N_CACHE.get(l)
    if got is None or got[0] is not knobs:
        got = _K21N_CACHE[l] = (knobs, dict(knobs, k21n=float(v)))
    return got[1]
# END OPUS-ENGINEER a62 K21N BY LAYER v1
