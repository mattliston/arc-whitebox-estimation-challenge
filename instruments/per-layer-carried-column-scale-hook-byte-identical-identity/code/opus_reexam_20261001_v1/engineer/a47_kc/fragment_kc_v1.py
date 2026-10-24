# BEGIN OPUS-ENGINEER KAPPA_C (a47 kc v1, 2026-10-10; the trainer's per-layer dial kc, layers 1-14; OFF (all 0) = bit-identical). After the
# learned mean correction d of layer l the loop takes the rank-2 update S = host d' + d host' + d d' out of the post covariance, scaled by
# (1 + rc_l) with the card's own knob rc = _GN_RC (-0.08 at layers 1-14). The trainer's card adds kc_l S back after every other state
# hook: C = C0 - (1 + rc_l - kc_l) S. Natively one Python number per layer in the existing lines (rc_eff = rc_l - kc_l): no new operation
# at a layer where rc or kc is non-zero; the diagonal (float64) and the off-diagonal part (float32) take the same factor as before.
_GN_KC = {}
# END OPUS-ENGINEER KAPPA_C
