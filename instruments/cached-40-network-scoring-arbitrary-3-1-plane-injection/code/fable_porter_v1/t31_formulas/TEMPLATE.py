# TEMPLATE for a (3,1)-plane correction formula (fable-porter PT-4). Copy this file, edit _t31_extra, then run
#     bash P2/fable_porter_v1/eval_t31.sh <your_formula.py> <layers, e.g. 2-14> <tag> [rank subset, e.g. 37,39,42]
# (builds the candidate on the pinned bank, smoke + cost on rank 35, paired VALID40 against the cached bank rows, one line in eval_service.log).
#
# CONTRACT
# - This file is appended verbatim to the candidate module: flopscope only (fnp.*, and the candidate's own helpers such as _mm),
#   no imports, no numpy/torch (the builder refuses them). Keep Python-level work tiny (no loops over units): the bank's worst
#   platform residual is 0.385 s of the 0.4 s limit, and every flopscope op is a platform request (~1.1-1.4 ms).
# - _t31_extra(l, ctx) returns X, an (n, n) float64 plane in RAW units: X[a, c] is ADDED to K31[a, c] = kappa4(h_a, h_a, h_a, h_c)
#   (rows a = the cubed unit) before the live pair map, which then uses T31 = t4_a rho_ac + offdiag(X_ac / (v_a s_a s_c)).
#   The diagonal of X is ignored. The bank's own model is K31[a, c] = k4_a Cpre_ac / v_a = radial * vv_a * Cpre_ac.
#   Called only at the enabled layers (subset of 1..14). Verified: X = c * radial * vv_a * Cpre_ac reproduces T31 -> (1 + c) T31 exactly.
# - ctx = the live loop's locals at the pair-map call. Valid for the CURRENT layer: l, n, w32 (W[l] f32), w (f64), mu, Cpre (f64 pre-activation
#   covariance), vv (= diag Cpre), bpre, K21 (the full (2,1) plane; diag(K21) = k3), k4 (= radial*vv*vv), k22, radial (incoming isotropic
#   fourth-cumulant coefficient), pp (pushed operator columns, f32, n x len(lam); old tier reconstructed + young columns), lam (f32 column
#   weights: kappa3 = sum_k lam_k pp_k^3), diagonal (= sum_k lam_k pp_ak^3), live_indices/alive, response_indices/response_mask, omitted,
#   cp_pre (the read plane), star_pre, shared_old (old-tier state: 'columns', 'core', 'pre_basis', ...), newborn (previous layer's births).
#   Names assigned LATER in a layer (g, d, a, t, s, u, delta, selected, births, aux, host, target, rest, ...) still hold the PREVIOUS
#   layer's values at this point.
# - ctx['_t31_carry'] is a per-network dict for state carried across layers (e.g. per-column fourth-order weights aligned with lam).
# - Stacking-order builder (preferred): build_physics.py / eval_physics.sh apply the same contract as in-place string edits valid on the
#   bank AND on engineer2's build_e2.py output (usage: eval_physics.sh <k31row spec | none> <this formula> <layers> <tag> [ranks]).
#   There, the module dict _PHYS_HANDLES['newborn_common'] holds the newborn push's common product W[previous,:]^T base[previous,:]
#   (f32, n x births of the previous layer; base = the previous layer's birth base columns _bd_base32[:, selected]); it is refreshed by
#   every push, so at the pair-map call it is the current layer's. Read it directly (it is a module global); do not modify it.
# - Optional _t31_commit(l, ctx): runs right after layer l's births are committed (lam extended). There the current layer's a (birth
#   coefficient), t, s (two-point diagonal values), u, delta, selected, births, g, alive, p (new columns) and lam (new) are valid.
#   Layers 0..14 all call it (not only the enabled ones); it is how per-column state stays aligned with lam.
# - Cost reference (bank, rank 35, 414.0 GFLOPs, util 0.1883): the existing (2,1) plane read (_shared_old384_read + _slr_inner) is 26% of
#   the bill (~108 GFLOPs, util ~0.049): young columns read densely, old columns through the 384-wide old-tier basis. A dense read over all
#   columns costs ~2*|response|*|live|*len(lam) flops per layer; len(lam) is ~2k at L1 and ~17-23k at L10-14. Plumbing alone: 6n^2 + 4n
#   flops and 6 ops per enabled layer.

def _t31_extra(l, ctx):
    n = ctx['n']
    return fnp.zeros((n, n), dtype=fnp.float64)
