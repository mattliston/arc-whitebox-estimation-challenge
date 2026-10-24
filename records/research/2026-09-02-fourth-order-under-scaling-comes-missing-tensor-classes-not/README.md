# Fourth-order under-scaling comes from missing tensor classes, not the rank cap

2026-09-02 · lesson · informational · measured

The initial attribution of the kurtosis deficit to the fourth-cumulant carrier rank cap was retracted after raising the cap made the result slightly worse. Missing distinct fourth-order tensor classes in the plane-only transport explained the deficit, and frozen per-layer scale tables provided an empirical stopgap.

Recorded measurement:

    Found 2026-09-02 by the phase-2 cross-review (public_g24_phase2_cross_review_v1/REVIEW.md), against the 32M truth on r35 and r100: card v right to 0.3%, k3 3.5→11% off, k4 = 0.99, 0.84, 0.71, 0.61, 0.54, 0.47, 0.42, 0.38 × truth at L1/3/5/7/9/11/13/15 (r100 identical within 0.03; residual after rescale 12-29%).
    
    **Why:** `_cp4_reset` caps the CP4 carrier at rank 2048 by λ²‖u‖⁸ energy; unit-vector births have tiny energy and are dropped first; the cap binds from layer 3; the dropped terms' diagonal is subtracted from the carried k4 — the card discards a growing fraction of its own kurtosis. Since forcing true (v, k3, k4) in-card puts the card at the noise floor (T-43) and each state alone moves ~40% of the residual at the final read, this is the largest priced single defect on record.
    
    **How to apply:** (1) zero-cost test: force k4 := card_k4 / s_l with s_l fit on r35 and frozen to r100 (RETRO_FORCE in public_g20 estimator_retro.py), independent scoring, F2J off/on; (2) derived fix: carry the dropped terms' diagonal forward ("orphan diagonal" push, 1024²/layer) instead of discarding it; or raise the CP4 cap (2048→4096→8192) and price the bill; (3) the phase-1 kappa4-is-a-function-of-kappa3 law is phase-1 history here (error 52-81%). Related: [[p2-third-order-carrier-wall]], [[same-sample-oracle-inflation-law]], [[mechanistic-decomposition-until-solved]]. Seed caution from the same review: the 32M truth contains the 8M draw and indep32M contains indep4M — never pair supply and scoring draws from one seed law.
    
    **CAUSE FOUND (2026-09-02, T-49, teammate A): NOT the CP4 cap.** Raising the cap makes k4 slightly worse. The card's fourth-order PLANES (K22 0.92→0.41, K31 0.52→0.30 of truth by depth) lose ~half of the (3,1)/(2,2) pre-plane content each layer because it comes from the fourth-order DISTINCT classes born at each layer, which the plane-only push (unit-vector CP4 births + quartic star) does not carry; k4 tracks K22. The 3-star words are only −4% of connected κ4 (4-point certificate), so a CP4 star-birth fix is dead; the derived fix is a GEMM-class born-word plane feed (4-path/loop words → next-layer (3,1)/(2,2) planes, ~10-20 GEMM-eq/layer before pruning). The frozen per-layer k4 rescale table (panel mean; universal, sd ≤ 0.018 across 8 nets; public_g19_attribution_v1/k4_ratio_panel_8M.json) is the deployable stopgap (host −30..−37%). Engine defect: fourth_emission_1024 builds (3,1) with uncorrected raw3 (:179) → every `_dep` K31 supply was defective.
    
    **k3 SCALE TABLE (2026-09-02, T-56):** the card's k3 is also mildly under-scaled (panel ratio 0.996 → 0.929 over L1-15, sd ≤ 0.016) and, although the scale is only ~30% of the deep k3 error energy, injecting the frozen per-layer k3 rescale on top of the k4 table takes the host from 5.27e-8 → 4.12e-8 (r35) and 5.00e-8 → 3.84e-8 (r100), F2J off, independent truth — two frozen zero-cost tables halve the host error on both nets. The recursion converts coherent error disproportionately. Files: k4_scale_panel.json (public_g19_attribution_v1), k3_scale_panel_probe.json (public_g26_k3_corrector_features_v1, key ratio_mean). The remaining deep k3 error is ~70% unit-specific (tape features explain 10-12%).

## Programs

- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)

[Research graph](../../../graph/index.html#node=2026-09-02-fourth-order-under-scaling-comes-missing-tensor-classes-not) · [Copy provenance](../../../PROVENANCE.json)

