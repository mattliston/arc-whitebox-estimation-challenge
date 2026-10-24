# Per-unit cumulant source needs random error below about 25% of the current error to retain value

2026-09-25T06:00Z · finding · informational · measured

Truth-directed interpolation overstates tolerance to an independently noisy source. Across held-out networks, random errors at half the current residual erased the gain; the source must be accurate at every supplied layer.

Recorded measurement:

    - 2026-09-24 23:05 analyst: k3+k4 SOURCE SPEC (fable_analyst_v1/RESULT.md section 35; k34_spec.py; 7 held-out nets, 8M truth = effectively exact, L12-14 mean correction 0.5; all gated). (i) ACCURACY: along the card's own error direction concave (25/50/75/100% of the way to truth = -7.7/-14.7/-18.3/-18.4% raw vs deployed; optimum ~0.88), but a source's RANDOM error hurts ~3.5x more than the card's own: truth + independent noise at 25%/50%/100% of the card's error rms = -11.7% / +6.2% / +79%; break-even ~42%. (ii) UNITS: NO deployable concentration. Truth-free Jacobian-norm ranking 64/128/256 units per layer = -0.2/-3.7/-9.2% (15/31/57% of the value; random 128/256 = 12/26%). |adjoint| ranking -15.7/-21.4/-25.2% but it picks units with the final error (overlap with the truth-free top-128 only 33-38%); first-order-gain ranking -50% is LEAKAGE (a random-sign fix chosen the same way = -42.7%). (iii) LAYERS: L13 -6.3%, L14 -5.8%, L12 -2.7% alone (additive; L13-14 = 85%); L10-11 <= 1%. (iv) ORIGIN: earlier fixes never reach later layers (per-unit k3/k4 feed only their layer's pair map, not the carriers; later k3/k4 error moves <= 0.7%); L6-8 / L9-11 buy 2-5%. SPEC: k3 within ~1.5% and k4 within ~7% relative rms (25% of the card's 6%/25%) on ALL units at L12-14, correction 0.5 -> ~-12% (-18% exact); break-even ~2.6%/11%; 4x the card's accuracy; sampling needs ~2e6 samples (~175x bill); 128 targeted units give only ~-4% even if exact. Also: section 34's injection was pair-map-only; the all-consumer form is identical (-18.1 vs -18.4%).
    - 2026-09-24 23:00 lead: ANALYST §35 (k3+k4 SOURCE SPEC; S6b + B2 ep1, 7 held-out, 8M, correction .5).
      (i) Accuracy: moving the card toward truth by 25/50/75/100% gives -7.7/-14.7/-18.3/-18.4% (concave, optimum ~.88). But the source's own RANDOM error hurts ~3.5x more than the card's systematic error: noise at 25% of the card's error rms keeps 69% of the value; at 50% it loses (+6.2%). Break-even ~42%.
      (ii) Units: no deployable concentration; truth-free Jacobian top-128 = 31% of the value.
      (iii) Layers: L13-14 carry 85%.
      (iv) Origin: earlier fixes don't propagate, since per-unit k3/k4 feed only their own layer's pair map; the source must act AT L12-14.
      SPEC: k3 within ~1.5%, k4 within ~7% relative rms (25% of the card's error), all units at L12-14, correction .5; worth ~-12% (exact -18%). Sampling needs ~2M forward samples (~175x the bill). Nothing on our books reaches it.
    - 2026-09-24 23:20 analyst: CROSS-FRAME NOTE for the k3+k4 spec. Covhunter's current-card rows (k34_supply_price.py) match section 35 (net 150 L12-14 eps .5/1: -8.3/-6.9% vs my -8.1/-5.8% at s=1; -13.0/-15.7% vs -13.8/-15.7% at s=.5). Caveats: (1) eps .5 >= eps 1 only at s=1; at s=.5 eps 1 wins (-18.4 vs -14.7%, 7 nets). Eps is error ALONG the card's own direction; random source error at 25/50% of the card's rms = -11.7/+6.2%, so the spec stays <=25% random (break-even ~42%). (2) The L2-14 surplus over L12-14 is the sum of per-layer local pair-map effects (1-5% per band), not a repair of the carried error (per-unit fixes never reach the carriers); a source covering L2-14 must meet the spec at every layer. Asked covhunter for random-noise rows at L12-14 and L2-14.
    2026-09-24 23:02 covhunter: k3+k4 LEVER ON THE CURRENT CARD (RESULT sec 29; S22 torch counterpart, loopR1 + round-1 dials + hooks): truth supply k = card + eps(truth - card), 7 held-out 8M nets geo (r35/r100 32M): L12-14 eps1 s1 -13.3%, s.5 -19.6%; eps.5 s1 -12.3%, s.5 -15.0%; L2-14 eps.5 s1 -19.2%, eps1 s.5 -23.7%; all 7/7. Control (s.5, no supply) +3.0%. Matches the analyst's sec 34 frame (-11.0/-18.4%) -> the lever is intact on the deployed physics. Noise rows (random source error at .25/.5 x the card's error rms, analyst's seeds) running.
    - 2026-09-24 23:02 lead: COVHUNTER §29: the k3+k4 lever HOLDS on the CURRENT card (loopR1 + round-1 dials + hooks; 7 held-out 8M nets, 7/7). Truth per-unit pre-activation k3+k4 in the pair map, correction s at L12-14:
      - L12-14 eps 1: -13.3% (s 1) / -19.6% (s .5).
      - L2-14 eps 1: -16.9% / -23.7%.
      - L2-14 eps .5: -19.2% / -22.6%.
      - r35/r100 (32M) are smaller but agree.
      Per the analyst's caveats, the realistic spec comes from NOISE rows (random source error is ~4x costlier than error along the card's own direction), and the accuracy spec applies at every covered layer. Noise rows (r .25/.5 of the card's error rms; L12-14 and L2-14; s .5) running.
    2026-09-24 23:21 covhunter: k3+k4 NOISE ROWS (RESULT sec 29; current card, s .5, truth + r x card-error-rms random): L12-14 r.25 -12.6% (7/7), r.5 +7.3% (1/7); L2-14 r.25 -14.4% (7/7), r.5 +13.2% (0/7). Quadratic: break-even r* ~.42 (L12-14) / ~.40 (L2-14). SOURCE SPEC: random per-unit error <= ~25% of the card's current error rms at EVERY fed layer (k3 ~1-2%, k4 ~3-7% relative rms, ~4x better than the card) to capture ~60% of the lever; r100 fragile (+33/+59% at r .5) -> price on the 7 held-out nets.
    - 2026-09-24 23:23 lead: k3+k4 SOURCE SPEC FINAL (covhunter §29 noise rows, current card, 7 held-out, s .5): random per-unit error at r = .25 of the card's error rms gives -12.6% (L12-14) / -14.4% (L2-14), 7/7; r = .5 gives +7.3% / +13.2%. Break-even r ~.40-.42 (the analyst's frame agrees). SPEC: random error <= ~25% of the card's at EVERY fed layer, i.e. k3 within ~1-2% and k4 within ~3-7% relative rms per unit (~4x better than the card). That buys ~60% of the lever (-12..-14% raw). Protocol: any candidate source first passes an OFFLINE per-unit accuracy screen vs the 8M/32M truths (error <= 25% of the card's); only survivors get closed-loop pricing (MPS eval mode, 7 held-out nets). Net 100 is fragile, so no single-net screens.

## Programs

- [k34_spec.py](k34_spec.py)
- [k34_supply_price.py](k34_supply_price.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)
- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)
- [../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py](../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py)
- [../../../instruments/native-matching-low-rank-residual-plane-training-avoids-differentiating/code/fable_engineer_v1/lrstar_torch_hook.py](../../../instruments/native-matching-low-rank-residual-plane-training-avoids-differentiating/code/fable_engineer_v1/lrstar_torch_hook.py)
- [../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py](../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py](../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L729](../../TEAM_LOG_2026-09-22_to_09-26.md#L729)
- [TEAM_LOG_2026-09-22_to_09-26.md#L732](../../TEAM_LOG_2026-09-22_to_09-26.md#L732)
- [TEAM_LOG_2026-09-22_to_09-26.md#L733](../../TEAM_LOG_2026-09-22_to_09-26.md#L733)
- [TEAM_LOG_2026-09-22_to_09-26.md#L734](../../TEAM_LOG_2026-09-22_to_09-26.md#L734)
- [TEAM_LOG_2026-09-22_to_09-26.md#L735](../../TEAM_LOG_2026-09-22_to_09-26.md#L735)
- [TEAM_LOG_2026-09-22_to_09-26.md#L736](../../TEAM_LOG_2026-09-22_to_09-26.md#L736)
- [TEAM_LOG_2026-09-22_to_09-26.md#L737](../../TEAM_LOG_2026-09-22_to_09-26.md#L737)
- [TEAM_LOG_2026-09-22_to_09-26.md#L740](../../TEAM_LOG_2026-09-22_to_09-26.md#L740)
- [TEAM_LOG_2026-09-22_to_09-26.md#L741](../../TEAM_LOG_2026-09-22_to_09-26.md#L741)
- [TEAM_LOG_2026-09-22_to_09-26.md#L742](../../TEAM_LOG_2026-09-22_to_09-26.md#L742)
- [TEAM_LOG_2026-09-22_to_09-26.md#L743](../../TEAM_LOG_2026-09-22_to_09-26.md#L743)
- [TEAM_LOG_2026-09-22_to_09-26.md#L744](../../TEAM_LOG_2026-09-22_to_09-26.md#L744)
- [TEAM_LOG_2026-09-22_to_09-26.md#L745](../../TEAM_LOG_2026-09-22_to_09-26.md#L745)
- [TEAM_LOG_2026-09-22_to_09-26.md#L746](../../TEAM_LOG_2026-09-22_to_09-26.md#L746)
- [TEAM_LOG_2026-09-22_to_09-26.md#L747](../../TEAM_LOG_2026-09-22_to_09-26.md#L747)
- [TEAM_LOG_2026-09-22_to_09-26.md#L754](../../TEAM_LOG_2026-09-22_to_09-26.md#L754)
- [TEAM_LOG_2026-09-22_to_09-26.md#L756](../../TEAM_LOG_2026-09-22_to_09-26.md#L756)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../../../experiments/third-order-planes/gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-per-unit-cumulant-source-needs-random-error-below-about) · [Copy provenance](../../../PROVENANCE.json)

