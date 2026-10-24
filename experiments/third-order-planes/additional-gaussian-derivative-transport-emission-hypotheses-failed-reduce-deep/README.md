# Additional Gaussian-derivative transport and emission hypotheses failed to reduce deep plane error

2026-08-30 · experiment · rejected · measured

MORNING SWEEP: three more wedges close, all with receipts. (6) GC-TRANSPORT (Stein shift on the F3 b/d transport columns with carried univariates): NULL - L12 rel 0.0778 vs 0.0777, mode coef -0.2176 vs -0.218. (7) DIAG RESET: v4's carried k3u is HEALTHY - rel 0.038-0.068, corr 0.999, BETTER than the closure host k3 (0.063-0.108); the Gaussian diag reset is not the carrier. (8) COINCIDENT-BOND/rho-truncation: deep error is rho-dense at high |rho| (11.6x at 0.3-0.5) but those pairs are 0.74% of all pairs; 71% of error energy sits at |rho|<0.2; max|rho|=0.515 (no phase-1-style coincident bonds at w1024) - minor contributor only. RUNNING TALLY of refuted carriers for the k3^2/v-row deep-plane mode: 4th-order supply, GC completion columns, mean-error subtraction, f16 precision, fresh kappa3^2 distinct (w1024), GC transport columns, diag reset, bond truncation. The mode's mechanism remains OPEN; its ~1.3x/layer growth with per-layer fresh-deficit-proportional-to-k3^2(l) dynamics says a per-layer emission deficit that the enumerated GC pattern classes do not contain - next sharpest probe: measure the deficit ONE LAYER deep (seed the tensor with TRUTH at L11, run one layer, diff at L12) to isolate the per-layer fresh deficit from accumulation, giving its exact single-step form.

Recorded measurement:

    MORNING SWEEP: three more wedges close, all with receipts. (6) GC-TRANSPORT (Stein shift on the F3 b/d transport columns with carried univariates): NULL - L12 rel 0.0778 vs 0.0777, mode coef -0.2176 vs -0.218. (7) DIAG RESET: v4's carried k3u is HEALTHY - rel 0.038-0.068, corr 0.999, BETTER than the closure host k3 (0.063-0.108); the Gaussian diag reset is not the carrier. (8) COINCIDENT-BOND/rho-truncation: deep error is rho-dense at high |rho| (11.6x at 0.3-0.5) but those pairs are 0.74% of all pairs; 71% of error energy sits at |rho|<0.2; max|rho|=0.515 (no phase-1-style coincident bonds at w1024) - minor contributor only. RUNNING TALLY of refuted carriers for the k3^2/v-row deep-plane mode: 4th-order supply, GC completion columns, mean-error subtraction, f16 precision, fresh kappa3^2 distinct (w1024), GC transport columns, diag reset, bond truncation. The mode's mechanism remains OPEN; its ~1.3x/layer growth with per-layer fresh-deficit-proportional-to-k3^2(l) dynamics says a per-layer emission deficit that the enumerated GC pattern classes do not contain - next sharpest probe: measure the deficit ONE LAYER deep (seed the tensor with TRUTH at L11, run one layer, diff at L12) to isolate the per-layer fresh deficit from accumulation, giving its exact single-step form.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_gct_r35.log#L14](tensor_v4_gct_r35.log#L14)

<details>
<summary>All result files and records (1)</summary>

- [tensor_v4_gct_r35.log](tensor_v4_gct_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-additional-gaussian-derivative-transport-emission-hypotheses-failed-reduce-deep) · [Copy provenance](../../../PROVENANCE.json)

