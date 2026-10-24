# Squared-third-cumulant plane emission terms added through validated Gaussian responses

2026-08-30 · experiment · informational · measured

KAPPA3-SQUARED WIRING, STEP 1 (plane class): added the second-order pattern-pair response to emission_K21 (pairs of D(3,0)/D(0,3)/P(2,1)/P(1,2) insertions, kappa-weighted k3 and carried third-cumulant repeated-index plane (K21), reusing the Mehler series with counts to (6,0); scalar framework = second_order_triple.py w256, validated there by 3D Gauss-Hermite; env SECOND_ORDER=1; assembly is exact product-rule since r1/r2 already carry their GC second order). RESULT r35: deep planes improve with the mode's own depth profile - L12 0.0802->0.0777, L13 0.0855->0.0826, L14 0.0935->0.0901 (~3.5% relative) - but the k3^2/v-mode COEFFICIENT moves only -0.2255 -> -0.2177: the plane-class emission carries ~1/6 of the kappa3-squared content. THE MAJORITY is emitted into the DISTINCT tensor via the completion's missing three-node second-order pattern-pairs (55 pairs, connected filter) and reaches the pre planes through the push.

Recorded measurement:

    KAPPA3-SQUARED WIRING, STEP 1 (plane class): added the second-order pattern-pair response to emission_K21 (pairs of D(3,0)/D(0,3)/P(2,1)/P(1,2) insertions, kappa-weighted k3 and carried K21, reusing the Mehler series with counts to (6,0); scalar framework = second_order_triple.py w256, validated there by 3D Gauss-Hermite; env SECOND_ORDER=1; assembly is exact product-rule since r1/r2 already carry their GC second order). RESULT r35: deep planes improve with the mode's own depth profile - L12 0.0802->0.0777, L13 0.0855->0.0826, L14 0.0935->0.0901 (~3.5% relative) - but the k3^2/v-mode COEFFICIENT moves only -0.2255 -> -0.2177: the plane-class emission carries ~1/6 of the kappa3-squared content. THE MAJORITY is emitted into the DISTINCT tensor via the completion's missing three-node second-order pattern-pairs (55 pairs, connected filter) and reaches the pre planes through the push. That wiring (on-device, ~3-5x completion cost) is the named next implementation - diagnosis now carries three independent confirmations: row-weight k3^2/v 0.92-0.94 cross-net, the depth-profile match, and the plane-class fix moving exactly its predicted minority share.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [tensor_v4_so2_r35.log#L16](tensor_v4_so2_r35.log#L16)

<details>
<summary>All result files and records (1)</summary>

- [tensor_v4_so2_r35.log](tensor_v4_so2_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-squared-third-cumulant-plane-emission-terms-added-through-validated) · [Copy provenance](../../../PROVENANCE.json)

