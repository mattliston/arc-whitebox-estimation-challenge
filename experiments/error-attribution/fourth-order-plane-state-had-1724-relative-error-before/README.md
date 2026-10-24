# Fourth-order plane state had 17–24% relative error before the emission step

2026-09-04 · experiment · rejected · measured

The emission's Edgeworth channels take fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31); the card carries them in a rank-2048 fourth-cumulant canonical-polyadic carrier (CP4) with 17-24% error (engine diag). Oracle arm running (truth K22/K31 fed each layer). If binding, descend: CP4 birth formula vs rank truncation vs push, graded against mps_planes truth.

Code availability: The shared fourth-order engine and consumption instrument were recovered. The cited entry says the oracle arm was still running; its 17–24% state-error observation must not be read as the completed oracle result.

Recorded measurement:

    The emission's Edgeworth channels take K22/K31; the card carries them in a rank-2048 CP4 with 17-24% error (engine diag). Oracle arm running (truth K22/K31 fed each layer). If binding, descend: CP4 birth formula vs rank truncation vs push, graded against mps_planes truth. MEASURED (T-20): truth k4u+K22/K31 -> v-arm 7.4 (not lifted) but the k4 arm becomes +5.7 and joint 18.7 (vs 14.4). Not the variance limiter; it unlocks N3.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py](code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py](code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L1](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-order-plane-state-had-1724-relative-error-before) · [Copy provenance](../../../PROVENANCE.json)

