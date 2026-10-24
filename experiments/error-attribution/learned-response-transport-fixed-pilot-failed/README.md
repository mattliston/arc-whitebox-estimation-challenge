# Learned response transport: fixed pilot failed

2026-09-06T08:34Z · experiment · rejected · measured

The response itself is measurably better at predicting gates: using two independent2^20-sample streams each, late8..14 uniform cross-mean squared error (MSE) learned/ Gaussian=.019153/.017730 and learned/host=.872946/.858712 on35/100. The >10% host advantage has a narrow sampling margin (.68/.89 jackknife standard errors at that threshold). The predeclared matched20/4 two-stage pilot FAILED.

Recorded measurement:

    The response itself is measurably better at predicting gates: using two independent2^20-sample streams each, late8..14 uniform cross-MSE learned/ Gaussian=.019153/.017730 and learned/host=.872946/.858712 on35/100. The >10% host advantage has a narrow sampling margin (.68/.89 jackknife standard errors at that threshold). No more gate sampling was performed. Synthetic full-tensor repeated restoration passed1.26e-16. Both current disabled paths were byte-identical to original P/Q trajectories and all enabled P0 responses were captured before current mean corrections. Fixed response changed only inherited CP legs8..14, original newborn geometry, trace and declared repeated targets retained. Frozen final-mean shifts were4.56e-6/4.07e-6 RMS on35/100, not accuracy measurements. The predeclared matched20/4 two-stage pilot FAILED. Main P1 received one 40-epoch fit while auxiliary response P0 remained frozen with its own honest recurrent state. One40-epoch final Q fit on actual on-policy rows. Exactly24 teacher and24 final rows; no current-target feature leakage. Actual final h/host/sigma/P1 state byte-identical to training row inputs. Four fixed holdouts37/39/42/48, existing1e9 truths: | Quantity | Pooled raw final MSE | |---|---:| | Original bank | 2.120165611358589e-8 | | Immutable matched control | 2.235463499383767e-8 | | Response transport | 2.247214819383274e-8 | New/control=1.0052567711, new/bank=1.0599241905,0/4 wins. Thresholds were<=.90 and<=.92 with>=3 wins. The control was reused with exact payload hash checks and its archived MSE reproduced exactly, not retrained.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/gnn_mean_response_dual_v1.py](code/astra_phase2_reexamination_v1/gnn_mean_response_dual_v1.py)
- [code/astra_phase2_reexamination_v1/judge_mean_response_transport_pilot_row_v1.py](code/astra_phase2_reexamination_v1/judge_mean_response_transport_pilot_row_v1.py)
- [code/astra_phase2_reexamination_v1/judge_mean_response_transport_pilot_v1.py](code/astra_phase2_reexamination_v1/judge_mean_response_transport_pilot_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/mean_response_feature_diagonal_v1.py](code/astra_phase2_reexamination_v1/mean_response_feature_diagonal_v1.py)
- [code/astra_phase2_reexamination_v1/mean_response_transport_reference_v1.py](code/astra_phase2_reexamination_v1/mean_response_transport_reference_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md#L1](MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md#L1)
- [MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md#L14](MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md#L14)

<details>
<summary>All result files and records (3)</summary>

- [MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md](MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md) · record
- [mean_response_transport_pilot_verdict_v1.json](mean_response_transport_pilot_verdict_v1.json) · result
- [MEAN_RESPONSE_TRANSPORT_PILOT_DESIGN_V1.md](MEAN_RESPONSE_TRANSPORT_PILOT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-learned-response-transport-fixed-pilot-failed) · [Copy provenance](../../../PROVENANCE.json)

