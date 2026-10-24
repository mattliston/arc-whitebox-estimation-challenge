# Exact layer-2 read-tail correction fixed the layer-3 surrogate gap but lost 0.806% adjusted

2026-10-18T13:28:41Z · experiment · rejected · measured

Restoring the exact layer-2 read tail removed the early native-surrogate discrepancy. The fixed pair still worsened final adjusted error, so matching this intermediate state did not improve the endpoint.

Recorded measurement:

    Fixed 35/200: adjusted +.806%; projected 1.45851e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/rtt2_l2_restore_v3`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyze_native_v1.py](code/analyze_native_v1.py)
- [code/fragment_v3.py](code/fragment_v3.py)
- [code/candidate_95c_rtt_l2_tail_exact_HELD_v3.py](code/candidate_95c_rtt_l2_tail_exact_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5684](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5684)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [BUILD_v3.json](BUILD_v3.json) · result
- [DECISION_v1.json](DECISION_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-exact-layer-2-read-tail-correction-fixed-layer-3) · [Copy provenance](../../../PROVENANCE.json)

