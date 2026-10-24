# Exact signed-pair read rotation: improved a dormant microkernel but lost 0.854695% adjusted natively

2026-10-18T05:01:19Z · experiment · rejected · measured

Rotating both channels of each signed pair reduced contraction rounding in a saved microkernel. That microkernel used a dormant peeling schedule; the deployed read rotation worsened both native raw scores and bill.

Recorded measurement:

    Saved rounding -51.255%; native first 2: raw +.509084%, bill +.343859%, adjusted +.854695% → 1.4639112178e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_bill_v1/build_pair_rotation_v1.py](code/astra_oct18_bill_v1/build_pair_rotation_v1.py)
- [code/astra_oct18_bill_v1/candidate_v61j_pair_rotation_HELD_v1.py](code/astra_oct18_bill_v1/candidate_v61j_pair_rotation_HELD_v1.py)
- [code/astra_oct18_bill_v1/fragment_pair_rotation_v1.py](code/astra_oct18_bill_v1/fragment_pair_rotation_v1.py)
- [code/astra_oct18_bill_v1/saved_pair_rotation_v3.py](code/astra_oct18_bill_v1/saved_pair_rotation_v3.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5213](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5213)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5217](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5217)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-exact-signed-pair-read-rotation-improved-dormant-microkernel-but) · [Copy provenance](../../../PROVENANCE.json)

