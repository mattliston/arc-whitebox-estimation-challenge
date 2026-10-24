# Restoring dropped grouped-read keys saves only 0.004% bill for 217 additional operations

2026-10-16T04:39:10Z · experiment · rejected · measured

The dropped layer 6 keys supplied negligible saving on the current stack, while the 62 expanded keys were never dispatched. No boundary run was justified.

Recorded measurement:

    bill-.004%;+217ops;2networks;62keysneverdispatched

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_keys_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_S6_A_l6g4_v1.py](code/cand_S6_A_l6g4_v1.py)
- [code/cand_S6_B_l6g4_b709_v1.py](code/cand_S6_B_l6g4_b709_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3872](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3872)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [KEYS_RESULT_v1.json](KEYS_RESULT_v1.json) · result
- [fragment_keys_A_l6g4_v1.py.txt](fragment_keys_A_l6g4_v1.py.txt) · result
- [fragment_keys_B_l6g4_b709_v1.py.txt](fragment_keys_B_l6g4_b709_v1.py.txt) · result
- [diag_A_7103248.json](diag_A_7103248.json) · result
- [diag_A_7103250.json](diag_A_7103250.json) · result
- [diag_B_7103248.json](diag_B_7103248.json) · result
- [diag_B_7103250.json](diag_B_7103250.json) · result
- [diag_s6_7103248.json](diag_s6_7103248.json) · result
- [diag_s6_7103250.json](diag_s6_7103250.json) · result
- [guard_diag_A.json](guard_diag_A.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-restoring-dropped-grouped-read-keys-saves-only-0-004) · [Copy provenance](../../../PROVENANCE.json)

