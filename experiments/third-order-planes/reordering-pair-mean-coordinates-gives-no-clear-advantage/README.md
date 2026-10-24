# Reordering pair-mean coordinates gives no clear advantage

2026-09-25T20:25Z · experiment · rejected · measured

Sorting the same sampled rows by pair-mean energy does not resolve the accuracy loss from the one-frame split. The measured savings and errors remain unsuitable as a broad tier change.

Recorded measurement:

    - 2026-09-25 13:25 [engineer2] SPLIT u/v, TWO MORE NATIVE POINTS (S28, nets 35/100, same probe):
      v0b = v0 with the frame's Nystrom rows ordered by pair-MEAN energy (same row set; free): bill -1.259/-1.528%, final MSE +0.57/+1.10% (v0 +0.41/+1.43%), so no clear gain over v0. The u truncation at every layer costs roughly +0.5..+1.4% raw on these nets, consistent with covhunter's +1.65 pt for u truncated at every join (TRAIN24). The full lever nets about zero to -0.9% adjusted and would need VALID40 to tell. I'd not invest further.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/uvsplit`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/s28uv/cand_uvp0b_T.py](code/s28uv/cand_uvp0b_T.py)
- [code/uvp_bill_probe.py](code/uvp_bill_probe.py)
- [code/uvp_fragment_v0b.py](code/uvp_fragment_v0b.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1285](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1285)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1286](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1286)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-reordering-pair-mean-coordinates-gives-no-clear-advantage) · [Copy provenance](../../../PROVENANCE.json)

