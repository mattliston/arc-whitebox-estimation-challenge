# Restricting pair-mean truncation to the final read gives a small safe saving

2026-09-25T20:25Z · experiment · adopted · measured

Applying the split only to the final-layer read preserves earlier layers exactly and gives a small adjusted improvement on two native networks. It is kept for a later bundled candidate because the isolated improvement is below the standalone gate.

Recorded measurement:

    - 2026-09-25 13:25 [engineer2] SPLIT u/v, TWO MORE NATIVE POINTS (S28, nets 35/100, same probe):
      v0b = v0 with the frame's Nystrom rows ordered by pair-MEAN energy (same row set; free): bill -1.259/-1.528%, final MSE +0.57/+1.10% (v0 +0.41/+1.43%), so no clear gain over v0. The u truncation at every layer costs roughly +0.5..+1.4% raw on these nets, consistent with covhunter's +1.65 pt for u truncated at every join (TRAIN24). The full lever nets about zero to -0.9% adjusted and would need VALID40 to tell. I'd not invest further.
      v0c = the split at LAYER 15 ONLY (joins untouched; the u/v form built from F at first use): layers 0-14 bit-identical to S28. Bill -0.130/-0.171%, final MSE +0.01/+0.03%, adjusted about -0.12/-0.15%. Only L15's rebuild is caught: the final k3 read does not go through _cp_slr_inner_sq, since _l15_rowcut_k3 reads the rebuilt rows directly. A small, raw-neutral item (uvp_fragment_v0c.py 287ffe5f), below a 0.2% bar on its own; it could ride with another change.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/uvsplit`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/uvp_fragment_v0c.py](code/uvp_fragment_v0c.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1285](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1285)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1286](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1286)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1287](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1287)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-restricting-pair-mean-truncation-final-read-gives-small-safe) · [Copy provenance](../../../PROVENANCE.json)

