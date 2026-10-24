# Compact setup repair removed duplicate handles and met a strengthened 4.20 s startup bound

2026-10-18T10:43:23Z · experiment · adopted · measured

Static offset and stride proofs removed duplicate setup handles, followed by pruning, batched views, lean holders and group-node sharing. The frozen compact repair was rescored and passed a strengthened startup envelope before submission; all five hosted workers then completed setup.

Recorded measurement:

    3552 duplicate handles and 32 requests removed; projected envelope 4.155312 s <=4.20; 32 networks: adjusted -0.522646%, raw -0.028920%, bill -0.493847%; hosted setup 4.093628..4.406806 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/candidate_compact10_capacity35_memodiet_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5561](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5561)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5577](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5577)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5584](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5584)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5606](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5606)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5611](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5611)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [V65U_RELEASE_SEAL_v1.json](V65U_RELEASE_SEAL_v1.json) · record
- [HOSTED_SETUP_336935_v1.json](HOSTED_SETUP_336935_v1.json) · result
- [V65U_RELEASE_CONFIRMATION_v2.json](V65U_RELEASE_CONFIRMATION_v2.json) · result
- [FINAL_GATE_v2.json](FINAL_GATE_v2.json) · result
- [HOSTED_SETUP_GATE_v1.json](HOSTED_SETUP_GATE_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336935: compact setup repair: 100 networks passed at 1.4468565569762618e-09](../../../submissions/336935/README.md)

[Research graph](../../../graph/index.html#node=2026-10-18-compact-setup-repair-removed-duplicate-handles-met-strengthened-4) · [Copy provenance](../../../PROVENANCE.json)

