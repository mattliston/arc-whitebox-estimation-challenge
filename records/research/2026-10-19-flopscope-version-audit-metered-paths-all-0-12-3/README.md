# Flopscope version audit: metered paths all 0.12.3; charge censuses are pre-weight, split unbilled, layout 2.07% of bill

2026-10-19T05:02:35Z · lesson · informational · derived

Every metered local path (judge client and server, setup gate) runs flopscope 0.12.3; the release common.py environments run 0.12.0 and the orchestrator venv 0.10.0, which matters only because the in-process charge census had run there. The census wraps the charge hook before weights are applied, so its costs must be multiplied by the 0.12.3 default weights (split/transpose/unstack 0.0, take/argsort 4.0, float64 rate 2.0) before pricing a cut; re-weighted, the layout share of the bill is 2.07% and the split charges (1.1%) are not billed at all.

Recorded measurement:

    base a7ffd713 under 0.12.3: 40,051 charges, 250.96 GF pre-weight (identical op structure); split charges 2.76 GF = 1.1% not billed; true layout bill 2.07% (reshape .73, copyto .44, put_along_axis .20, getitem .18, stack .17, concatenate .12, astype .10); whestbench 0.16.1

## Programs

- [probe_charges_v2.py](probe_charges_v2.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5907](../../TEAM_LOG_2026-10-01_to_10-22.md#L5907)

<details>
<summary>All result files and records (3)</summary>

- [charges_v123_a7ffd713_7103256.json](charges_v123_a7ffd713_7103256.json) · result
- [charges_v123_a7ffd713_7103256.log](charges_v123_a7ffd713_7103256.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-flopscope-version-audit-metered-paths-all-0-12-3) · [Copy provenance](../../../PROVENANCE.json)

