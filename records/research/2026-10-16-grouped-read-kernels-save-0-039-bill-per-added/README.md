# Grouped read kernels save 0.039% bill per added wall-second; compact operand layout cuts kernel CPU 42%

2026-10-16T01:10:51Z · finding · informational · measured

Paired platform pricing identified the grouped read dispatch as the least wall-efficient arithmetic saving. Compact operand layout supplied a free runtime improvement, while residual time remained the tightest margin.

Recorded measurement:

    336334->336342 bill-.424%,meanwall+10.9s,max+14.7s;compactB CPU-42%;offmean81.1/worst94.0,bill+.710%;compact16mean83.6/worst96.9,bill+.508%;residual.364/.396

## Programs

- [wallprice_judge_v1.py](wallprice_judge_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3619](../../TEAM_LOG_2026-10-01_to_10-22.md#L3619)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [price_CB1.json](price_CB1.json) · result
- [price_OFF1.json](price_OFF1.json) · result
- [price_ON1.json](price_ON1.json) · result
- [WALL_PRICE_v1.json](WALL_PRICE_v1.json) · result
- [guard_CB1.json](guard_CB1.json) · result
- [guard_OFF1.json](guard_OFF1.json) · result
- [guard_OFF_7103247,7103248.json](guard_OFF_7103247,7103248.json) · result
- [guard_ON1.json](guard_ON1.json) · result
- [guard_ON_7103247,7103248.json](guard_ON_7103247,7103248.json) · result
- [log_CB1.txt](log_CB1.txt) · result
- [log_OFF.txt](log_OFF.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-grouped-read-kernels-save-0-039-bill-per-added) · [Copy provenance](../../../PROVENANCE.json)

