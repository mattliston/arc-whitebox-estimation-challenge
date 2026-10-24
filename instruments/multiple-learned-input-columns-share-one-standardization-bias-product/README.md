# Multiple learned-input columns share one standardization and bias product with exact single-column parity

2026-10-02T19:50Z · tool · informational · measured

A multi-column native fragment stores the per-unit vectors where they are formed, standardizes their stacked rows and applies one product to the first-layer bias. Its single-column case reproduces the submitted file byte for byte, including bill and request count, and a seven-column smoke test runs successfully.

Recorded measurement:

    single-column prediction hash e0c2f7f2cef6e4df,43,049 requests; final-layer three-column price estimate0.05% bill and105 requests; hidden-layer extra two columns about200 requests

## Programs

- [code/fragment_hc_v2.py](code/fragment_hc_v2.py)
- [code/unit_hc_v2.py](code/unit_hc_v2.py)
- [code/weights_from_trainer.py](code/weights_from_trainer.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L452](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L452)
- [TEAM_LOG_2026-10-01_to_10-22.md#L466](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L466)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-multiple-learned-input-columns-share-one-standardization-bias-product) · [Copy provenance](../../PROVENANCE.json)

