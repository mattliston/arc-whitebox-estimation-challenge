# Shared-family canonical strides tripled three product walls despite byte and FLOP parity

2026-10-18T08:01:44Z · incident · failed · measured

A setup-saving shared-family layout preserved predictions and bill but changed strides enough to slow native add/subtract paths. Exact-layout repair recovered wall time while retaining byte parity, after which setup and residual margin still required qualification.

Recorded measurement:

    Wall +13.576%/+12.531%; projected max ~123.5 s >118; three products take 3x wall; exact layout 37.836/36.127 s, calibrated max 117.359 s.

## Programs

- [candidate_union_exactlayout_HELD_v1.py](candidate_union_exactlayout_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5400](../../TEAM_LOG_2026-10-01_to_10-22.md#L5400)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5410](../../TEAM_LOG_2026-10-01_to_10-22.md#L5410)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5413](../../TEAM_LOG_2026-10-01_to_10-22.md#L5413)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log
- [EXACT_LAYOUT_SETUP_DECISION_v1.json](EXACT_LAYOUT_SETUP_DECISION_v1.json) · record
- [EXACT_LAYOUT_NORMAL_DECISION_v1.json](EXACT_LAYOUT_NORMAL_DECISION_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-shared-family-canonical-strides-tripled-three-product-walls-despite) · [Copy provenance](../../../PROVENANCE.json)

