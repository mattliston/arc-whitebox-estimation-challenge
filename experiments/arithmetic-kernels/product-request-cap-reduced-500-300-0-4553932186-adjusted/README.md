# Product request cap reduced from 500 to 300: -0.4553932186% adjusted on 32 networks

2026-10-18T09:53:12Z · experiment · adopted · measured

The fixed two-million-FLOP-per-request product policy lowered its cap from 500 to 300 to recover residual margin. New rounding required a fresh seven-then-32 comparison, which confirmed a smaller arithmetic saving and passed local startup and prediction gates.

Recorded measurement:

    32 networks: adjusted -.4553932186% → 1.44489521412e-9; held 25: -.478989579%; 28/32 and 24/25 wins; raw +.060451%, bill -.515453%; normal/slow wall <=111.799.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/request_backup_cap300_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_request_backup_cap300_HELD_v1.py](code/candidate_request_backup_cap300_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5511](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5511)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5516](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5516)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5521](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5538](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5538)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5540](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5540)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [CAP300_RELEASE_SEAL_v1.json](CAP300_RELEASE_SEAL_v1.json) · record

</details>

## Related work

- shipped in: [Submission #336925: 300-request product cap with complete startup checks: failed setup on three workers](../../../submissions/336925/README.md)

[Research graph](../../../graph/index.html#node=2026-10-18-product-request-cap-reduced-500-300-0-4553932186-adjusted) · [Copy provenance](../../../PROVENANCE.json)

