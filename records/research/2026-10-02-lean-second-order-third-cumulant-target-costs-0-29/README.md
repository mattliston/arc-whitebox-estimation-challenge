# Lean second-order third-cumulant target costs 0.29% of bill and about 830 requests per network

2026-10-02T18:44Z · finding · informational · derived

Combining the second-order covariance and mixed-moment lists before evaluating them removes duplicate matrix passes. The exact combined target costs 0.29% of bill and about 830 requests; omitting correlation powers reduces it to 0.15% and about 560 requests, while the diagonal alone has negligible bill.

Recorded measurement:

    full 52 n x n passes/layer, 59 requests/layer, 0.29% bill and about 830 requests/network; no rho terms 26 passes, 0.15%, about 560; parts 1256: 33 passes, 0.18%, about 600; no rho with parts12: 16 passes, 0.09%, about 430; diagonal only about 280 requests

## Programs

- [base_from_request.py](base_from_request.py)
- [so2.py](so2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L422](../../TEAM_LOG_2026-10-01_to_10-22.md#L422)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-lean-second-order-third-cumulant-target-costs-0-29) · [Copy provenance](../../../PROVENANCE.json)

