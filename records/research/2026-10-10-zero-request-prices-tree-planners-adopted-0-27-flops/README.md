# Zero request prices in tree planners adopted: -0.27% FLOPs for 311 additional first-network requests

2026-10-10T08:05Z · finding · adopted · measured

Probing the tree planners' request prices at a quarter and at zero on candidate generation 42 (S42) + (a): every price 0 picks deeper plans worth -0.27% / -0.20% of metered FLOPs (first / warm) for +311 / +576 requests, about 1.3M flops of metered FLOPs per extra request.

Recorded measurement:

    price / 4: -0.24% / -0.17% bill for +252 / +426 requests; price 0: -0.27% / -0.20% for +311 / +576; kernel +2.1-2.5%

## Programs

- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py)
- [candidate.py](candidate.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L521](../../TEAM_LOG_2026-10-01_to_10-22.md#L521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L577](../../TEAM_LOG_2026-10-01_to_10-22.md#L577)
- [TEAM_LOG_2026-10-01_to_10-22.md#L581](../../TEAM_LOG_2026-10-01_to_10-22.md#L581)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Estimated request cost: about 125k FLOPs for one submission and 20-25k for a designated pair](../2026-10-10-estimated-request-cost-about-125k-flops-one-submission-20/README.md)
- shipped in: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../../../submissions/335147/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-zero-request-prices-tree-planners-adopted-0-27-flops) · [Copy provenance](../../../PROVENANCE.json)

