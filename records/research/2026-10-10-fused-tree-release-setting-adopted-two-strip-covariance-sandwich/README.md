# Fused-tree release setting adopted; two-strip covariance sandwich rejected at +0.27% FLOPs

2026-10-10T03:05Z · decision · adopted · measured

Pricing the safe variant by the podium model: no un-fused trees (a) costs +0.03% metered FLOPs and halves P(network loss), so it goes into every bank from candidate generation 43 (S43) on; the two-strip covariance sandwich (d) costs +0.27% metered FLOPs and loses at every board position.

Recorded measurement:

    (a) 42,610 / 41,581 requests (-1,086 / -1,162), bill +0.02..0.05%, P(loss) 2.2% -> 1.1%; (d) +0.27% bill for -2,660 requests: podium change -1.31 / -4.61 / -0.32 points at x1.00 / x0.96 / x0.93

## Programs

- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/variant_table.py)
- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py)
- [../2026-10-10-worst-residuals-isolated-timing-spikes-1-11-1-17x/platform_networks.py](../2026-10-10-worst-residuals-isolated-timing-spikes-1-11-1-17x/platform_networks.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L479](../../TEAM_LOG_2026-10-01_to_10-22.md#L479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L493](../../TEAM_LOG_2026-10-01_to_10-22.md#L493)
- [TEAM_LOG_2026-10-01_to_10-22.md#L503](../../TEAM_LOG_2026-10-01_to_10-22.md#L503)
- [TEAM_LOG_2026-10-01_to_10-22.md#L505](../../TEAM_LOG_2026-10-01_to_10-22.md#L505)
- [TEAM_LOG_2026-10-01_to_10-22.md#L511](../../TEAM_LOG_2026-10-01_to_10-22.md#L511)
- [TEAM_LOG_2026-10-01_to_10-22.md#L512](../../TEAM_LOG_2026-10-01_to_10-22.md#L512)
- [TEAM_LOG_2026-10-01_to_10-22.md#L515](../../TEAM_LOG_2026-10-01_to_10-22.md#L515)
- [TEAM_LOG_2026-10-01_to_10-22.md#L514](../../TEAM_LOG_2026-10-01_to_10-22.md#L514)

<details>
<summary>All result files and records (2)</summary>

- [a16_loss_risk.json](a16_loss_risk.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Estimated request cost: about 125k FLOPs for one submission and 20-25k for a designated pair](../2026-10-10-estimated-request-cost-about-125k-flops-one-submission-20/README.md)
- shipped in: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../../../submissions/335147/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-fused-tree-release-setting-adopted-two-strip-covariance-sandwich) · [Copy provenance](../../../PROVENANCE.json)

