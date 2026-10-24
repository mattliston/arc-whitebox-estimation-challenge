# Estimated request cost: about 125k FLOPs for one submission and 20-25k for a designated pair

2026-10-10T03:16Z · lesson · informational · measured

The tree planners priced a request at 750k flops; by the podium model a request is worth about 125k flops today (150k near the line) for a single file, and only 20-25k for the primary of a designated pair because the safe file covers a loss.

Recorded measurement:

    1,000 requests at S42's count = +2.5 points of P(loss) x P(podium) vs 0.05-0.06% bill; pair: a primary request costs 0.14-0.22 of its single-file risk price

## Programs

- [a16_pair.py](a16_pair.py)
- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L521](../../TEAM_LOG_2026-10-01_to_10-22.md#L521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L527](../../TEAM_LOG_2026-10-01_to_10-22.md#L527)
- [TEAM_LOG_2026-10-01_to_10-22.md#L529](../../TEAM_LOG_2026-10-01_to_10-22.md#L529)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Designated pair uses the lower-FLOP submission and a safer implementation of the same estimator](../2026-10-10-designated-pair-uses-lower-flop-submission-safer-implementation-same/README.md)
- enables: [Zero request prices in tree planners adopted: -0.27% FLOPs for 311 additional first-network requests](../2026-10-10-zero-request-prices-tree-planners-adopted-0-27-flops/README.md)
- ← enables: [Fused-tree release setting adopted; two-strip covariance sandwich rejected at +0.27% FLOPs](../2026-10-10-fused-tree-release-setting-adopted-two-strip-covariance-sandwich/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-estimated-request-cost-about-125k-flops-one-submission-20) · [Copy provenance](../../../PROVENANCE.json)

