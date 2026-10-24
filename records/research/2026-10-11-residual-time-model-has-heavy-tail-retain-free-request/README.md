# Residual-time model has a heavy tail; retain free request reductions without buying costly margin

2026-10-11T06:30Z · lesson · informational · measured

The loss model refit on 2,300 graded networks: per-network residual over its run's median has a power-law tail (p99.9 1.187, max 1.313), so a 1.29x spike kills even the safe file; byte-identical request cuts are worth taking, metered FLOPs-costing ones cost 3-5x what they buy; the candidate generation 47 (S47) safe file keeps candidate generation 46 (S46)'s counts and no extra-request twin.

Recorded measurement:

    p99 1.117, p99.9 1.187, max 1.313; generalized-Pareto shape +0.28 (+0.46 above 1.12); S46-count safe P(loss) 1.4% (fat 2.0%) per 100, 4.2% per 300; -30 ms of Python = about 4,200 requests

## Programs

- [loss_model_v2.py](loss_model_v2.py)
- [loss_model_v3.py](loss_model_v3.py)
- [twin_risk.py](twin_risk.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L993](../../TEAM_LOG_2026-10-01_to_10-22.md#L993)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1023](../../TEAM_LOG_2026-10-01_to_10-22.md#L1023)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1024](../../TEAM_LOG_2026-10-01_to_10-22.md#L1024)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1096](../../TEAM_LOG_2026-10-01_to_10-22.md#L1096)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1879](../../TEAM_LOG_2026-10-01_to_10-22.md#L1879)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Residual time is dominated by client requests: estimator Python accounts for only 3-5%](../2026-10-11-residual-time-dominated-client-requests-estimator-python-accounts-only/README.md)
- shipped in: [Submission #335340: ridge-re-sized refit, request caches and seven-level full-packet strip pushes](../../../submissions/335340/README.md)
- ← enables: [Submission #335329 fails: network 14 exceeds the residual cap at 0.4083 s](../../../submissions/335329/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-residual-time-model-has-heavy-tail-retain-free-request) · [Copy provenance](../../../PROVENANCE.json)

