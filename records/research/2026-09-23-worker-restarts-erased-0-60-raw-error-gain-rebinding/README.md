# Worker restarts erased a 0.60% raw-error gain by rebinding cold frames

2026-09-23T12:24Z · incident · informational · measured

All five workers repeated setup during the submission, adding metered frame binding on five public networks. The resulting utilization noise obscured the model improvement.

Recorded measurement:

    - 2026-09-23 05:24 lead: #332005 (S6b) GRADED 3.1758e-9: ties S5e (3.1763e-9). Raw -0.59% (refit transfers; 40/50 nets better) but util +0.51%: the platform set up each worker twice (10 setups vs 5), doubling the cold first-MLPs, each ~+4-4.5% flops for frame binding. Median per-MLP flops change +0.01%. So the cold-MLP binding cost is now a score-noise term (~±0.25% util depending on the platform's restart count); cutting cold binding is worth more than it looked. Telemetry: worst wall 103.0 s (S5e 108.0), residual max 0.335 s.
    2026-09-23 05:33 [engineer2] #332005 (S6b) TELEMETRY vs #331992 (S5e) (telemetry_compare.py 332005 331992): score 3.17578e-9 vs 3.17628e-9 (flat). Raw MSE on the 50 public MLPs -0.60% mean (-0.73% median, 40/50 wins: the refit works) but util +0.262% mean, because ALL FIVE WORKERS RE-RAN SETUP mid-run (10 setup_executions vs 5; worker processes restarted, caches cold): the next MLP on each worker, indices 17/18/19/20/21, billed +3.9/+3.9/+3.4/+4.3/+4.5% (all five PUBLIC), plus the first-of-worker 0-4 at +0.33% each (batched-frame binding). No failures, all generation 0; the same happened to #331947 (2 workers) and #331539 (w-3 x3), so it is platform-side. Without it S6b would sit near -0.5%. Margins: wall mean 91.96 s, max 103.0 s (S5e 108.0); cold 0-4 mean 95.0 / max 103.0 (S5e 97.9 / 108.0), warm max 99.1 (S5e 102.2). Residual mean 0.25 s, p95/p99/max 291/328/335 ms (S5e 304/334/343), -18.2 ms mean (88/100 lower); cold 0-4 mean 305 / max 335, warm max 298; the restart-cold MLPs 17-21 carry the next-highest residuals (287-298 ms). corr(overhead, residual) +0.95.

## Programs

- [telemetry_compare.py](telemetry_compare.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L203](../../TEAM_LOG_2026-09-22_to_09-26.md#L203)
- [TEAM_LOG_2026-09-22_to_09-26.md#L204](../../TEAM_LOG_2026-09-22_to_09-26.md#L204)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../../../submissions/332005/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-worker-restarts-erased-0-60-raw-error-gain-rebinding) · [Copy provenance](../../../PROVENANCE.json)

