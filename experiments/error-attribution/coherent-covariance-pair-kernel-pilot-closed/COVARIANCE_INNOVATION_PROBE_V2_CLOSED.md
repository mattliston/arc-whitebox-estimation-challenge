# Coherent covariance pair-kernel pilot: closed

September 5, 2026. Eight existing public 8M empirical mean/covariance
trajectories, layers 8..14; six train, held-out ranks 400/500. Current
host features frozen before current targets; past mean/covariance supplied
as explicit oracle state. No deployable or closed-loop score claim.

The fixed symmetric 315-coefficient response pair kernel completed its
predeclared 100-epoch MPS fit in 7.90 seconds. Final held-out ratios to
the zero-correction baseline:

| Rank | Covariance-error MSE ratio | Next-mean-response MSE ratio |
| --- | --- | --- |
| 400 | 0.9456272230 | 0.9899835232 |
| 500 | 0.9371556228 | 0.9864905832 |

It fails the <=0.75 gate on both metrics for each network. No more labels,
200-net fit, coefficient/basis sweep, candidate port, panel, or submission
for this fixed response pair-kernel family. Stored MC covariances are not
independent replicas, so no noise-corrected ceiling is claimed.

Evidence: covariance_innovation_probe_rows_v2.json,
covariance_innovation_probe_verdict_v2.json, and
covariance_innovation_probe_model_v2.npz. Preserve partial v1 artifacts;
the missing 1e9 mean files and coherent v2 protocol are documented in
COVARIANCE_INNOVATION_PROBE_V2.md. Official bank remains 329926,
adjusted 4.982231460183304e-9; user goal below 1e-9 remains unfinished.
