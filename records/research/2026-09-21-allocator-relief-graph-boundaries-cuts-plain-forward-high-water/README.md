# Allocator relief at graph boundaries cuts the plain forward high-water from 6,793,330,688 to 2,965,045,248 bytes

2026-09-21T11:20Z · tool · adopted · measured

With the model source and arithmetic unchanged, calling garbage collection plus the documented unused-mapping pressure relief (malloc_zone_pressure_relief) at graph boundaries let the plain35/checkpointed35/checkpointed100/checkpointed35 training sequence complete under a 7.5 GB trigger with finite gradients, byte-exact repeat gradients and unchanged parameters through three zero-rate Adam steps. The raw allocator counters were found inconsistent (455 bad before/after snapshots) and are not used for ownership claims; the schedule was adopted as the resource basis of the 160-network fit.

Recorded measurement:

    plain35 high-water 6,793,330,688 B before vs 2,965,045,248 B after; whole sequence peak 5,706,481,664 B; wall 99.509694 s; backward durations 29.684643416665494 / 30.11640950012952 / 29.870359583757818 s; native rank-35 final RMS discrepancy of the CPU engine 4.0870829656109e-06

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L16](RESULT.md#L16)
- [RESULT.md#L22](RESULT.md#L22)
- [REEXAMINATION_INDEX_20260919.md#L500](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L500)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [qualification_result.json](qualification_result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- enables: [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](../../../experiments/learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md)
- shipped in: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)
- ← enables: [CPU estimator mean-encoder refit: first backward pass exceeded the 8 GB process-tree memory guard](../2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/README.md)
- ← enables: [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](../../../experiments/learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-allocator-relief-graph-boundaries-cuts-plain-forward-high-water) · [Copy provenance](../../../PROVENANCE.json)

