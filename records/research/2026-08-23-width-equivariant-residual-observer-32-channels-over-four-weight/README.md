# Width-equivariant residual observer with 32 channels over four weight powers: design deferred

2026-08-23T20:08Z · decision · rejected · derived

Before the full-budget directive, the planned learned stage was a width-equivariant recurrent residual observer (32 hidden channels, 8 channels per weight power W, W^2, W^3, W^4, 20 local features, a 64-wide cell) sitting on a CP3 host under the 10% score floor; a static budget audit recommended the cap-1536 host with the T2 tier (128 hidden channels, 256 cell width) and a fast target-free ridge readout (fast_residual_ridge_v1) was frozen but held before activation. The directive to spend the whole 2^41 budget on the closure itself retired this floor-bound design before any fit.

Recorded measurement:

    host measurements used by the audit: cp3_cap1536 199,018,976,258 FLOPs, final raw MSE 3.7339069924135637e-6 (0.3091 percent worse than cap 1792); recommended combined static 210,004,501,370 FLOPs, conservative 212,201,606,393 with margin 7,700,719,162 below the floor; observer static baselines recurrent H32/P4/C64 2,056,671,096 operations, local 1,048,563,456

## Programs

- [ridge_science.py](ridge_science.py)
- [../../../experiments/series-expansions/hermite-pair-series-order-sweep-order-4-reached-3/code/analytic_hermite10_full_covariance/estimator.py](../../../experiments/series-expansions/hermite-pair-series-order-sweep-order-4-reached-3/code/analytic_hermite10_full_covariance/estimator.py)
- [verify_target_free.py](verify_target_free.py)

## Evidence

- [BUDGET_AUDIT.json#L1](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json#L1)
- [STATIC_METER_ESTIMATE.json#L1](STATIC_METER_ESTIMATE.json#L1)
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L22](FULL_BUDGET_CLOSURE_DIRECTIVE.md#L22)

<details>
<summary>All result files and records (8)</summary>

- [DESIGN.md](DESIGN.md) · record
- [STATIC_METER_ESTIMATE.json](STATIC_METER_ESTIMATE.json) · result
- [SCIENCE_FREEZE_V1.json](SCIENCE_FREEZE_V1.json) · record
- [BUDGET_AUDIT.json](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json) · result
- [DESIGN-55454d75.md](DESIGN-55454d75.md) · record
- [DESIGN-c0c9c6e6.md](DESIGN-c0c9c6e6.md) · record
- [SCALING_AND_GATES.json](SCALING_AND_GATES.json) · result
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md](FULL_BUDGET_CLOSURE_DIRECTIVE.md) · record

</details>

## Related work

- rejects: [Sampling-free closure accuracy target below 1e-8 raw error using up to the full 2^41 budget](../2026-08-24-sampling-free-closure-accuracy-target-below-1e-8-raw/README.md)

[Research graph](../../../graph/index.html#node=2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight) · [Copy provenance](../../../PROVENANCE.json)

