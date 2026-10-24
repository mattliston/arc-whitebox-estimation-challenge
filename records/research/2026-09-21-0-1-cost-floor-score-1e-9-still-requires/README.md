# At the 0.1 cost floor, score 1e-9 still requires at least 47.6313% less raw error

2026-09-21 · finding · informational · measured

The official score and raw error establish a cost-floor bound and distinguish uniform hypothetical cost savings from an implemented estimator improvement.

Recorded measurement:

    The precise reported adjusted score is 3.50773885697508e-09, and reported
    networks and zero failures; it exposes50 rounded per-network score rows.
    With identical predictions, the multiplier floor .1 gives adjusted score
    at least 1.9095373282596e-09. Cost reduction alone therefore cannot reach the
    reference target1e-9. Mean raw MSE must reach1e-8 or better even at the floor:
    a reduction of at least 47.6313% from the reported raw MSE.
    71.4916% to reach that point target.
    For3e-9, unchanged predictions with uniform effective-cost scaling would
    need a reduction of 14.4748%. Every recorded cost would remain
    The current528-parameter covariance fit remains the sole numerical experiment.
    The bounded public review found a participant report on factorized K3 and
    https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [REEXAMINATION_INDEX_20260919.md#L329](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L329)
- [REEXAMINATION_INDEX_20260919.md#L336](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L336)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-0-1-cost-floor-score-1e-9-still-requires) · [Copy provenance](../../../PROVENANCE.json)

