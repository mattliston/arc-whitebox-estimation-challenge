# Clipped regression-slope gain for inherited third-order transport at layers 1-14: adjusted +1.39%/+3.30% on nets 35/100

2026-09-20 · experiment · rejected · measured

The candidate exposes the already computed pair-map second raw moment and uses q = clip((raw2 - mu*host)/pre_v, 0, 1) as the gain for the inherited CP (third-cumulant factor) transport at layers 1 to 14 of the lazy-rotation card, keeping all learned literals, rank 384 and the schedule. Both development networks got worse before cost, so the exact native rule is closed; the record notes the known counterexample that at alpha 0 the regression slope gives the wrong sign for the mixed marginal-third against the all-distinct third term.

Recorded measurement:

    raw ratios 1.013756229998082 (net 35) and 1.0328132673079302 (net 100); adjusted ratios 1.0139079577751877 / 1.0329667629978465, pooled 1.0233986556259882 vs the <=0.95 gate; +59,580,332 billed FLOPs per call (cold bills 398,140,498,029 / 400,952,643,804 vs parent 398,080,917,697 / 400,893,063,472)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/regression_slope_transport_v1/build.py](code/regression_slope_transport_v1/build.py)
- [code/regression_slope_transport_v1/candidate.py](code/regression_slope_transport_v1/candidate.py)
- [code/regression_slope_transport_v1/score_dev.py](code/regression_slope_transport_v1/score_dev.py)
- [code/regression_slope_transport_v1/finish.py](code/regression_slope_transport_v1/finish.py)
- [code/regression_slope_transport_v1/unit.py](code/regression_slope_transport_v1/unit.py)
- [code/regression_slope_transport_v1/verify_null.py](code/regression_slope_transport_v1/verify_null.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/regression_slope_transport_v1/kernel.py](code/regression_slope_transport_v1/kernel.py)
- [code/regression_slope_transport_v1/null.py](code/regression_slope_transport_v1/null.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [NEGATIVE_RESULT.md#L8](NEGATIVE_RESULT.md#L8)
- [NEGATIVE_RESULT.md#L18](NEGATIVE_RESULT.md#L18)
- [NEGATIVE_RESULT.md#L19](NEGATIVE_RESULT.md#L19)
- [NEGATIVE_RESULT.md#L21](NEGATIVE_RESULT.md#L21)
- [NEGATIVE_RESULT.md#L23](NEGATIVE_RESULT.md#L23)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-clipped-regression-slope-gain-inherited-third-order-transport-layers) · [Copy provenance](../../../PROVENANCE.json)

