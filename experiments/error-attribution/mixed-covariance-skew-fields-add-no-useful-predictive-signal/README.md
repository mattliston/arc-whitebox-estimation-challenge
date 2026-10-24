# Mixed covariance and skew fields add no useful predictive signal in the historical residual screen

2026-10-01T02:41Z · experiment · rejected · measured

A fixed ridge screen augments two local skew fields with two fields formed from the product of off-diagonal standardized skew and covariance matrices. Network-held-out fits use six historical networks at layers 10–12 and score covariance and following-response residuals. The augmented fields explain far less than the required residual fraction and worsen the response comparison against the local control, so no native port is admitted.

Recorded measurement:

    ridge 0.001; six networks × three layers; local covariance/response loss ratios 0.996811269500633/0.9973792380270088; augmented 0.9966483141442749/0.9980045734878765; covariance wins 6/6, response wins 5/6; required both ratios ≤0.8 and improvement ≥0.02

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_mixed_covariance_skew_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/screen.py](code/screen.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L114](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L114)
- [HILL_CLIMB_CURRENT.md#L120](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L120)
- [result.json#L1](result.json#L1)
- [closeout.json#L1](closeout.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · record
- [result.json](result.json) · result
- [sufficient_statistics.json](sufficient_statistics.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-mixed-covariance-skew-fields-add-no-useful-predictive-signal) · [Copy provenance](../../../PROVENANCE.json)

