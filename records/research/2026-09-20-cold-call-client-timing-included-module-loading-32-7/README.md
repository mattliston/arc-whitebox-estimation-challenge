# Cold-call client timing included module loading: the 32.7% packing residual gain shrinks to 0.366% prediction-only

2026-09-20 · finding · informational · measured

Inspection of the installed whestbench SDK after #331682 was accepted showed that subprocess_worker.py loads the estimator and calls setup before the timed predict, so the conservative local qualification, which executed the module inside the first prediction budget, had credited module parsing time to the residual. Fixed prediction-only pairs on nets 35/100 gave parent residuals 0.180817789/0.178862699 s and packed residuals 0.182794069/0.175569976 s, a mean ratio of 0.996339965486, while module loading alone fell 80.25% (ratio 0.197491186227); packing therefore was not a demonstrated platform residual repair.

Recorded measurement:

    prediction-only residual ratio 0.996339965486 (0.366% difference); module-load mean ratio 0.197491186227 (80.25% reduction); conservative cold residual gain 32.7%; profiled module loading inside the first budget 0.131024625 s

## Evidence

- [SDK_TIMING_SCOPE.md#L3](SDK_TIMING_SCOPE.md#L3)
- [SDK_TIMING_SCOPE.md#L4](SDK_TIMING_SCOPE.md#L4)
- [SDK_TIMING_SCOPE.md#L10](SDK_TIMING_SCOPE.md#L10)
- [SDK_TIMING_SCOPE.md#L11](SDK_TIMING_SCOPE.md#L11)
- [PREDICTION_ONLY_RESULT.md#L3](PREDICTION_ONLY_RESULT.md#L3)
- [PREDICTION_ONLY_RESULT.md#L6](PREDICTION_ONLY_RESULT.md#L6)
- [PREDICTION_ONLY_RESULT.md#L7](PREDICTION_ONLY_RESULT.md#L7)
- [PROFILE_FINDINGS.md#L7](PROFILE_FINDINGS.md#L7)
- [REEXAMINATION_INDEX_20260919.md#L866](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L866)

<details>
<summary>All result files and records (5)</summary>

- [SDK_TIMING_SCOPE.md](SDK_TIMING_SCOPE.md) · record
- [PREDICTION_ONLY_RESULT.md](PREDICTION_ONLY_RESULT.md) · result
- [PROFILE_FINDINGS.md](PROFILE_FINDINGS.md) · result
- [closeout_after_sdk_probe_v1.json](closeout_after_sdk_probe_v1.json) · record
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- rejects: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../../../submissions/331682/README.md)
- ← measured by: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../../../submissions/331682/README.md)

[Research graph](../../../graph/index.html#node=2026-09-20-cold-call-client-timing-included-module-loading-32-7) · [Copy provenance](../../../PROVENANCE.json)

