# Damping the failed terminal-readout direction has a descriptive gain ceiling below 0.01%

2026-10-01T04:36Z · finding · informational · derived

A post-failure quadratic decomposition of the saved out-of-fold terminal-readout direction separates its helpful linear component from a larger positive variance term. The fitted descriptive secant minimum could improve the pooled loss by only 0.00987%. This calculation is not a validation result or a newly selected candidate, and no damping or ridge rescue is attempted.

Recorded measurement:

    linear contribution −0.161932615031%; quadratic contribution +0.664086232023%; descriptive minimizing alpha 0.121921376488; maximum descriptive gain 0.00987152366145%

## Programs

- [failure_diagnostic.py](failure_diagnostic.py)
- [../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/code/collect.py](../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/code/collect.py)
- [../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/code/prepare_capture.py](../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/code/prepare_capture.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L38](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L38)
- [closeout.json#L19](../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/closeout.json#L19)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [failure_diagnostic.json](failure_diagnostic.json) · result
- [failure_diagnostic_guard.json](failure_diagnostic_guard.json) · result
- [closeout.json](../../../experiments/learned-correction/adding-64-terminal-hidden-state-directions-worsens-every-cross/closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-damping-failed-terminal-readout-direction-has-descriptive-gain-ceiling) · [Copy provenance](../../../PROVENANCE.json)

