# Eager setup prefix size changes scheduling reliability after all keys are already fused

2026-09-30 · lesson · informational · derived

The released source normally fused all649 keys using128 eager and521 deferred bindings. Enlarging the eager prefix alone changes scheduling and reliability rather than adding a new steady-state compute saving.

Recorded measurement:

    Normal release setup:128 eager+521 deferred=649 fused keys;no larger-prefix candidate was selected or run.

## Programs

- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [build.py](build.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L551](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L551)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-eager-setup-prefix-size-changes-scheduling-reliability-after-all) · [Copy provenance](../../../PROVENANCE.json)

