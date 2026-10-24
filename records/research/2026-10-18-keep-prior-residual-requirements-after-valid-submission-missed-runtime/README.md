# Keep prior residual requirements after a valid submission missed the runtime cushion and score target

2026-10-18T12:10:33Z · decision · informational · measured

The graded compact repair improved score but missed both the 1.445e-9 target and the prior residual cushion. The prior runtime requirements were retained and further submissions required recalibration; native source activation preserved training parameters and made no PyTorch parity claim.

Recorded measurement:

    Score 1.4468565569762618e-9; median residual .3516070825, max .399547937; prior cushion .3452224775/.390682602; max headroom .000452063 s.

## Programs

- [../../../submissions/336935/estimator.py](../../../submissions/336935/estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5641](../../TEAM_LOG_2026-10-01_to_10-22.md#L5641)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GRADE_336935_FINAL_v1.json](GRADE_336935_FINAL_v1.json) · result
- [PUBLIC_FINAL_336935_v1.json](PUBLIC_FINAL_336935_v1.json) · result
- [eval_336935.json](../../../submissions/336935/eval_336935.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-keep-prior-residual-requirements-after-valid-submission-missed-runtime) · [Copy provenance](../../../PROVENANCE.json)

