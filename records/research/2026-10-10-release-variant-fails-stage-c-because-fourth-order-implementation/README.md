# Release variant fails stage C because a fourth-order implementation lacks the per-layer switch table

2026-10-10T22:11Z · incident · failed · derived

The release lane's no-diet twin swapped in cohort-age local fourth-order term (K4L) v7, which has no _K4L_E table that the engineer's per-layer K4L switch per-layer literal switch (DLY@) reads; stage A cannot see errors past layer 10 and stage C caught it in four minutes; the fragment was fixed to v2 and the engineer's twin f89b8951 submitted instead.

Recorded measurement:

    stage C NameError in the first network; DLY v2 rebuilt safe = same predictions (pred sha 83cfbcb2); K4Lv7 twin bill -0.078%, +419 ops

## Programs

- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py)
- [candidate.py](candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L709](../../TEAM_LOG_2026-10-01_to_10-22.md#L709)
- [TEAM_LOG_2026-10-01_to_10-22.md#L718](../../TEAM_LOG_2026-10-01_to_10-22.md#L718)
- [TEAM_LOG_2026-10-01_to_10-22.md#L720](../../TEAM_LOG_2026-10-01_to_10-22.md#L720)
- [TEAM_LOG_2026-10-01_to_10-22.md#L721](../../TEAM_LOG_2026-10-01_to_10-22.md#L721)
- [TEAM_LOG_2026-10-01_to_10-22.md#L722](../../TEAM_LOG_2026-10-01_to_10-22.md#L722)

<details>
<summary>All result files and records (2)</summary>

- [judge_s46_primary.txt](judge_s46_primary.txt) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #335261: kernel-price variant of the same estimator gives no FLOP advantage](../../../submissions/335261/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-release-variant-fails-stage-c-because-fourth-order-implementation) · [Copy provenance](../../../PROVENANCE.json)

