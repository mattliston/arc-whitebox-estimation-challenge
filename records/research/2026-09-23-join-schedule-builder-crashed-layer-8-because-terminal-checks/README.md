# Join-schedule builder crashed at layer 8 because terminal checks ran before the final join

2026-09-23T07:22Z · incident · failed · measured

The corrected builder applies the terminal join-count assertion only after the last scheduled join. Rebuilt outputs match the previously submitted estimator.

Recorded measurement:

    2026-09-23 00:22 porter: GN knobs native (build_gn_knobs.py): certified vs definition (gap 1e-4), VALID40 parity native 0.98342 vs torch 0.98337 knobs/row (corr .973); build_stack.sh (e2 v7 default, --gn/--refit/kernel flags) reproduces S3 and S3+GN exactly. WARNING: fable_join_v1/build_join_schedule.py (23:11) crashes every3 at L8; porter chains use the v1 fired builder. S3+GN VALID40 running.
    - 2026-09-23 00:23 lead: JOIN BUILDER FIX: my 23:11 revision of fable_join_v1/build_join_schedule.py crashed every3 builds at L8 (porter caught it; the terminal join-count check ran at every non-join layer). Fixed copy fable_join_v1/build_join_schedule_v2.py applies the check only at layers >= the last scheduled join; S2 rebuilt with v2 is native bit-identical to the fired #331947 (r35 MSE 1.612368033799512e-08, flops 408,474,919,414; r100 1.576382292246693e-08). The broken file is kept unedited with BROKEN_build_join_schedule_2311.txt beside it. Fired S2 and all builds made with v1_fired_S2 are unaffected.

## Programs

- [build_gn_knobs.py](build_gn_knobs.py)
- [../../../experiments/late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/code/fable_porter_v1/build_stack.sh](../../../experiments/late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/code/fable_porter_v1/build_stack.sh)
- [build_join_schedule.py](build_join_schedule.py)
- [build_join_schedule_v2.py](build_join_schedule_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L117](../../TEAM_LOG_2026-09-22_to_09-26.md#L117)
- [TEAM_LOG_2026-09-22_to_09-26.md#L118](../../TEAM_LOG_2026-09-22_to_09-26.md#L118)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [BROKEN_build_join_schedule_2311.txt](BROKEN_build_join_schedule_2311.txt) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-join-schedule-builder-crashed-layer-8-because-terminal-checks) · [Copy provenance](../../../PROVENANCE.json)

