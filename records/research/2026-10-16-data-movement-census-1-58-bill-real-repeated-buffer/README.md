# Data-movement census: 1.58% of bill, real repeated-buffer cache opportunity below 0.03%

2026-10-16T06:22:55Z · finding · rejected · measured

Most data movement was structural kernel work or fresh-buffer address reuse that could not safely be cached by client object identity. Several small exact rewrites remained.

Recorded measurement:

    245.03GF;reshape.466%,copyto.365%,scatter.197%,gather.182%,stack.117%,astype.098%,asarray.065%;apparentrepeats.056%;real<.03%

## Programs

- [repeat_census_v1.py](repeat_census_v1.py)
- [repeat_census_v2.py](repeat_census_v2.py)
- [site_census_v1.py](site_census_v1.py)
- [../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py](../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3953](../../TEAM_LOG_2026-10-01_to_10-22.md#L3953)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EXACT_DM_CENSUS_SUMMARY_v1.json](EXACT_DM_CENSUS_SUMMARY_v1.json) · result
- [guard_census_gov3.json](guard_census_gov3.json) · result
- [guard_repeat2_gov3.json](guard_repeat2_gov3.json) · result
- [guard_repeat_gov3.json](guard_repeat_gov3.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-data-movement-census-1-58-bill-real-repeated-buffer) · [Copy provenance](../../../PROVENANCE.json)

