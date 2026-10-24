# Scatter-based gathers adopted in the higher-request variant: -0.074..-0.078% FLOPs, byte-identical

2026-10-12T01:24Z · decision · adopted · derived

Flopscope bills a fancy-index gather 4 per output element but put_along_axis 1 per source element; routing 9 gather sites through the scatter helper is byte-identical and became part of every twin; the rest of the byte-identical harvest (same-dtype astype, reshapes, repeats) is under 0.01%.

Recorded measurement:

    gathers 0.220% of bill; XSG -0.074..-0.078%, +261..+465 requests; reshapes 0.53% but inside the tree kernels; repeated computations 0.015%

## Programs

- [probe_gathers_v1.py](probe_gathers_v1.py)
- [make_xsg_v2.py](make_xsg_v2.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py](../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1261](../../TEAM_LOG_2026-10-01_to_10-22.md#L1261)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1262](../../TEAM_LOG_2026-10-01_to_10-22.md#L1262)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1263](../../TEAM_LOG_2026-10-01_to_10-22.md#L1263)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1264](../../TEAM_LOG_2026-10-01_to_10-22.md#L1264)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #335483: byte-identical scatter-based gather variant](../../../submissions/335483/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-scatter-based-gathers-adopted-higher-request-variant-0-074) · [Copy provenance](../../../PROVENANCE.json)

