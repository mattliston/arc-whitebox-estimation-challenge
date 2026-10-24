# Depth L-1 trees are no request lever: real depth drops save 4-11 requests a call at 0.75-3.8% bill per 1,000

2026-10-16T15:49:02Z · finding · rejected · measured

A census of S0 (warm network 100) put leaf matmuls at 78% of the bill and an analytic model suggested that running every tree product one level shallower would free about 1,900 requests at <= 0.25% bill per 1,000. The in-process probe of the product kernel at depth L and L-1 on all 47 input keys killed it: L6 to L5 is no depth change in this build (the padded six-level plan), real drops save only 4-11 requests a call at 0.75-3.8% bill per 1,000, and some L-1 shapes return wrong results. Five-full stayed until the byte-identical diets replaced it.

Recorded measurement:

    Leaf matmuls 78% of bill (hz strips 26%, ipt 25%, k-split 16%, d7 7%). 1024^3 L7->6: -8 requests, +7.5e7 flops = +0.030% bill a call = 3.8%/1,000; 1024x1024x2048 L7->6: +8 requests and +0.06%; L4->3 on 832/576x384x384: -8..-10 at 0.75-0.86%/1,000.

## Programs

- [probe_lm1_v1.py](probe_lm1_v1.py)
- [../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py](../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4304](../../TEAM_LOG_2026-10-01_to_10-22.md#L4304)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4306](../../TEAM_LOG_2026-10-01_to_10-22.md#L4306)

<details>
<summary>All result files and records (3)</summary>

- [probe_lm1_v1.json](probe_lm1_v1.json) · result
- [LM1_CENSUS_v2.txt](LM1_CENSUS_v2.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-depth-l-1-trees-no-request-lever-real-depth) · [Copy provenance](../../../PROVENANCE.json)

