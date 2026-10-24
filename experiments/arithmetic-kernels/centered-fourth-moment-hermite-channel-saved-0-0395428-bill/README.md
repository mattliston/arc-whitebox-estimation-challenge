# Centered fourth-moment Hermite channel saved 0.0395428% bill in synthetic checks

2026-10-18T16:39:38Z · experiment · informational · measured

Centering the Hermite channel for the mixed fourth moment before its bilinear pair map incorporates three mean corrections and reduces the postprocess to the centered channel minus twice the squared covariance. Synthetic binary64 and binary32 checks agreed within their reported rounding errors; the component was held without a native accuracy or runtime claim.

Recorded measurement:

    Fourteen-layer saving 94,968,832 FLOPs, averaging 0.0395428% of current-source science32 bills; about 28 fewer numeric calls. At unchanged raw MSE, the record projects 1.44685656e-9 to 1.44628443e-9. Synthetic relative agreement: binary64 2.50e-16, binary32 1.16e-7.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v2.py](code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v2.py)
- [code/astra_oct18_stack_scalar_v1/build_centered_channel_v2.py](code/astra_oct18_stack_scalar_v1/build_centered_channel_v2.py)
- [code/astra_oct18_stack_scalar_v1/probe_centered_channel_v1.py](code/astra_oct18_stack_scalar_v1/probe_centered_channel_v1.py)
- [code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v2-7b203c5e.py](code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v2-7b203c5e.py)
- [code/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v1.py](code/astra_oct18_stack_scalar_v1/candidate_95c_centered22_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5718](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5718)
- [RESULT_v1.md#L5](RESULT_v1.md#L5)
- [RESULT_v1.md#L7](RESULT_v1.md#L7)
- [RESULT_v1.md#L9](RESULT_v1.md#L9)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SEARCH_STATE_v20.json](../../join-sketch/nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/SEARCH_STATE_v20.json) · record
- [SEARCH_STATE_v19.json](../../join-sketch/restoring-omitted-inherited-core-trace-had-negligible-inconsistent-signed/SEARCH_STATE_v19.json) · result
- [SUMMARY_v1.json](../../join-sketch/nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/SUMMARY_v1.json) · result
- [RESULT_v1.json](../../join-sketch/nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/RESULT_v1.json) · result
- [DIFF128_RESULT_v1.json](../../join-sketch/nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/DIFF128_RESULT_v1.json) · result
- [COMMON_BIAS_ORACLE_v1.json](../../join-sketch/nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/COMMON_BIAS_ORACLE_v1.json) · result
- [RESULT_v1.md](RESULT_v1.md) · record
- [BUILD_CENTERED22_v2.json](BUILD_CENTERED22_v2.json) · result
- [PROBE_CENTERED_CHANNEL_v1.json](PROBE_CENTERED_CHANNEL_v1.json) · result
- [CENTERED22_PROJECTED_PRICE_v1.json](CENTERED22_PROJECTED_PRICE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-centered-fourth-moment-hermite-channel-saved-0-0395428-bill) · [Copy provenance](../../../PROVENANCE.json)

