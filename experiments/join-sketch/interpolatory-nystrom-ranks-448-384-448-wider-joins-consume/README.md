# Interpolatory Nyström ranks 448/384/448: wider joins consume the arithmetic savings

2026-09-26 · experiment · rejected · measured

Increasing interpolatory join ranks preserved the four-network training accuracy more closely, but native cost and adjusted score were worse.

Recorded measurement:

    Four-network raw -0.086%; native versus #332361: bill +0.621% / +0.408%, adjusted +3.652% / +1.008%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/candidate_interpolation_448-384-448_v1.py](code/codex_resume_20260926_v1/candidate_interpolation_448-384-448_v1.py)
- [code/codex_resume_20260926_v1/candidate_interpolation_448-384-448_v1_fragment.py](code/codex_resume_20260926_v1/candidate_interpolation_448-384-448_v1_fragment.py)
- [code/codex_resume_20260926_v1/build_interpolation_native.py](code/codex_resume_20260926_v1/build_interpolation_native.py)
- [code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py](code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py)

## Evidence

- [RESULTS.md#L24](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L24)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-interpolatory-nystrom-ranks-448-384-448-wider-joins-consume) · [Copy provenance](../../../PROVENANCE.json)

