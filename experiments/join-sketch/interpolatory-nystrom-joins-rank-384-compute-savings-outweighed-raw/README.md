# Interpolatory Nyström joins at rank 384: compute savings are outweighed by raw error

2026-09-26 · experiment · rejected · measured

Setting all interpolatory Nyström joins to rank 384 saved compute but lost adjusted score in both native checks.

Recorded measurement:

    Four-network raw +2.932%; native versus #332361: bill -1.188% / -1.241%, raw +4.417% / +3.712%, adjusted +3.176% / +2.425%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/candidate_interpolation_384-384-384_v1.py](code/codex_resume_20260926_v1/candidate_interpolation_384-384-384_v1.py)
- [code/codex_resume_20260926_v1/candidate_interpolation_384-384-384_v1_fragment.py](code/codex_resume_20260926_v1/candidate_interpolation_384-384-384_v1_fragment.py)
- [code/codex_resume_20260926_v1/build_interpolation_native.py](code/codex_resume_20260926_v1/build_interpolation_native.py)
- [code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py](code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py)

## Evidence

- [RESULTS.md#L23](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L23)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-interpolatory-nystrom-joins-rank-384-compute-savings-outweighed-raw) · [Copy provenance](../../../PROVENANCE.json)

