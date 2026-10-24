# Refitted correction and final-layer pair-mean truncation: held-out adjusted score improves 2.3550%

2026-09-26 · experiment · adopted · measured

A bundle combined the next learned-correction checkpoint and matched dials with inner-dimension split kernels, final-layer pair-mean truncation at 256 and the approximate future-basis core. All 28 exported tensors matched the trained checkpoint exactly.

Recorded measurement:

    Matched native VALID10 versus #332361: adjusted -2.3550%, paired bootstrap CI [-2.5672%, -2.0935%], 10/10 wins; raw -0.2191%, CI includes zero; native mean CPU +13.69%; peak group RSS 5.565 GB, no swap growth.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/build_bundle.py](code/codex_resume_20260926_v1/build_bundle.py)
- [code/codex_resume_20260926_v1/candidate_bundle_core_nystrom_v1.py](code/codex_resume_20260926_v1/candidate_bundle_core_nystrom_v1.py)
- [code/codex_resume_20260926_v1/candidate_bundle_precore_v1.py](code/codex_resume_20260926_v1/candidate_bundle_precore_v1.py)
- [code/codex_resume_20260926_v1/candidate_r8_literals_v1.py](code/codex_resume_20260926_v1/candidate_r8_literals_v1.py)
- [code/fable_engineer2_v1/depth7/p6_fragment_v2.py](code/fable_engineer2_v1/depth7/p6_fragment_v2.py)
- [code/fable_engineer2_v1/uvsplit/uvp_fragment_v0c.py](code/fable_engineer2_v1/uvsplit/uvp_fragment_v0c.py)
- [code/fable_engineer_v1/ksplit_fragment_v5w.py](code/fable_engineer_v1/ksplit_fragment_v5w.py)
- [code/fable_engineer_v1/ksplit_fragment_v6_final.py](code/fable_engineer_v1/ksplit_fragment_v6_final.py)
- [code/fable_engineer_v1/treuse_fragment_v2.py](code/fable_engineer_v1/treuse_fragment_v2.py)
- [code/fable_fire_v1/fire_candS28/candS28_S27_refit_ks_cap_sched_g5_mem.py](code/fable_fire_v1/fire_candS28/candS28_S27_refit_ks_cap_sched_g5_mem.py)

## Evidence

- [RESULTS.md#L37](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L37)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record
- [bundle_vs_s28_valid10_v1.json](bundle_vs_s28_valid10_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332492: approximate join core, shared star diagonal and refitted learned correction](../../../submissions/332492/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-refitted-correction-final-layer-pair-mean-truncation-held-out) · [Copy provenance](../../../PROVENANCE.json)

