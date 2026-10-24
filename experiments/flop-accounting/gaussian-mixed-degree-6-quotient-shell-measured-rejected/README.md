# Gaussian mixed degree-6 quotient shell: measured and rejected

2026-08-27 · experiment · rejected · measured

The exact mixed Hermite (1,2,3) shell is disjoint from the incumbent third-cumulant canonical-polyadic carrier (CP3) (1,1,2) birth, and its factor-36 diagram law plus repeated-plane quotient were verified exactly. On rank0, however, it changes raw mean squared error (MSE) only from 2.856783180e-8 to 2.854296094e-8. Late-q correlation is 0.02312 and projected adjusted score is 1.210095886e-8 at 932,289,050,615 FLOPs, above the live target.

Recorded measurement:

    The exact mixed Hermite (1,2,3) shell is disjoint from the incumbent CP3 (1,1,2) birth, and its factor-36 diagram law plus repeated-plane quotient were verified exactly. On rank0, however, it changes raw MSE only from 2.856783180e-8 to 2.854296094e-8. Late-q correlation is 0.02312 and projected adjusted score is 1.210095886e-8 at 932,289,050,615 FLOPs, above the live target. Receipt SHA-256 3458f0487b618805cd112b6ee29362d2f5266efb5fff64759b72dc08f9fb3e38.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py)
- [code/public_q24_gaussian_mixed_d6_shell_v1/run_source_replay.py](code/public_q24_gaussian_mixed_d6_shell_v1/run_source_replay.py)
- [code/public_q24_gaussian_mixed_d6_shell_v1/score_once.py](code/public_q24_gaussian_mixed_d6_shell_v1/score_once.py)
- [code/public_q24_gaussian_mixed_d6_shell_v1/verify_formula.py](code/public_q24_gaussian_mixed_d6_shell_v1/verify_formula.py)

## Evidence

- [TARGET_SCORE.json#L80](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/TARGET_SCORE.json#L80)

<details>
<summary>All result files and records (1)</summary>

- [TARGET_SCORE.json](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/TARGET_SCORE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-gaussian-mixed-degree-6-quotient-shell-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

