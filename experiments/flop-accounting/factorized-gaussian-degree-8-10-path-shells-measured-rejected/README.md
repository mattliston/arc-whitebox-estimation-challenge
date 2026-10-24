# Factorized Gaussian degree-8/10 path shells: measured and rejected

2026-08-27 · experiment · rejected · measured

The exact repeated-slice quotients for D8 paths (1,3,4)/(2,2,4) and D10 paths (1,4,5)/(2,3,5) were verified and replayed once. No arm is adjusted-positive. Best raw-only D8(1,3,4) reaches 2.856564055e-8 but projects 1.211016554e-8 adjusted; the cheapest adjusted arm D8(2,2,4) worsens raw and projects 1.194382394e-8.

Recorded measurement:

    The exact repeated-slice quotients for D8 paths (1,3,4)/(2,2,4) and D10 paths (1,4,5)/(2,3,5) were verified and replayed once. No arm is adjusted-positive. Best raw-only D8(1,3,4) reaches 2.856564055e-8 but projects 1.211016554e-8 adjusted; the cheapest adjusted arm D8(2,2,4) worsens raw and projects 1.194382394e-8. The D6+D8+D10 stack reaches raw 2.854023125e-8 but projects 1.243842515e-8. Dense triangle sectors require at least 13.194T added FLOPs and are cost-closed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py)
- [code/public_q24_gaussian_higher_path_shells_v1/run_source_replay.py](code/public_q24_gaussian_higher_path_shells_v1/run_source_replay.py)
- [code/public_q24_gaussian_higher_path_shells_v1/score_once.py](code/public_q24_gaussian_higher_path_shells_v1/score_once.py)
- [code/public_q24_gaussian_higher_path_shells_v1/verify_formula.py](code/public_q24_gaussian_higher_path_shells_v1/verify_formula.py)

## Evidence

- [TARGET_SCORE-1f23eb49.json#L1](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/TARGET_SCORE-1f23eb49.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [TARGET_SCORE-1f23eb49.json](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/TARGET_SCORE-1f23eb49.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-factorized-gaussian-degree-8-10-path-shells-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

