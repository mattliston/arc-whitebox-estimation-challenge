# Tucker and hierarchical-Tucker third-cumulant recompression: source-only accuracy failed

2026-08-27 · experiment · rejected · measured

The r64/r96 Tucker fallback was superseded by the more expressive frozen r128/k32 quotient nested-HT teacher reconstruction. Even this favorable source-only screen fails: quotient raw mean squared error (MSE) is 6.77351956876131e-8 and the unprojected raw control is 6.66858015624021e-8 versus a 3.060640051205329e-8 gate at 551,371,243,519 projected FLOPs. Receipt SHA-256 b2080ee26fc6c9b3b2715ffc502d3bed4798a7b132a29e76df7c3227f82e68bd.

Recorded measurement:

    The r64/r96 Tucker fallback was superseded by the more expressive frozen r128/k32 quotient nested-HT teacher reconstruction. Even this favorable source-only screen fails: quotient raw MSE is 6.77351956876131e-8 and the unprojected raw control is 6.66858015624021e-8 versus a 3.060640051205329e-8 gate at 551,371,243,519 projected FLOPs. Receipt SHA-256 b2080ee26fc6c9b3b2715ffc502d3bed4798a7b132a29e76df7c3227f82e68bd.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/cost_audit.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/cost_audit.py)
- [code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py](code/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py)

## Evidence

- [TARGET_SCORE.json#L1](../cheaper-pure-closure-third-cumulant-canonical-polyadic-carrier-state/TARGET_SCORE.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [TARGET_SCORE.json](../cheaper-pure-closure-third-cumulant-canonical-polyadic-carrier-state/TARGET_SCORE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-tucker-hierarchical-tucker-third-cumulant-recompression-source-only-accuracy) · [Copy provenance](../../../PROVENANCE.json)

