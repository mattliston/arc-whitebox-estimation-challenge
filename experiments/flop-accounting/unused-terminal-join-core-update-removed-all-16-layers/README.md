# Unused terminal join-core update removed: all 16 layers remain bit-identical

2026-09-26 · experiment · adopted · derived

The weighted Gram-core update at the final old-tier join at layer 13 was removed because no later join reads that core. Factors, the lazy coordinate map and prediction consumers were preserved.

Recorded measurement:

    Native ranks 35/100: bit-identical across all 16 layers; bill -0.3624% / -0.3583% versus the staged generation-29 candidate. Earlier-join symmetric blocks add only about 0.07% savings and change rounding.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/build_coretrim.py](code/codex_resume_20260926_v1/build_coretrim.py)
- [code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py](code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py)

## Evidence

- [RESULTS.md#L35](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L35)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record
- [coretrim_parity_v1.json](coretrim_parity_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332492: approximate join core, shared star diagonal and refitted learned correction](../../../submissions/332492/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-unused-terminal-join-core-update-removed-all-16-layers) · [Copy provenance](../../../PROVENANCE.json)

