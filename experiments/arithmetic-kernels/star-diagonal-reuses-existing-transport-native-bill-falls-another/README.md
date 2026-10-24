# Star diagonal reuses an existing transport: native bill falls another 0.3372% and 0.3393%

2026-09-26 · experiment · adopted · derived

With P=W.T@Q and T=(W.T**2)@B.T, the required star diagonal is sum(P*T, axis=1). Reusing T removed a redundant product by an exact algebraic reassociation with floating-point differences.

Recorded measurement:

    Additional bill savings 0.3372% / 0.3393%; final combined native smoke adjusted gains versus #332361 2.5031% / 2.1431%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_star_diagreuse.py](code/build_star_diagreuse.py)

## Evidence

- [RESULTS.md#L38](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L38)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record
- [bundle_diagreuse_comparison_v1.json](bundle_diagreuse_comparison_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332492: approximate join core, shared star diagonal and refitted learned correction](../../../submissions/332492/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-star-diagonal-reuses-existing-transport-native-bill-falls-another) · [Copy provenance](../../../PROVENANCE.json)

