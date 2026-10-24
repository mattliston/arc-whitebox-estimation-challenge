# Old positive-tail cutoff 3.5: held-out adjusted gain 0.80237% remains below the submission threshold

2026-09-26 · experiment · inconclusive · measured

Only local reconstruction of old-tier strongly positive rows was omitted at layers7–14 while full carriers, pre-bases and young columns were preserved. Native training and ten held-out networks improved, but the source remained a subthreshold component.

Recorded measurement:

    TRAIN2 adjusted-.5211%;TRAIN6-.5979%,CI[-1.1441%,-.1582%],5/6wins;VALID10 adjusted-.80237%,CI[-1.27044%,-.31734%],8/10wins,raw-.16233%,bill-.62533%,CPU-2.05085%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_old_positive_tail_v1.py](code/candidate_old_positive_tail_v1.py)
- [code/build_old_positive_tail_v1.py](code/build_old_positive_tail_v1.py)
- [code/old_positive_tail_fragment_v1.py](code/old_positive_tail_fragment_v1.py)
- [code/candidate_bundle_diagreuse_v1.py](code/candidate_bundle_diagreuse_v1.py)
- [code/candidate_old_positive_tail_v2.py](code/candidate_old_positive_tail_v2.py)
- [code/candidate_old_positive_tail_v3.py](code/candidate_old_positive_tail_v3.py)

## Evidence

- [NEXT_RESEARCH.md#L47](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L47)

<details>
<summary>All result files and records (10)</summary>

- [NEXT_RESEARCH.md](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [old_positive_tail3_vs35_train6_v2_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail3_vs35_train6_v2_comparison.json) · result
- [old_positive_tail_train6_v2_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train6_v2_comparison.json) · result
- [old_positive_tail_train2_v1_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train2_v1_comparison.json) · result
- [old_positive_tail_valid10_v1_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_valid10_v1_comparison.json) · result
- [old_positive_tail_train8_v2_comparison_limits_v1.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train8_v2_comparison_limits_v1.json) · result
- [old_positive_tail_train2_v3_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train2_v3_comparison.json) · result
- [old_positive_tail_train6_v1_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train6_v1_comparison.json) · result
- [old_positive_tail3_vs35_train8_v2_provisional_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail3_vs35_train8_v2_provisional_comparison.json) · result
- [old_positive_tail_train8_v2_provisional_comparison.json](../old-positive-tail-cutoff-3-0-cache-matched-six/old_positive_tail_train8_v2_provisional_comparison.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-old-positive-tail-cutoff-3-5-held-out-adjusted) · [Copy provenance](../../../PROVENANCE.json)

