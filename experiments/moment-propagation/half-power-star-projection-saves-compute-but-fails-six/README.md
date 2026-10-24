# Half-power star projection saves compute but fails the six-network adjusted-score continuation

2026-09-26 · experiment · rejected · measured

The final Q.T@A pass was omitted while A.T@Q0 and A@V were retained, yielding truncated A V V.T. The exact-rank fixture passed, but raw error consumed the expected compute saving on the six-network continuation.

Recorded measurement:

    Initial2 adjusted-1.299%;next6 adjusted-.0380%,CI[-.6061%,+.4995%],raw+.5373%,bill-.5712%;unprofiled6 CPU-1.10%;profiled-control2 CPU comparison invalid.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_star_halfpower_v1.py](code/candidate_star_halfpower_v1.py)
- [code/build_star_halfpower_v1.py](code/build_star_halfpower_v1.py)
- [code/star_halfpower_fragment_v1.py](code/star_halfpower_fragment_v1.py)
- [code/candidate_bundle_diagreuse_v1.py](code/candidate_bundle_diagreuse_v1.py)

## Evidence

- [NEXT_RESEARCH.md#L31](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L31)

<details>
<summary>All result files and records (3)</summary>

- [NEXT_RESEARCH.md](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [star_halfpower_train6_v1_comparison.json](star_halfpower_train6_v1_comparison.json) · result
- [star_halfpower_train2_v1_comparison.json](star_halfpower_train2_v1_comparison.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-half-power-star-projection-saves-compute-but-fails-six) · [Copy provenance](../../../PROVENANCE.json)

