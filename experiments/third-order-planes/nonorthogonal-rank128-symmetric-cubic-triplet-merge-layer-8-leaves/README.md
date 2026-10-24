# Nonorthogonal rank128 symmetric cubic-triplet merge at layer 8 leaves query residual 0.566 versus limit 0.01

2026-09-20 · experiment · rejected · measured

An unrestricted two-sweep alternating-least-squares CP fit of rank 128 on the saved lazy-rotation rank-35 entry-8 old 6144-atom packet, explicitly symmetrised over all six permutations into 128 mixed cubic triples (384 stored columns inside the retained 384-dimensional space), was tested as a different constructor from the old orthogonal cube projection. Both primary fidelity ratios exceed the fixed 0.01 limit by a wide margin, closing the constructor without a rank, seed, sweep or ridge rescue.

Recorded measurement:

    residual energy / original signal: 512 two-leg Gaussian queries 0.5662357525; complete next repeated-moment plane 0.2257116323; cubic diagonal 0.1282944409; full symmetrised tensor 0.5920465631; unsymmetrised fit 0.6298104239; limit 0.01; construction GEMM subtotal 21,365,784,576 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/nonorthogonal_triplet_merge_information_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/information.py](code/information.py)
- [code/unit.py](code/unit.py)
- [code/algebra.py](code/algebra.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)
- [NEGATIVE_RESULT.md#L16](NEGATIVE_RESULT.md#L16)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-nonorthogonal-rank128-symmetric-cubic-triplet-merge-layer-8-leaves) · [Copy provenance](../../../PROVENANCE.json)

