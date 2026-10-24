# Replacing 6144 signed atoms with 384 orthogonal cubes leaves query residual 0.9540261570726674

2026-09-19 · experiment · rejected · measured

Keeping the old linear span did not preserve the tensor's mixed cubic coordinates under a one-slice orthogonal decomposition.

Recorded measurement:

    Phase 2 only. The fixed constructor retains the old 384-dimensional physical
    span but replaces its 6,144 signed cubic atoms with 384 orthogonal cubes.
    saved current-Lazy rank35 entry8 state, before its next join.
    Both prescribed squared residual/signal gates of <=0.01 FAIL:
    | 512 Gaussian two-leg queries after the recorded weight push | 0.9540261570726674 |
    | Complete 1024 by 1024 repeated-coordinate plane after that push | 0.8740800687347923 |
    | Cubic diagonal, descriptive only | 0.8504987027021403 |
    | Full old tensor before that push, all signed cross terms | 0.964405347741994 |
    The full old tensor norm squared is 21.143643621508645; the retained
    orthogonal-cube energy is 0.7526006421748065. Thus keeping the linear span
    31.20866772543698. These are modeled-state fidelity results, not final mean
    largest scaled dense identity error is 4.541183937712605e-12. On the saved
    3.2277073571993677e-15. No numerical or resource failure masks the result.
    2.6365506649017334 seconds. Peak including worker self-report is
    600621056 bytes and added swap is zero. No new estimator calls, mean
    merge is about 2.76 MB. No full 6144-square diagnostic Gram was saved.
    Conventional FP64 GEMM leading billed subtotals: construction 11,400,118,272;
    qualification 1,283,457,024; diagnostics 158,846,681,088. QR, eigendecomposition,
    Close only this one-slice, 384-atom orthogonal constructor. No different
    raw 2.169264529783277e-8. The 4e-9 bank is achieved; 3e-9, 2e-9 and the
    approximately 1e-9 goal remain unmet and active. Qualified Phase 2 submissions

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/orthogonal_cubic_merge_information_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/algebra.py](code/algebra.py)
- [code/finish.py](code/finish.py)
- [code/information.py](code/information.py)
- [code/unit.py](code/unit.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L7](NEGATIVE_RESULT.md#L7)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L1110](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1110)
- [REEXAMINATION_INDEX_20260919.md#L1113](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1113)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-replacing-6144-signed-atoms-384-orthogonal-cubes-leaves-query) · [Copy provenance](../../../PROVENANCE.json)

