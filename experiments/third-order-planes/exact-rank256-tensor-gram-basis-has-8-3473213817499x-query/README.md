# Exact rank256 tensor Gram basis has 8.3473213817499x query error versus rounded rank384 parent

2026-09-19 · experiment · rejected · measured

Removing random sketch approximation still left the exact mode-one basis substantially worse on modeled-state tensor diagnostics.

Recorded measurement:

    The deterministic rank-256 basis fails the predeclared information gate on
    both diagnostics. Against the actual rounded parent rank-384 compression,
    next-query squared residual energy is 8.3473213817499x
    7.3163411893286x. There are 0/512
    5.6326008685495. Both required ratios
    were <=0.75. No estimator endpoint, adjusted score, or production bill was
    This resolves the fixed rank-256 exact-Gram alternative to the earlier
    rank-384 compression. The prior sketch's 43.03259x/39.56989x failure is
    unchanged. This is not a universal lower bound for every rank-256 basis or
    was tried. The comparison has deliberately favorable FP64 trial arithmetic
    The modeled-state packet is the unchanged submitted Lazy rank35, entryL8,
    896 active rows and 8128 signed columns. Its original capture exactly
    reproduced all16 prediction arrays and the complete398080917697F bill.
    labels. Verification uses the same512 probes as the old diagnostic and is
    is 1.1852e-15, the minimum eigenvalue is
    1.0179e-14, normalized eigen residual is
    4.0544e-15, and maximum of
    4.9787e-15. The numerical gate passes.
    The256-dimensional basis retains
    99.656678050% of one-mode
    subtotal of 249824526336 operations,
    formation took 3.636253s locally; the whole information
    program took 9.313723s. The two guards completed and retired,
    peak process RSS 1340751872B, zero added swap. No new
    Official Lazy#331599 remains graded adjusted3.8164492657795035e-9,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_tensor_exact_gram_information_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/algebra.py](code/algebra.py)
- [code/information.py](code/information.py)
- [code/unit.py](code/unit.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L7](NEGATIVE_RESULT.md#L7)
- [REEXAMINATION_INDEX_20260919.md#L1416](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1416)
- [REEXAMINATION_INDEX_20260919.md#L1421](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1421)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-exact-rank256-tensor-gram-basis-has-8-3473213817499x-query) · [Copy provenance](../../../PROVENANCE.json)

