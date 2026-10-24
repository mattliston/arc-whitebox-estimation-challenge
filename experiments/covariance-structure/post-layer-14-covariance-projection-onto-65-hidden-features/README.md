# Post-layer-14 covariance projection onto 65 hidden features removes 44-49% of error; final MSE ratio 1.002-1.012

2026-09-20 · experiment · rejected · measured

As a prerequisite for covariance supervision, the orthogonal Frobenius projection of the true-minus-estimated covariance of the lazy-rotation card after layer 14 onto the target-free sqrt(C_diag)*[H14,1] active basis (rank 65) was replayed through the native final block; it removes 44 to 49% of the covariance error but only 2 to 10% of the Gaussian next-variance response error and does not improve the final predictions, failing both gates, whereas substituting the full reference covariance improves raw error by 4.8 to 9.9%.

Recorded measurement:

    projected MSE / parent 1.0039280884164066 (35 A), 1.0017091930376703 (35 B), 1.0116283003206332 (100 A) vs <=0.80; full-covariance control 0.9507067444350218 / 0.9520061007263826 / 0.9009814335286453; rank-35 cross residual 1.0062472828123523 vs <=0.75 (control 0.9284628947343043); covariance residual energies 0.551057/0.557276/0.506699; variance-response residuals 0.898546/0.898493/0.979425

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_covariance_supervision_v1/screen.py](code/terminal_covariance_supervision_v1/screen.py)
- [code/terminal_covariance_supervision_v1/unit.py](code/terminal_covariance_supervision_v1/unit.py)
- [code/terminal_covariance_supervision_v1/finalize.py](code/terminal_covariance_supervision_v1/finalize.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/terminal_covariance_supervision_v1/common.py](code/terminal_covariance_supervision_v1/common.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L21](RESULT.md#L21)
- [RESULT.md#L22](RESULT.md#L22)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-post-layer-14-covariance-projection-onto-65-hidden-features) · [Copy provenance](../../../PROVENANCE.json)

