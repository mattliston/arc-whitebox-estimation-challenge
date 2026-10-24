# Late covariance error has low-rank structure poorly explained by the estimator's own eigenvectors

2026-10-20T08:31:31Z · finding · informational · measured

The covariance error was 0.4-0.6% of the true covariance Frobenius norm and its leading eigenvector carried about 20% of squared error. The estimator own top-32 covariance eigenspace explained only 13-22% of diagonal error.

Recorded measurement:

      (1) The pre-covariance error dC = Cpre_card - Cpre_true is 0.4-0.6% of |C| (Frobenius), the per-unit variance error dv 0.06-0.12% of v, mean relative bias ~0; its top eigenvector carries ~20% of |dC|^2 (top 8: 35-49%) but the card's OWN top-32 eigen-subspace explains only 13-22% of diag(dC): the error is not along directions the card knows.
    [fable-lead] 2026-10-20T08:31:31Z FOUR OFFLINE CLOSURES on the late-layer error (nets 35/100, card quantities from the k21dump files at L8-14 vs the 32M mcall truth; lead/runs/fable_oct19_kc_v1/{dcov_structure,dv_regress,k22cov_offline,mehler_offline,edge2_offline}_v1.py):

## Programs

- [dcov_structure_v1.py](dcov_structure_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6148](../../TEAM_LOG_2026-10-01_to_10-22.md#L6148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6147](../../TEAM_LOG_2026-10-01_to_10-22.md#L6147)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-late-covariance-error-has-low-rank-structure-poorly-explained) · [Copy provenance](../../../PROVENANCE.json)

