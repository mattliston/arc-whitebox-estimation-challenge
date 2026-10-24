# Fixed score-aligned covariance tangent: oracle gate failed

2026-09-06T12:43Z · experiment · rejected · measured

Analytic risk residual .3173365088; actual W14 next-mean response residual .3486850827; ordinary covariance residual .4669549997. Thus cross blocks matter substantially (core-only .861707/.823676 risk/value), but both fixed .25 gates fail. This is meaningful geometric information, not a learnable correction or a global rank/cost impossibility claim.

Recorded measurement:

    Analytic risk residual .3173365088; actual W14 next-mean response residual .3486850827; ordinary covariance residual .4669549997. Thus cross blocks matter substantially (core-only .861707/.823676 risk/value), but both fixed .25 gates fail. Normal stationarity2.55e-15; small explicit dense certificate <=1.45e-15. Host time .036s, peak RSS352,763,904 bytes. This is meaningful geometric information, not a learnable correction or a global rank/cost impossibility claim. Do not call the32% risk remainder an adjusted or raw estimator score. No rank/frame/strength sweep, predictor training, carrier write or submission follows this failed fixed tangent.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_covariance_score_tangent_v1.py](code/check_covariance_score_tangent_v1.py)
- [code/covariance_score_subspace_v1.py](code/covariance_score_subspace_v1.py)
- [code/covariance_score_tangent_v1.py](code/covariance_score_tangent_v1.py)
- [code/gaussian_next_mean_risk_metric_v1.py](code/gaussian_next_mean_risk_metric_v1.py)

## Evidence

- [COVARIANCE_SCORE_TANGENT_V1_CLOSED.md#L1](COVARIANCE_SCORE_TANGENT_V1_CLOSED.md#L1)
- [COVARIANCE_SCORE_TANGENT_V1_CLOSED.md#L12](COVARIANCE_SCORE_TANGENT_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [COVARIANCE_SCORE_TANGENT_V1_CLOSED.md](COVARIANCE_SCORE_TANGENT_V1_CLOSED.md) · record
- [covariance_score_tangent_wide_v1.json](covariance_score_tangent_wide_v1.json) · result
- [COVARIANCE_SCORE_TANGENT_DESIGN_V1.md](COVARIANCE_SCORE_TANGENT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-fixed-score-aligned-covariance-tangent-oracle-gate-failed) · [Copy provenance](../../../PROVENANCE.json)

