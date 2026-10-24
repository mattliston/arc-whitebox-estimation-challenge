# Additional Gram-Charlier pair-covariance terms explain at most 1.8% of pre-variance error

2026-10-20T09:54:06Z · experiment · rejected · measured

Marginal third/fourth cumulant and mixed second-order terms had weak correlations with the needed covariance correction. No tested missing closure term was large and aligned enough to explain the remaining error.

Recorded measurement:

    [fable-lead] 2026-10-20T09:54:06Z GRAM-CHARLIER COVARIANCE TERMS DEAD (lead/runs/fable_oct19_kc_v1/cov_terms_offline_v1.py; nets 35/100, L8-13 -> next-layer pre-variance error dv, mixed-order Mehler kernels to n=4): first-order marginal terms T30 (k3_i/6 sum C^n/n! h_{3+n} h_n, rms 6-8e-5 = 1/3 of dv) and T40 (k4_i/24 ..., rms 3e-5) have |corr| <= 0.11 with the needed correction and betas -0.7..+0.2 (present already or irrelevant; not missing at size 1); second-order terms T33 (k3 k3/36), T3_21 (k3 K21/12), T21sq (K21^2/8), T2112 (K21 K21^T/4) are 1e-8..2e-5 rms with |corr| <= 0.14 and shares <= 1.8%. No closure term of the pair map is missing at a size that matters; the covariance error is plane CONTENT (the K21/K31 objects themselves) and inheritance.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cov_terms_offline_v1.py](code/cov_terms_offline_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6169](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6169)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-additional-gram-charlier-pair-covariance-terms-explain-most-1) · [Copy provenance](../../../PROVENANCE.json)

