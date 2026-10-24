# Estimated-quality correction coefficients transferred with held-out gains 20.0% and 15.2%

2026-08-30 · experiment · informational · measured

DEPLOYABLE-CLASS chain graded (v4 tensor supply + FULL injection tangent + consumption), cross-net: rank 35 -9.4%, rank 100 -15.0% at the FIXED oracle-tuned betas (old injections gave -5.3/-7.0). ESTIMATE-QUALITY BETAS fitted by joint least squares on one net and applied FROZEN to the other improve both held-out directions: fit35->apply100 20.0% (2.196e-8), fit100->apply35 15.2% (2.270e-8). Fitted v-beta shrinks 0.82 -> 0.42-0.55 (matches the end-frame quality math: beta_opt scales with vector-residual corr); k3 stable ~0.35; k4 channel cross-net UNSTABLE (0.15 vs -0.06, near-zero value - k4u supply at 30% deep rel does not convert). v4b k211 completion = NULL at w1024 (L12 rel 0.0804 vs 0.0802).

Recorded measurement:

    DEPLOYABLE-CLASS chain graded (v4 tensor supply + FULL injection tangent + consumption), cross-net: rank 35 -9.4%, rank 100 -15.0% at the FIXED oracle-tuned betas (old injections gave -5.3/-7.0). ESTIMATE-QUALITY BETAS fitted by joint least squares on one net and applied FROZEN to the other improve both held-out directions: fit35->apply100 20.0% (2.196e-8), fit100->apply35 15.2% (2.270e-8). Fitted v-beta shrinks 0.82 -> 0.42-0.55 (matches the end-frame quality math: beta_opt scales with vector-residual corr); k3 stable ~0.35; k4 channel cross-net UNSTABLE (0.15 vs -0.06, near-zero value - k4u supply at 30% deep rel does not convert). v4b k211 completion = NULL at w1024 (L12 rel 0.0804 vs 0.0802). FRAME CAVEATS for promotion: (1) betas need the full 10-net LOO (infra: mps_cov2/host_cov for 8 more nets); (2) v4 runs on TRUE mean/cov inputs (validation frame) - the deploy-raw gate (closure-fed states) is the required conversion gate before any submission claim. Deep-plane error structure: rank-1 dominated (62% L12, 72% L14 of error energy), mode ~ (-k4 direction) x (m/active-unit direction) - fourth-order-supply-flavored; ORACLE_FOURTH wedge queued.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_beta_transfer.py](code/run_beta_transfer.py)

## Evidence

- [run_beta_transfer.py#L1](code/run_beta_transfer.py#L1)
- [BETA_TRANSFER_tensor.json#L1](BETA_TRANSFER_tensor.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [BETA_TRANSFER_tensor.json](BETA_TRANSFER_tensor.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-estimated-quality-correction-coefficients-transferred-held-out-gains-20) · [Copy provenance](../../../PROVENANCE.json)

