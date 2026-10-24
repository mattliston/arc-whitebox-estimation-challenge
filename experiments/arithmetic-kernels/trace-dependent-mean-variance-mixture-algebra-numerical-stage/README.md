# Trace-dependent mean/variance mixture: algebra and numerical stage

2026-09-06T18:26Z · experiment · informational · measured

- Exact small third/fourth tensor and third-cumulant repeated-index plane (K21)/fourth-cumulant triple-single plane (K31)/fourth-cumulant two-pair plane (K22) identities agree within1.92e-13. The v1 float64 raw-to-sixth-cumulant calculation fails its1e-11 relative gate through cancellation. The probability interpretation remains unqualified.

Recorded measurement:

    - Exact small third/fourth tensor and K21/K31/K22 identities agree within1.92e-13. The v1 float64 raw-to-sixth-cumulant calculation fails its1e-11 relative gate through cancellation. It stays failed. - The separate80-digit Gamma-polynomial/Wick reference passes the unchanged third-through-sixth identities, directional relative errors below4.63e-73. This validates algebra, not physical accuracy. - Rounded actual-bank tapes, fixed ranks35/100, layers8..14: all14 necessary active-space conditions pass with substantial margin, b^T C_active^-1 b/q=.4189..4979. This is not an exact PD certificate. - The exact original coherent-history rank35 replay passes its old input and host byte-equality checks. Captured b/B are from the actual source-v2 history protocol, not inferred from labels. - Eight versus32 nodes passes every fixed late cell: relative mean disagreement <=1.534e-13, relative covariance <=2.865e-12, and relative induced covariance-increment disagreement <=4.978e-8. Gates were 1e-8,1e-8,1e-3 respectively. No node/strength/range fitting. - Exact-call stored matrices do NOT provide a global Gaussian model: min C=-1.36e-7..-1.81e-7, min D=-3.33e-7..-4.39e-7, min V=-2.38e-6..-5.32e-6. C/B were explicitly symmetrized for the proposed model, maximum changes1.53e-7. Nothing was PSD-clipped. Every queried conditional bivariate Gaussian was numerically valid, but that is not a certificate of one positive joint distribution. Exact worker completed15.11seconds, peak5,207,359,488bytes, nice10 and three CPU threads. No process remains live. It read fixed coherent history inputs, not current-output errors for scoring. The implemented information consumer supplies only mean/raw-second increments. With direct dense elementwise evaluation, sixteen Gaussian Hermite6 pair-AA maps plus exact-K31 Edgeworth subtraction cost about 12e9 extra FLOPs per16-layer MLP, roughly2.5% of the490e9 bank bill. This is an analytical operation estimate, not a flopscope meter. A physically coherent full-state rollout also needs pair-BA and pair-BB raw moments, and marginal third/fourth raw moments, all centered after the changed mean. The same component derivatives can be reused, but these pair products raise a rough complete consumer estimate to30e9 extra FLOPs (~6% bank). Leaving the old K21/radial centering unchanged would not implement that full model. No such rollout is implemented.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/capture_gamma_pair_inputs_v2.py](code/astra_phase2_reexamination_v1/capture_gamma_pair_inputs_v2.py)
- [code/astra_phase2_reexamination_v1/certify_fresh_trace_meanmix_v1.py](code/astra_phase2_reexamination_v1/certify_fresh_trace_meanmix_v1.py)
- [code/astra_phase2_reexamination_v1/certify_fresh_trace_meanmix_v2.py](code/astra_phase2_reexamination_v1/certify_fresh_trace_meanmix_v2.py)
- [code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_exact_v1.py](code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_exact_v1.py)
- [code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_feasibility_v1.py](code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_feasibility_v1.py)
- [code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_value_v1.py](code/astra_phase2_reexamination_v1/check_fresh_trace_meanmix_value_v1.py)
- [code/astra_phase2_reexamination_v1/fresh_trace_meanmix_consumer_v1.py](code/astra_phase2_reexamination_v1/fresh_trace_meanmix_consumer_v1.py)
- [code/astra_phase2_reexamination_v1/fresh_trace_meanmix_v1.py](code/astra_phase2_reexamination_v1/fresh_trace_meanmix_v1.py)
- [code/astra_phase2_reexamination_v1/gamma_radial_pair_v1.py](code/astra_phase2_reexamination_v1/gamma_radial_pair_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md#L1](FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md#L1)
- [FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md#L31](FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md#L31)

<details>
<summary>All result files and records (6)</summary>

- [FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md](FRESH_TRACE_MEANMIX_STAGE1_VERDICT_V1.md) · record
- [FRESH_TRACE_MEANMIX_DESIGN_V1.md](FRESH_TRACE_MEANMIX_DESIGN_V1.md) · record
- [fresh_trace_meanmix_algebra_v1.json](fresh_trace_meanmix_algebra_v1.json) · result
- [fresh_trace_meanmix_algebra_v2.json](fresh_trace_meanmix_algebra_v2.json) · result
- [fresh_trace_meanmix_feasibility_v1.json](fresh_trace_meanmix_feasibility_v1.json) · result
- [fresh_trace_meanmix_exact_r35_v1.json](fresh_trace_meanmix_exact_r35_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-trace-dependent-mean-variance-mixture-algebra-numerical-stage) · [Copy provenance](../../../PROVENANCE.json)

