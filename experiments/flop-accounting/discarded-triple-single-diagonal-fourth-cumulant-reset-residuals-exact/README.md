# Discarded triple-single and diagonal fourth-cumulant reset residuals: exact transport derived

2026-08-26 · experiment · rejected · measured

_reset_stars computes three residuals and the deployed supply threads only k22r; the k31 off-diagonal residual and the k4 diagonal residual are discarded every layer. Exact push chains for both derived and brute-verified at 1e-16 (run_state_audit_k31.py docstring holds the multiplicities; all matrix-matrix multiplication (GEMM)-able, no n^4 F-form block; ~9e9/layer, fits the 288e9 margin). Deep-audit verdict rank 35: threading BOTH lifts the masked-median k4 ratio at every layer, L15 0.842 vs baseline 0.823 - closes ~11 percent of the depth deficit.

Recorded measurement:

    _reset_stars computes three residuals and the deployed supply threads only k22r; the k31 off-diagonal residual and the k4 diagonal residual are discarded every layer. Exact push chains for both derived and brute-verified at 1e-16 (run_state_audit_k31.py docstring holds the multiplicities; all GEMM-able, no n^4 F-form block; ~9e9/layer, fits the 288e9 margin). Deep-audit verdict rank 35: threading BOTH lifts the masked-median k4 ratio at every layer, L15 0.842 vs baseline 0.823 - closes ~11 percent of the depth deficit. The remaining deficit lives elsewhere (K22 F-form ij|ij route is cost-blocked; pair-covariance errors unmeasured until the prober's MC pairs land). RAW VERDICT: CLOSED - +2.7 percent worse, 0/3 wins (35: +1.6, 100: +4.2, 160: +2.3), the FIFTH co-adaptation confirmation: the deep k4 ratio improved at every layer and the final mean got worse. Better state metrics do not convert on this host; the card is a cancellation optimum. Only massive mass restorations (the original K22 supply, 7 percent/layer) or full truth (teacher-forced 8.2e-9) clear it; +0.02-ratio-scale unilateral state fixes break cancellation and lose.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_k211_supply_v1/run_state_audit_k31.py](code/public_k211_supply_v1/run_state_audit_k31.py)
- [code/public_k211_supply_v1/run_k31_raw_probe.py](code/public_k211_supply_v1/run_k31_raw_probe.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [k31_audit.log#L46](k31_audit.log#L46)
- [k31_raw_probe.log#L1](k31_raw_probe.log#L1)

<details>
<summary>All result files and records (2)</summary>

- [k31_audit.log](k31_audit.log) · log
- [k31_raw_probe.log](k31_raw_probe.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-discarded-triple-single-diagonal-fourth-cumulant-reset-residuals-exact) · [Copy provenance](../../../PROVENANCE.json)

