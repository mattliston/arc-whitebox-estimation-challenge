# Quadratic skew-plane covariance proxy: closed on information

2026-09-05T23:28Z · experiment · rejected · measured

The n4 exact Gaussian polynomial/conditional-projection identities pass below7e-17. The analytic covariance response agrees with a finite difference of the actual parent map at relative9.692e-11. Heldout400: covariance ratio .995716259902, next-mean response .993161296841.

Recorded measurement:

    The n4 exact Gaussian polynomial/conditional-projection identities pass below7e-17. The analytic covariance response agrees with a finite difference of the actual parent map at relative9.692e-11. No exact global fourth tensor is claimed for the pairwise proxy. Cost v1 projected12.88% extra. Source audit found that FlopScope.12 ignores the symmetric flag for solve_cost: it charges generic LU. The v1 number already charged that LU cost, but a literal positive-definiteness guard also needs a Cholesky. Version2 includes BOTH and projects13.448% extra on35/100, still within the original15% gate. Neither is a native bill. Heldout400: covariance ratio .995716259902, next-mean response .993161296841. Heldout500: covariance ratio .996535640480, next-mean response1.000413621442. New/control next-mean ratios .999834525933 /1.000066289726. All original 25% and incremental10% gates fail. The normalized fit is well conditioned (condition2.745), not an optimizer issue. No larger fit, proxy expansion, rollout, panel, candidate, native port or submission follows this result. Job64.421s, peakRSS6,273,368,064bytes. All workers complete. Receipts: quadratic_skew_covariance_certificate_v1.json, cost_v1/v2 JSONs, quadratic_skew_covariance_verdict_v1.json, and quadratic_skew_covariance_progress_v1/. The model NPZ is diagnostic only. Official usable #329926 remains adjusted4.982231460183304e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/check_quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/fit_quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/fit_quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v1.py](code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v1.py)
- [code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v2.py](code/astra_phase2_reexamination_v1/generate_covariance_innovation_probe_v2.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md#L1](QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md#L1)
- [QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md#L24](QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md#L24)

<details>
<summary>All result files and records (4)</summary>

- [QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md](QUADRATIC_SKEW_COVARIANCE_V1_CLOSED.md) · record
- [QUADRATIC_SKEW_COVARIANCE_V1.md](QUADRATIC_SKEW_COVARIANCE_V1.md) · record
- [quadratic_skew_covariance_certificate_v1.json](quadratic_skew_covariance_certificate_v1.json) · result
- [quadratic_skew_covariance_verdict_v1.json](quadratic_skew_covariance_verdict_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-quadratic-skew-plane-covariance-proxy-closed-information) · [Copy provenance](../../../PROVENANCE.json)

