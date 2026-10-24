# Distinct third cumulant times covariance edge: fixed round closed

2026-09-06T00:51Z · experiment · rejected · measured

The three primary augmented diagrams pass at widths8/16: coefficient relative errors 2.13e-16 / 2.06e-16; full and low-rank query errors below 7.1e-19. This is a genuinely omitted term, but omission does not establish useful alignment with the current error. Stop: no rank100, other layer, q8/q16 sweep, damping, fitting, rollout, candidate or submission.

Recorded measurement:

    The three primary augmented diagrams pass at widths8/16: coefficient relative errors 2.13e-16 / 2.06e-16; full and low-rank query errors below 7.1e-19. This is a genuinely omitted term, but omission does not establish useful alignment with the current error. The first fixed wide cell was rank35 L12 -> next pre-K21 L13, repeated indices0,32,...,992, all1024 partner indices. Current source-v2 coherent8M oracle previous means/covariances; frozen source inputs before reading targets; original covariance target byte-exact. Pre/post CP20352/21568. Existing K21 diagonal discrepancy on queried coordinates was .03784%. Natural coefficient1 gives unweighted plane-error MSE ratio1.00254042; leading covariance-consumer-weighted ratio1.01336446. Weighted correction cosine with the missing plane is -.04847, versus the predeclared >=.4 gate. Correction norm is7.69% of weighted baseline error. Full source fails the information gate, before any approximation is relevant. Fixed top4 covariance-edge approximation also fails: unweighted source relative error83.30%, weighted84.76%, versus15% gates. Top4 covariance trace fraction14.576%. This differs from projecting the whole third tensor, but the tested edge truncation is still inaccurate for this source. Stop: no rank100, other layer, q8/q16 sweep, damping, fitting, rollout, candidate or submission. This closes the specified source test, not all third-order improvements. Whole run11.036s, peak4.702GB. All jobs complete.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/certify_third_covariance_edge_v1.py](code/astra_phase2_reexamination_v1/certify_third_covariance_edge_v1.py)
- [code/astra_phase2_reexamination_v1/check_third_covariance_edge_wide_v1.py](code/astra_phase2_reexamination_v1/check_third_covariance_edge_wide_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/third_covariance_edge_v1.py](code/astra_phase2_reexamination_v1/third_covariance_edge_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [THIRD_COVARIANCE_EDGE_V1_CLOSED.md#L1](THIRD_COVARIANCE_EDGE_V1_CLOSED.md#L1)
- [THIRD_COVARIANCE_EDGE_V1_CLOSED.md#L6](THIRD_COVARIANCE_EDGE_V1_CLOSED.md#L6)

<details>
<summary>All result files and records (4)</summary>

- [THIRD_COVARIANCE_EDGE_V1_CLOSED.md](THIRD_COVARIANCE_EDGE_V1_CLOSED.md) · record
- [third_covariance_edge_certificate_v1.json](third_covariance_edge_certificate_v1.json) · result
- [third_covariance_edge_r35_l12_v1.json](third_covariance_edge_r35_l12_v1.json) · result
- [THIRD_COVARIANCE_EDGE_DESIGN_V1.md](THIRD_COVARIANCE_EDGE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-distinct-third-cumulant-times-covariance-edge-fixed-round-closed) · [Copy provenance](../../../PROVENANCE.json)

