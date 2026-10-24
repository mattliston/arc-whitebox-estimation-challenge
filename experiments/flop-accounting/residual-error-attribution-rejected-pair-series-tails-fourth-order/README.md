# Residual-error attribution rejected pair-series tails, fourth-order cap, fifth cumulants and roundoff

2026-08-25 · experiment · rejected · measured

The deployed card's irreducible residual (floor about 2.96e-8): born at layers 2-6 (born fractions 54-67 percent) then propagated; diffuse at unit level; state-dependent with no positional coherence. Owner candidates measured OUT: Mehler tail (t25->7 saves 109.1e9 at 5e-5 effect), fourth-cumulant canonical-polyadic carrier (CP4) dropped planes (cap doubling moves 0.031-0.036x error for +144.1e9), coincident bonds (nonexistent at width 1024 - max off-diagonal rho 0.58 at layer 15), marginal kappa5 (carrier built end to end, Monte Carlo (MC)-verified closed form, 0.001x error at 80.6e6 FLOPs), the cubic push (exact by term-by-term algebra), f32 roundoff (full-f64 probe, plus-minus 0.002 percent mean squared error (MSE)). The floor is the compounded born error (teacher-forcing capacity 8.2e-9 vs floor 3.1e-8).

Recorded measurement:

    The deployed card's irreducible residual (floor about 2.96e-8): born at layers 2-6 (born fractions 54-67 percent) then propagated; diffuse at unit level; state-dependent with no positional coherence. Owner candidates measured OUT: Mehler tail (t25->7 saves 109.1e9 at 5e-5 effect), CP4 dropped planes (cap doubling moves 0.031-0.036x error for +144.1e9), coincident bonds (nonexistent at width 1024 - max off-diagonal rho 0.58 at layer 15), marginal kappa5 (carrier built end to end, MC-verified closed form, 0.001x error at 80.6e6 FLOPs), the cubic push (exact by term-by-term algebra), f32 roundoff (full-f64 probe, plus-minus 0.002 percent MSE). The floor is the compounded born error (teacher-forcing capacity 8.2e-9 vs floor 3.1e-8).

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_coincident_bond_census_v1/capture_pairs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_coincident_bond_census_v1/capture_pairs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_floor_attribution_v1/run_floor_attribution.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_floor_attribution_v1/run_floor_attribution.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/build_estimator_k5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/build_estimator_k5.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/wire_k5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k5_probe_v1/wire_k5.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [FLOOR_ATTRIBUTION.json#L41](FLOOR_ATTRIBUTION.json#L41)
- [CENSUS_COUNTS.json#L1](CENSUS_COUNTS.json#L1)
- [BUILD_RECEIPT_STAGE2.json#L1](BUILD_RECEIPT_STAGE2.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [FLOOR_ATTRIBUTION.json](FLOOR_ATTRIBUTION.json) · result
- [CENSUS_COUNTS.json](CENSUS_COUNTS.json) · result
- [BUILD_RECEIPT_STAGE2.json](BUILD_RECEIPT_STAGE2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-residual-error-attribution-rejected-pair-series-tails-fourth-order) · [Copy provenance](../../../PROVENANCE.json)

