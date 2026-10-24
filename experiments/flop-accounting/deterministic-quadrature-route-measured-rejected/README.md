# Deterministic quadrature route: measured and rejected

2026-08-26 · experiment · rejected · measured

INFEASIBLE, impossibility MEASURED not assumed (rank 35, cross-checked 36/120/234): at budget an unbiased hybrid has 58,845 pairs, so 1e-9 needs 99.912 percent surrogate variance capture; measured ceilings - affine/Stein-chain 20.5, ANY degree<=2 oracle 43.2, perfect 512-dim active subspace 63.1, even a FREE degree-64-exact rule leaves 2.8 percent -> 2.85e-8 floor, above the fire bar. Rules race: best gain over Monte Carlo (MC) 1.62x (lattice QMC 1.16x RMSE, antithetic 0.84x variance; GH-on-top-2 53x WORSE), every exponent -0.95..-1.09; predictions matched measurements. TWO NEW FACTS: (1) real nets carry ~21 percent final-layer linear-in-z Hermite mass (three methods: 0.205/0.210/0.222) vs 0.7 percent mean-field - retention saturates after layer 1; the depth-D degree mass law = Galton-Watson total progeny of the arccos-kernel PGF (m1=0.7335); (2) 'plain MC 1.5e-7 at 65k' in the briefing was the shipped hybrid - true plain MC is 1.0-1.3e-6.

Recorded measurement:

    INFEASIBLE, impossibility MEASURED not assumed (rank 35, cross-checked 36/120/234): at budget an unbiased hybrid has 58,845 pairs, so 1e-9 needs 99.912 percent surrogate variance capture; measured ceilings - affine/Stein-chain 20.5, ANY degree<=2 oracle 43.2, perfect 512-dim active subspace 63.1, even a FREE degree-64-exact rule leaves 2.8 percent -> 2.85e-8 floor, above the fire bar. Rules race: best gain over MC 1.62x (lattice QMC 1.16x RMSE, antithetic 0.84x variance; GH-on-top-2 53x WORSE), every exponent -0.95..-1.09; predictions matched measurements. TWO NEW FACTS: (1) real nets carry ~21 percent final-layer linear-in-z Hermite mass (three methods: 0.205/0.210/0.222) vs 0.7 percent mean-field - retention saturates after layer 1; the depth-D degree mass law = Galton-Watson total progeny of the arccos-kernel PGF (m1=0.7335); (2) 'plain MC 1.5e-7 at 65k' in the briefing was the shipped hybrid - true plain MC is 1.0-1.3e-6. Leaders' faster-than-1/N spend cannot be quadrature-on-the-net; remaining doors are exact analytic carry or a pathwise-coupled surrogate outside the exactly-integrable algebra.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_quadrature_derivation_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/assemble_verdict.py](code/assemble_verdict.py)
- [code/measure_curve_v2.py](code/measure_curve_v2.py)
- [code/measure_realnet.py](code/measure_realnet.py)
- [code/quad_verify.py](code/quad_verify.py)
- [code/realnet_rules.py](code/realnet_rules.py)
- [code/refit_nnls.py](code/refit_nnls.py)
- [code/theory_branching.py](code/theory_branching.py)

## Evidence

- [QUADRATURE_VERDICT.json#L103](QUADRATURE_VERDICT.json#L103)
- [QUADRATURE_DERIVATION.md#L205](QUADRATURE_DERIVATION.md#L205)

<details>
<summary>All result files and records (2)</summary>

- [QUADRATURE_VERDICT.json](QUADRATURE_VERDICT.json) · result
- [QUADRATURE_DERIVATION.md](QUADRATURE_DERIVATION.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-deterministic-quadrature-route-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

