# Third-order consumption of fifth-order corrections: measured and rejected

2026-08-26 · experiment · rejected · measured

The trivariate deriver re-derived the kappa5-free fifth-order class from first principles in the estimator's own Edgeworth-product-over-Mehler convention and proved the lambda3 shell's pair terms are EXACTLY the Lambda3*Lambda4 (a+b=7) plus Lambda3^3/6 (a+b=9) expansion - symbolically and to 1e-15 numerically. Term error is excluded; the shell's +5-11 percent harm on the supplied host must be order MISMATCH (pair planes at tilt order 3 over order-2 marginals) or asymptotic tail amplification. Measured tilt sizes on row 35 are tiny (max order-3 standardized weight ~9e-5 at L16, min sd 0.30), which rules the tail-amplification locus OUT for the univariate site.

Recorded measurement:

    The trivariate deriver re-derived the kappa5-free fifth-order class from first principles in the estimator's own Edgeworth-product-over-Mehler convention and proved the lambda3 shell's pair terms are EXACTLY the Lambda3*Lambda4 (a+b=7) plus Lambda3^3/6 (a+b=9) expansion - symbolically and to 1e-15 numerically. Term error is excluded; the shell's +5-11 percent harm on the supplied host must be order MISMATCH (pair planes at tilt order 3 over order-2 marginals) or asymptotic tail amplification. Measured tilt sizes on row 35 are tiny (max order-3 standardized weight ~9e-5 at L16, min sd 0.30), which rules the tail-amplification locus OUT for the univariate site. The direct attack is the univariate order-3 companion: He7 with k3*k4/144 and He9 with k3^3/1296, two extra columns in _univariate_edgeworth at ~zero flop cost; c-table already covers q<=9 and the two raw4 closed forms are c[4,7]=24 sigma^4 He2(-alpha) pdf, c[4,9]=24 sigma^4 He4(-alpha) pdf. MEASURED: the plain univariate retrofit is REFUTED - 4/4 rows worse, mean +186 percent (8.0-11.3e-8 vs 2.8-3.8e-8 card; first attempt was a scorer broadcast bug, fixed, function verified exact on captured inputs at rel 3e-4). With pair-only at +5-11 percent, this confirms the co-adaptation basin law at the univariate site: the carried k3/k4 are effective cumulants tuned to order-2 consumption, so exact higher-order terms double-count absorbed mass. CLOSED BY MEASUREMENT, all cells: pair-only +5-11 percent, univariate-only +186 percent, both order-matched +2.85 percent (3 rows, 0 wins), shared amplitude lambda=0.5 +1.36 percent - the loss is LINEAR in lambda through the origin (quadratic would predict +0.7), so no amplitude of the balanced order-3 retrofit pays. Fifth-order consumption on this host requires joint-retune or clean-slate per the basin law; the direct-retrofit family is dead. FRAME CAVEAT (found later): the univariate-only +186 percent cell was scored through Estimator().predict and is frame-polluted by the mismatched frozen corrector; the family closure STANDS on the valid host-frame cells (shell-only +5-11, both +2.85, lambda 0.5 +1.36, all via predict_with_native_tape against host baselines).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_k211_supply_v1/run_ue5_probe.py](code/public_k211_supply_v1/run_ue5_probe.py)
- [code/public_k211_supply_v1/run_shell_ue5_probe.py](code/public_k211_supply_v1/run_shell_ue5_probe.py)
- [code/public_k211_supply_v1/run_l5_lambda_scan.py](code/public_k211_supply_v1/run_l5_lambda_scan.py)
- [code/public_lambda3_shell_probe_v1/estimator_x.py](code/public_lambda3_shell_probe_v1/estimator_x.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/public_supply_card_v1/estimator.py](code/public_supply_card_v1/estimator.py)
- [code/public_trivariate_derivation_v1/l5_verify.py](code/public_trivariate_derivation_v1/l5_verify.py)
- [code/public_trivariate_derivation_v1/verify.py](code/public_trivariate_derivation_v1/verify.py)

## Evidence

- [L5_DERIVATION.md#L1](L5_DERIVATION.md#L1)
- [ue5_probe.log#L1](ue5_probe.log#L1)
- [shell_ue5_probe.log#L1](shell_ue5_probe.log#L1)
- [l5_lambda_scan.log#L1](l5_lambda_scan.log#L1)

<details>
<summary>All result files and records (4)</summary>

- [L5_DERIVATION.md](L5_DERIVATION.md) · record
- [ue5_probe.log](ue5_probe.log) · log
- [shell_ue5_probe.log](shell_ue5_probe.log) · log
- [l5_lambda_scan.log](l5_lambda_scan.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-third-order-consumption-fifth-order-corrections-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

