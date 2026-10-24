# Exact fifth-cumulant diagonal and two-index transport: measured and rejected

2026-08-26 · experiment · rejected · measured

Closed only for the implemented direct fifth-order spelling. The diag, K41, and K32 push chains verify to <=3.776008120484528e-16, yet adding them autonomously is violently unstable across both n=16, depth-5, 20M-antithetic-Monte Carlo (MC) seeds: C -> D mean squared error (MSE) is 0.0019371902666675653 -> 0.11720815944570878 and 0.0889081513905304 -> 644.8317168674707; the kappa6-free E spelling is worse again at 92.51253102041694 and 73436.32971540386. One-step teacher forcing can improve, so this is an autonomous partial-state instability result, not a proof that all higher-order carry is useless.

Recorded measurement:

    Closed only for the implemented direct fifth-order spelling. The diag, K41, and K32 push chains verify to <=3.776008120484528e-16, yet adding them autonomously is violently unstable across both n=16, depth-5, 20M-antithetic-MC seeds: C -> D MSE is 0.0019371902666675653 -> 0.11720815944570878 and 0.0889081513905304 -> 644.8317168674707; the kappa6-free E spelling is worse again at 92.51253102041694 and 73436.32971540386. One-step teacher forcing can improve, so this is an autonomous partial-state instability result, not a proof that all higher-order carry is useless. The advertised T5 width-power and T6 n=1024 cost sections were not implemented in the receipt and are not claimed here.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exact_ledger_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/ledger_rig.py](code/ledger_rig.py)
- [code/ledger_verify.py](code/ledger_verify.py)

## Evidence

- [LEDGER_VERIFY_RESULTS.json#L1057](LEDGER_VERIFY_RESULTS.json#L1057)

<details>
<summary>All result files and records (1)</summary>

- [LEDGER_VERIFY_RESULTS.json](LEDGER_VERIFY_RESULTS.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-exact-fifth-cumulant-diagonal-two-index-transport-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

