# Third-order pair kernel rho-cubed extension: inconsistent MSE effects on networks 35 and 100

2026-10-20T05:17:07Z · experiment · rejected · measured

Adding the next rho^3/6 kernel term gave -0.13% / -0.61% on network 35 and +0.36% / +1.21% on network 100 for the tested layer ranges. Neither sign consistently improved both networks.

Recorded measurement:

    - [fable-lead] 2026-10-20T05:17:07Z K21 KERNEL RHO-ORDER DEAD (native oracle, card's own plane through the next rho^3/6 term at L8-14 / L2-14): net 35 -0.13% / -0.61%, net 100 +0.36% / +1.21%, and the opposite sign is also worse on 100: no consistent kernel-truncation lever. Summary of the night's native tests on the shipped card: every cheap structural or scalar change is at its optimum or net-dependent noise; remaining candidates = gnnglate_m (-0.07% on 6 nets) and the prescreen2 knobs (edge, bfix, rowcut, gk4, k31r late, rc, iso).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rel_nosync_sci_v1.py](code/candidate_rel_nosync_sci_v1.py)
- [code/native_k31_oracle_v1.py](code/native_k31_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6137](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6137)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-third-order-pair-kernel-rho-cubed-extension-inconsistent-mse) · [Copy provenance](../../../PROVENANCE.json)

