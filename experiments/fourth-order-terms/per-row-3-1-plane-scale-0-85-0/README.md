# Per-row (3,1)-plane scale 0.85 + 0.5 μ κ₃/v²: native adjusted ratio 0.98606

2026-09-23T02:33Z · experiment · adopted · measured

A per-row scale on the isotropic (3,1) plane improves the covariance output with negligible added work. Fixed coefficients were retained because they won more consistently across held-out networks than a per-layer schedule.

Recorded measurement:

    2026-09-22 19:33 porter: physics patch build_physics.py (per-row K31 scale + extra-plane plumbing + newborn_common handle; string edits valid on bank AND build_e2 output). r35: k31row (1,0) bit-exact MSE (+602k flops); provisional (0.85,0.5)@L1-14 -1.47% (torch -1.27%). Paired VALID40 vs cached bank rows running.
    2026-09-22 19:58 physicist: VALID40 (torch, no refit, held out; coefficients from r35/r100 truth only): K31 per-row rowf = 0.85 + 0.5 mu k3/v^2 at L2-14 -> mean-MSE ratio 0.9857 (CI .9828-.9884), 40/40 wins; K31 x0.9 -> 0.9880, 40/40. Native port = one line in bank _nonlinear_pair_batch (T31 = (t4*rowf)[:,None]*rho), ~84 requests/net. fable_physicist_v1/RESULT.md
    2026-09-22 19:38 physicist: (timestamp correction: the line stamped 19:58 above was written at 19:37 machine time.)
    2026-09-22 19:49 physicist: final. Per-layer (a_l,b_l) schedule VALID40 0.9838 but 35/40 wins (fixed 0.85/0.5 rowf is the robust pick, 40/40). K22-plane scaling harmful (+7..+15%). Per-unit v error unchanged by all K31 variants (gain rides the off-diagonal covariance). RESULT.md complete.
    2026-09-22 19:55 porter: per-row K31 scale PROVISIONAL (c0 .85, c1 .5, L1-14) paired VALID40 vs cached bank: adj_ratio 0.98606, raw 0.98605, wins 38/40, util 0.188863 vs 0.188858; per-net raw -3.84%..+0.22% (median -1.03%). cand fable_porter_v1/cand_phys_k31r_085_05_bank.py
    2026-09-22 20:00 porter: build_physics.py verified on the e2 build (sha 24f3c465 = cand_e2_v5): disabled = e2 exactly; k31row(1,0) bit-exact; k31row(.85,.5) on e2 == on bank (1.623723963140016e-08 r35); zero plane bit-exact; plane control == direct doubling. Waiting for physicist constants.
    2026-09-22 20:15 physicist: K31 row factor TRAIN sweep (ranks 40/41/43/44, torch, no refit): best c0=0.8 c1=0.5 -> -2.03% mean, 4/4 (0.85/0.5 -1.79%, 0.9 -1.31%, 0.75 -1.96%, 0.7 -1.66%, 0.6 -0.37%); c1 flat 0.5-1.0.
    2026-09-22 21:08 porter: STACKED BASE cand_stack_e2v5_k31r.py (bank -> e2 v5 -> K31 row .85/.5 L1-14) paired VALID40 vs cached bank: adj 0.98034 (-1.97%), raw 0.98605, adj wins 40/40, util 0.187767 vs 0.188858; MSE bit-equal to bank-based physics on r35, r100, all 40 VALID. PARKED. Physics patch also verified on e2 v7 (hold form).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_physics.py](code/build_physics.py)
- [code/cand_phys_k31r_085_05_bank.py](code/cand_phys_k31r_085_05_bank.py)
- [code/cand_stack_e2v5_k31r.py](code/cand_stack_e2v5_k31r.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L20](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L20)
- [TEAM_LOG_2026-09-22_to_09-26.md#L23](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L23)
- [TEAM_LOG_2026-09-22_to_09-26.md#L24](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L24)
- [TEAM_LOG_2026-09-22_to_09-26.md#L26](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L26)
- [TEAM_LOG_2026-09-22_to_09-26.md#L27](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L27)
- [TEAM_LOG_2026-09-22_to_09-26.md#L28](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L28)
- [TEAM_LOG_2026-09-22_to_09-26.md#L31](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L31)
- [TEAM_LOG_2026-09-22_to_09-26.md#L39](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L39)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer](../../../submissions/331947/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-per-row-3-1-plane-scale-0-85-0) · [Copy provenance](../../../PROVENANCE.json)

