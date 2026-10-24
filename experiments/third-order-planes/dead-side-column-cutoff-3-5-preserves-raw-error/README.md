# Dead-side column cutoff at -3.5 preserves raw error but native kernels erase most predicted savings

2026-09-24T18:16Z · experiment · adopted · measured

Removing sufficiently inactive rows was accurate, while larger cuts and response-only or alive-only reconstruction failed. The native row count changed multiplication depth, turning much of the expected work saving into a runtime saving.

Recorded measurement:

    - 2026-09-24 11:16 analyst: E-48 ROW-DROP no-refit VALID16 (torch s6bcard + B2 ep1, 3 thr; base 1.746113e-8 = lead's 1.74614e-8): (a) dead-side cut a<-3.5 at L1-14: raw +0.09% (8/16), projected adj -2.21% (16/16) -> PASS; (b) old rebuild on response rows: raw +12.0% (0/16), adj +8.0% FAIL; (c) alive rows only: raw +21.9%, adj +16.5% FAIL. Near-dead/saturated k3 matters, a<-3.5 does not. (a) needs the native build + census to confirm the -2.3% bill. RESULT.md sec 28.
    - 2026-09-24 11:25 lead: ANALYST ROW-DROP (RESULT §28; torch s6bcard + B2 ep1, no refit, VALID16): (a) DEAD-SIDE CUT (all column work off on rows a < -3.5 at L1-14; 0 rows at L1-6, 32-128 at L7-14) raw +0.09% (8/16), projected adjusted -2.21% at E-48's -2.3% bill, 16/16: PASS. (b) old rebuild on response rows: raw +12.0%, FAIL. (c) alive rows only: raw +21.9%, FAIL. Near-dead units (-3.5 < a < -2.3) and saturated units matter through their skew terms. Porter builds native --deadcut35 (parity vs torch rows, native bill census, VALID40 bar alone: adjusted <= -1.5%, CI excl. 0, >=30/40). Analyst sweeps thresholds -3.0/-2.75/-2.5. S12 candidates: S11 + deadcut35 + exact 0.37% package (+ R320 if its refit passes).
    2026-09-24 11:30 porter: deadcut35 native (stack_S11_dc35_mem.py 63db1ee2): semantics exact (null = S11 bit-identical; dropped rows = torch (a) per layer; raw tracks torch), einsum gate PASS, validate OK; but REAL bill only -0.53% (4 nets; E-48 est. -2.3%): m<1024 pushes bill ~4-5.5% more per row (only m==1024 takes the six-level padded path). VALID40 held; options to the lead.
    - 2026-09-24 11:45 lead: native deadcut35 (porter build_deadcut.py, stack_S11_dc35_mem.py 63db1ee2) is semantically exact: dropped rows match torch (a) exactly and native raw tracks torch. But the REAL bill is only -0.525% (4 nets), not E-48's -2.3%: only m == 1024 takes the six-level padded push path, and a smaller m bills ~4-5.5% more per row. Decision: the engineer extends the padded path to m a multiple of 64 (bit-exact at 1024); deadcut then rounds kept rows up to 64. Deadcut's VALID40 is HELD. It stacks into S12 with R320 and the exact package. The analyst's threshold sweep may enlarge the cut.
    - 2026-09-24 11:30 analyst: DEAD-SIDE THRESHOLD SWEEP (no refit, VALID16, B2 ep1): a<-3.5 raw +0.09% (8/16), 620 row-layers cut L1-14; a<-3.0 +0.97% (5/16), 966; a<-2.75 +2.79% (1/16), 1180; a<-2.5 +8.0% (0/16), 1436. Knee between -3.5 and -3.0. Per-layer row counts (rounded to 32 as the card's select, and exact) for the engineer's pricing in RESULT.md sec 28 addendum; rough projection -3.0 ~ -2.6% adj vs -3.5 -2.2%.
    2026-09-24 11:38 porter: deadcut35 paired user-CPU census vs S11 (6 nets): flops -0.45% but USER CPU -10.35%, backend -10.8%, no paging; saving = Winograd add/subtract transforms (m<1024 pushes take shallower paths). Flagged to lead/engineer: the m%64 push6 extension may give the CPU back. Files: fable_engineer2_v1/platfit/optime_porter_dc35.json, optime_porter_S11_ref_dc35.json.
    - 2026-09-24 11:55 lead: porter's paired user-CPU census (6 nets, no paging): deadcut35 on CURRENT kernels = bill -0.45%, user CPU -10.35% (backend -10.8%), a WALL ENABLER (~-7..-9 s platform kernel). Cause: the m=1024 six-level padded path is transform-heavy. S11 spends 161 s of backend CPU on 207 GF of add/subtract vs 71 s on 1905 GF of matmul, so the Winograd transforms are ~20x slower per billed flop than matmul and use ~2/3 of our CPU. Decisions: the engineer measures the m%64 extension's bill AND CPU before we choose, and looks for cheaper-transform variants (same bill, less CPU), which is the wall lever for deep Strassen everywhere. engineer2 projects S11 + deadcut35 (+R320) room, to see whether six-level reads fit.
    2026-09-24 11:40 [engineer2] DEADCUT35 RE-PROJECTION (porter's paired user-CPU census, optime_split_v3, concurrent, 1 thread, nets 35c/100/37/39/42/48; stack_S11_dc35_mem.py 63db1ee2 vs S11): user CPU -10.29% mean (per net -8.0..-11.8%, se ~0.7%), system CPU flat (4-5 s both arms: not paging), op calls -218/net, flops only -0.448% (native: rows below 1024 lose the padded-six push path and take shallower depths, trading most of the analyst's ~-2.3% bill for time); backend saving is transforms (add/subtract -25 s over 6 nets) + matmul -5.7 s. Platform conversion: the only transform-type calibration (push5p6: local user +5.3% vs platform +3.0/+5.0% on two runs) gives platform/local 0.57-0.94 (0.75 central). On S11 (expected worst 97.6 s, kernel 84.91 s): S11 + dc35 = 92.6 / 91.0 / 88.8 s (x0.57 / x0.75 / x1.0), room 7.4 / 9.0 / 11.2 s: the pessimistic room clears the lead's 6 s trigger, so the six-level-read quarter (+3.7..+5.7 s, -0.85% bill, paper) is back on the table: S11 + dc35 + reads-quarter = 98.3 / 95.7 / 92.5 s. Provisional (ADDITIVE, overlapping cuts save less): + R320 (-1.1/-3.3/-5.35 s kernel, -0.27 s round trips) -> 91.2 / 87.4 / 83.2 s; + reads quarter -> 96.9 / 92.1 / 86.9 s; to be replaced by the porter's measured R320+dc35 pair. Bill: dc35 -0.45% (adjusted about -0.36% with the analyst's no-refit raw +0.09%), R320 -3.89%, reads quarter -0.85%; the exact flag (-0.37%) has no CPU census yet.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stack/stack_S11_dc35_mem.py](code/stack/stack_S11_dc35_mem.py)
- [code/build_deadcut.py](code/build_deadcut.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L343](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L343)
- [TEAM_LOG_2026-09-22_to_09-26.md#L349](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L349)
- [TEAM_LOG_2026-09-22_to_09-26.md#L351](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L351)
- [TEAM_LOG_2026-09-22_to_09-26.md#L352](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L352)
- [TEAM_LOG_2026-09-22_to_09-26.md#L353](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L353)
- [TEAM_LOG_2026-09-22_to_09-26.md#L354](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L354)
- [TEAM_LOG_2026-09-22_to_09-26.md#L355](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L355)
- [TEAM_LOG_2026-09-22_to_09-26.md#L356](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L356)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [optime_porter_dc35.json](optime_porter_dc35.json) · result
- [optime_porter_S11_ref_dc35.json](optime_porter_S11_ref_dc35.json) · result

</details>

## Related work

- shipped in: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../../../submissions/332195/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-dead-side-column-cutoff-3-5-preserves-raw-error) · [Copy provenance](../../../PROVENANCE.json)

