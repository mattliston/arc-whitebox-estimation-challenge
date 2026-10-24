# Tree routing of thin residual-plane products saves only 0.25% within a 2,000-request allowance

2026-09-25T04:35Z · experiment · rejected · measured

Routing all thin products through fast multiplication required too many additional remote calls. The request-limited subset consumed the residual allowance for too small a saving.

Recorded measurement:

      - Side lever: the low-rank star's thin products run as plain matmuls (2.8% of the bill); tree-kernel routing ~-0.7..-1.0%, rounding-level.
      GO: build routing first (parity, in-situ bill, op count, residual gate <= +0.010 s cold; price the requests, since residual is the tightest budget), then s1 r32 on top. Native VALID40 vs S21 rows: adjusted <= -1.2%, CI excl 0, raw <= +0.8%. Both are layered onto S22's literals later.
    - 2026-09-24 21:39 lead: ROUTING PARKED (engineer2 in-situ pricing): tree-kernel routing of the low-rank star's thin products gives only -0.25% of the bill within the ~2k-request budget (all 116 products: -0.77% at +8.6k requests). That is ~10-14 ms of residual, the whole per-build budget, for -0.25%: not worth it while the platform residual mean drifts up (.242 s, max .281 of .4). s1 r32 (request-neutral) proceeds alone on S21 to the VALID40 gate.
    2026-09-24 21:46 [engineer2] ROUTING of the low-rank star's plain products (lead GO item 1): priced in situ first (lowrank/thin_route_bench.py on S21): per product depth 2/3/4 = -18.6/-25.2/-28.7% (1024x1024x48) and ~-17/-22/-23% (32-wide), but 45/60/75 op calls vs 1; all 116 routable products = -0.52..-0.77% of the bill for +5.1k..+8.6k requests (breaks the residual budget). Built the request-optimal subset: the three star-plane range-finder LEFT products (A Omega, A^T Q, A Q') at depth 2 (45/net, ~+2.0k requests): fable_engineer2_v1/lrroute/S21_lrroute_mem.py 7ec98f9a = S21 eddbcfcd + lrroute_fragment.py (depth-2 A region 3.21M floats, 3 prebind keys). PARITY vs S21 (35/100/37/39): bill -0.249% measured in situ (per net -0.240..-0.256%), MSE -0.056% (rounding noise), adjusted -0.304%, maxrel 2.2-2.6e-6. validate/einsum PASS; user-CPU pair + paired residual gate running. Item 2 (s1 r32) built on top: lrroute/S21_lrr_s1lr_mem.py 21275e0a (+ s1lr_fragment.py; C w single via _mm at depth 6, s1 = L (R w) from the unrouted _lr_factor); its parity then VALID40 vs S21 rows are queued behind the routing gates (yielding to the porter's S22 chain).
    2026-09-24 21:48 [engineer2] ROUTING (tree-kernel routing of the low-rank star's plain products) PRICED: -0.25% for +2k requests, PARKED on the residual budget (lead 21:47). Measured: best subset = the three star-plane range-finder left products at depth 2 (45/net, ~+1,980 requests): in-situ parity bill -0.249%, adjusted -0.304%, rounding-level; full routing of all 116 products = -0.52..-0.77% for +5.1k..+8.6k requests. Build kept for the record: fable_engineer2_v1/lrroute/S21_lrroute_mem.py 7ec98f9a (+ user-CPU pair files platfit/optime_e2_S21lrr*.json); its residual gate was stopped. Do not re-derive; revisit only if the residual budget opens.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lowrank/thin_route_bench.py](code/lowrank/thin_route_bench.py)
- [code/lrroute/S21_lrroute_mem.py](code/lrroute/S21_lrroute_mem.py)
- [code/lrroute/lrroute_fragment.py](code/lrroute/lrroute_fragment.py)
- [code/lrroute/S21_lrr_s1lr_mem.py](code/lrroute/S21_lrr_s1lr_mem.py)
- [code/lrroute/s1lr_fragment.py](code/lrroute/s1lr_fragment.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L664](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L664)
- [TEAM_LOG_2026-09-22_to_09-26.md#L665](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L665)
- [TEAM_LOG_2026-09-22_to_09-26.md#L666](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L666)
- [TEAM_LOG_2026-09-22_to_09-26.md#L667](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L667)
- [TEAM_LOG_2026-09-22_to_09-26.md#L669](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L669)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-tree-routing-thin-residual-plane-products-saves-only-0) · [Copy provenance](../../../PROVENANCE.json)

