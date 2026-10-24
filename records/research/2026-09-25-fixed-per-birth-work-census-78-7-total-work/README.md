# Fixed and per-birth work census: 78.7% of total work scales with birth columns

2026-09-25T04:16Z · finding · informational · measured

The census separates young reads and pushes, old-plane reconstruction and joins from covariance and learned fixed costs. Final-layer old columns were already pushed within the frame, leaving no saving from that proposed reassociation.

Recorded measurement:

    2026-09-24 21:16 porter: S21 BILL CENSUS (fable_porter_v1/census_S21.json; bill_census_layers.py PT-38 + census_S21_report.py; fired S21 eddbcfcd; 1 U = 2*1024^3 flops): net 35 cold 157.0 U = PER-BIRTH 123.6 U (78.7%) + FIXED 33.4 U (21.3%); net 100 warm 155.0 U (78.6/21.4%). Per-birth: young read 23.4%, young push 22.7%, old-column reconstruct (U F, every layer) 11.5%, old read 7.9%, joins L7/10/13 7.5%, newborn push 3.7%, old basis push 1.0%, other 1.1%. Fixed: covariance 8.5%, low-rank star 7.6%, pair map 1.6%, GNN 1.3%, edge 1.2%, rest 1.1%. Unit costs (L1-14, in-layer counts): young col 0.692 U/1k/layer (push .332 + read .360); old col 0.323 U/1k/layer (reconstruct .180 + read .143) + basis push .174 U/layer; newborn .492 U/1k births; join .718 U/1k joined cols; fixed 2.28 U/layer; L15 bank pass 4.75 U. Columns: 2048 (L1) -> 23872 (L15 in), births 1024 -> 544/layer, old tier 6144/11712/16384 cols at rank 384, live rows 1024 -> 768.
    - 2026-09-24 21:19 lead: S21 CENSUS read (porter, census_S21.json; net 35 cold 157.0 U, net 100 155.0 U; 1 U = 2*1024^3):
      - Per-birth 78.7%: young read 23.4%, young push 22.7%, old-tier rebuild (U F every layer) 11.5%, old read 7.9%, joins 7.5%, newborn 3.7%, old basis 1.0%.
      - Fixed 21.3% = 2.28 U/layer: covariance 8.5%, low-rank star 7.6% (its range finder scales with columns), pair map 1.6%, GNN/edges 2.5%.
      - L15 re-pushes the whole bank (4.75 U, 3.0%) for the row-cut k3 diagonal.
      - Columns: 2048 at L1 -> 23872 into L15; births 1024 at L1 down to 544 at L13, 2 columns each.
      - Prior closures that bound this: E-48 (per-column price at floor; old rebuild on response rows +12% raw; alive rows only +21.9%), lean compression (fold/Tucker/pruning).
      - suliman's -3.9e10 offset is 1.1 U/layer, about half our 2.28 U/layer fixed cost.
    2026-09-24 21:24 porter: PRICING (lead, code reading only; S21 eddbcfcd, census net 35). (1) L15 RE-PUSH: the old part is ALREADY in the frame - _l15_rowcut_k3 runs _shared_old384_push on the kept rows: U15 = W15[:, rows]^T Q (0.12 U) then old = U15 F (2.53 U, inner dim 384); rebuild-then-push would be ~2.7x that. U15 F is intrinsic: the k3 cube needs every old column's value on every kept row (a frame-only form needs a 384^3 core contraction per row, ~10x more). Saving 0 U. (2) JOINS (11.77 U = 7.5%; L7 4.21, L10 4.28, L13 3.28 U): each join is ALREADY a randomized range finder - Y = J (w * (J^T Omega)) + Qp core Qp^T Omega, Omega = W_l[live, :384] (Gaussian-like weight columns), sketch = rank = 384 (no oversampling, no power step; the Gram form = ~half a power step on A = J w^1/2), then qr(Y), Fj = Qn^T J, joined core Fj w Fj^T, old-tier overlap/factors/core. Sizes: J = live x width = 960x6144 (L7), 832x5568 (L10), 832x4672 (L13). U by product: J^T Omega 2.76, J(.) 3.01, Fj 2.80, core 1.36, qr 0.62, old-tier parts ~1.2. No exact decomposition to replace. Cheaper variants all change the frame (NOT rounding-level): q=0 on A with a random width x 384 test matrix drops J^T Omega (-2.76 U = -1.76% bill) but loses the Gram's implicit half power step (star-plane lesson: q=0 doubled the error on a flat spectrum); a row-selection Omega (Nystrom) drops it too with coherence risk. Either needs the no-refit VALID16 <= +0.1% raw first. Neither item meets the rounding-level / near-zero-risk bar.

## Programs

- [bill_census_layers.py](bill_census_layers.py)
- [census_S21_report.py](census_S21_report.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L638](../../TEAM_LOG_2026-09-22_to_09-26.md#L638)
- [TEAM_LOG_2026-09-22_to_09-26.md#L639](../../TEAM_LOG_2026-09-22_to_09-26.md#L639)
- [TEAM_LOG_2026-09-22_to_09-26.md#L640](../../TEAM_LOG_2026-09-22_to_09-26.md#L640)
- [TEAM_LOG_2026-09-22_to_09-26.md#L641](../../TEAM_LOG_2026-09-22_to_09-26.md#L641)
- [TEAM_LOG_2026-09-22_to_09-26.md#L642](../../TEAM_LOG_2026-09-22_to_09-26.md#L642)
- [TEAM_LOG_2026-09-22_to_09-26.md#L643](../../TEAM_LOG_2026-09-22_to_09-26.md#L643)
- [TEAM_LOG_2026-09-22_to_09-26.md#L644](../../TEAM_LOG_2026-09-22_to_09-26.md#L644)
- [TEAM_LOG_2026-09-22_to_09-26.md#L645](../../TEAM_LOG_2026-09-22_to_09-26.md#L645)
- [TEAM_LOG_2026-09-22_to_09-26.md#L651](../../TEAM_LOG_2026-09-22_to_09-26.md#L651)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [census_S21.json](census_S21.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-fixed-per-birth-work-census-78-7-total-work) · [Copy provenance](../../../PROVENANCE.json)

