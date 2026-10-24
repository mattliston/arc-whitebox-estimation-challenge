# Column-work audit finds existing old-plane reads already operate in the frame

2026-09-24T17:46Z · finding · informational · measured

Attribution of metered work to products showed that the old-plane read already used the low-rank frame. Remaining exact savings came from repeated transforms, copies and frame binding.

Recorded measurement:

    - 2026-09-24 10:46 [engineer] S11 BILL + COLUMN CENSUS DONE (fable_engineer_v1/RESULT.md E-47; card_census.py on the fired S11 acf67b9d, .venv, r35 cold then r100 warm; attributed bill = total exactly). Card: r35 179.7 U (util .17551), r100 176.4 U (.17228); 1 U = 2*1024^3 flops. Shares r35: covariance push 8.2%, young push 22.0%, star push 13.2%, old-tier functions 25.0%, atoms/plane read 22.1%, port body 4.2% (newborn 3.3%, frame 0.9%), other 5.2%: within 1 point of the S2 calibration. Per column per layer: young 8.0e-4 U (9-10e-4 at L1-6, 7.0e-4 at L8-14), old 3.36e-4 (rebuild 2.0e-4 + inner 1.4e-4), join 6.6e-4 once per joined column; fixed ~3.6 U/layer. Births carry 70% of the bill; lifetime price per birth by birth layer is tabled (19e-3 U at L0 down to 0.3e-3 at L14) for test 1's bill ratios. Sent to the physicist.
    - 2026-09-24 10:59 [engineer] PER-COLUMN COST AUDIT (fable_engineer_v1/RESULT.md E-48; paper + one row-class run, nothing built). At fixed physics S11's column work is already the minimum product structure: the old-tier read is ALREADY inside the frame (inner at rank 384, then x U[live]^T; L14 costs 10.0e9 MACs, not 16e9). The response-row rebuild is blocked because every column's cube on all 1024 rows feeds every unit's k3 (pair map, unmasked; a dead unit's k3 term at a=-3 is 1.7x its Gaussian mean at skewness 0.3). Fixed-physics savings: 0.50 U/net (0.28%) bit-identical (one W^T transform per young push, trims), +0.94 U (0.52%) if 1.1 GB of transforms are cached between joins, +0.36 U (0.20%) rounding-level (upper-block joined core). Physics-change row cuts: a<-3.5 rows 4.2 U (2.3%), response-row old rebuild 6.5 U (3.6%), alive rows only 8.0 U (4.5%). Cheapest physics test: the L15 row-cut rule on the dead side at L1-14, no-refit VALID16.
    - 2026-09-24 11:07 [engineer] E-48 ADDENDUM (op-level split, card_census_v2.py): S11 column work on warm r100 = 128.8 U/net = leaf products 109.4 U (85%), transform adds 12.3 (9.6%), frame binding 3.2 (2.4%; 6.2 cold), copies 2.8 (2.2%, Winograd level stacks), other 1.1. Exact options now: cheap bit-identical package 0.67 U (0.37%: shared W^T transform per young push 0.33, compact gated columns on next-push rows 0.22, pp^2 reuse + no roster concat 0.12); F-transform cache 1.15 U (0.64%, ~1.1 GB); copy-free levels <= 2.7 U and less binding <= 3.2 U warm, both costing requests/CPU. Ceiling of exact per-column savings ~4%; without new memory/CPU 0.37%. --p6k928 dropped (S11 banked 3.0463e-9, worst RT 99.3 s).

## Programs

- [card_census_v2.py](card_census_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L323](../../TEAM_LOG_2026-09-22_to_09-26.md#L323)
- [TEAM_LOG_2026-09-22_to_09-26.md#L329](../../TEAM_LOG_2026-09-22_to_09-26.md#L329)
- [TEAM_LOG_2026-09-22_to_09-26.md#L339](../../TEAM_LOG_2026-09-22_to_09-26.md#L339)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../../experiments/arithmetic-kernels/grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-column-work-audit-finds-existing-old-plane-reads-already) · [Copy provenance](../../../PROVENANCE.json)

