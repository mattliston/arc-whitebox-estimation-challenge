# Submission #337028 source audit: no measured change reaches 1.42e-9; 44,279 cold and 42,577 warm requests

2026-10-19T02:43:21Z · finding · informational · measured

Every lever in the record was re-checked against the fired file: an own prebind refit only trims first-network lazy binds (-14%, later networks 0) because lazy binds are about 35 new keys times about 35 requests per network whatever the list; level fusion already runs on about 85% of product sides; the cost-map reading of the Smirnov grouped read as a quadrature was a misreading (P is the rank of the exact <3,3,6;40> scheme); earlier joins hit the age-4 cliff; the refit-after-drop (RAD) generations S61/S62 were null and S63 stuck. A fresh request census of the file counted 44,279 requests cold / 42,577 warm per network over about 170 functions with no site above 112 requests. Conclusion: the realistic outcome by 10-21 was one bank step from request cuts and small verified items.

Recorded measurement:

    hosted 8 us/request; #336935 residual median .3516 / max .3995; the bundle adds ~+0.75% residual instructions, requests ~+0.1%; L7 rank 320 = +1.7% MSE on 32 ranks; earlier joins +10.7% MSE for -4.5%; census net 35 cold 44,279 / net 100 warm 42,577: tree transforms+unwinds ~13.3K, physics body ~16K over ~170 functions, views ~8K, casts ~1.3K, copyto 1.2K, Smirnov circuits 2.8K; next cut target _l3o_last 1,329/net

## Programs

- [refit_prebind_inplace_v3.py](refit_prebind_inplace_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5889](../../TEAM_LOG_2026-10-01_to_10-22.md#L5889)

<details>
<summary>All result files and records (2)</summary>

- [refit_v1.log](refit_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-submission-337028-source-audit-no-measured-change-reaches-1) · [Copy provenance](../../../PROVENANCE.json)

