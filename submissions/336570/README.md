# Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual

2026-10-16T20:03:26Z · submission · failed · measured

V-A-noB (ff109deb) = the estimator submitted as #336412 (S2tr) + dm2 + lazybind v2 + JORD + DC35 + L15RC30 + DCV2 with option B removed, fired on the user word as a non-clean build (S2tr-level requests, median .355, P(public fail) about 49%) with a seeds-basis projection of 1.4589e-9 (below oliver 1.4606e-9) and a store-basis projection of about 1.470e-9. The cold first network of one worker, public MLP3, exceeded the residual cap at 0.430 s (the other cold nets .363-.383), so the run posted a penalized public score rather than a valid improvement. Its 49 completed public networks still served as a paired measurement of the four parts, which came out at the store value, not the seed value. The dieted iffy-noB variants could not be released because the STEP6 replay helper rejected a setup that binds all 649 keys.

Recorded measurement:

    Prereg seeds-basis 1.4589e-9 (bill -0.771% vs S2tr, dmse -0.384), store-basis ~1.470e-9; requests 47,386/45,499; worst wall 109.7 s; graded aggregate 0.0147, 1 failed, util 0.11168, wall mean 95.982 / max 110.388 s, residual mean 0.356 / max 0.43 s. 49 public nets paired vs #336412: final MSE +0.318% (SE .13, 13/49 lower), bill -0.541%, score -0.223% (31/49), aggregate -0.213% -> would have posted ~1.4727e-9.

Public score: 0.014699743050044517. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [cand_vanob.py](cand_vanob.py)
- [public_pair_v1.py](public_pair_v1.py)
- [../../records/research/2026-10-16-score-versus-runtime-risk-comparison-projected-1-0-score/cand_frontier_VAB_v1.py](../../records/research/2026-10-16-score-versus-runtime-risk-comparison-projected-1-0-score/cand_frontier_VAB_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4475](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4475)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4479](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4481](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4481)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4483](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4483)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4487](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4487)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4489](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4489)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4491](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4491)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4493](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4493)
- [SUBMISSION_LEDGER.md#L1030](../../records/SUBMISSION_LEDGER.md#L1030)
- [SUBMISSION_LEDGER.md#L1031](../../records/SUBMISSION_LEDGER.md#L1031)
- [SUBMISSION_LEDGER.md#L1032](../../records/SUBMISSION_LEDGER.md#L1032)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_336570.json](eval_336570.json) · result
- [STACK_vanob.json](STACK_vanob.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Dead-cut z raised 3.0902 to 3.5: -0.234% MSE for +0.065% bill on 24 seeds (rate 3.6); 4.0, 4.5 and off only add bill](../../experiments/covariance-structure/dead-cut-z-raised-3-0902-3-5-0/README.md)
- ← shipped in: [Dead-unit covariance floor 0.002 at all layers: 24 seeds net -0.082%, bill -0.148%, stacked as a marginal keeper](../../experiments/covariance-structure/dead-unit-covariance-floor-0-002-all-layers-24/README.md)
- ← shipped in: [Layer-15 row cut z lowered 3.5 to 3.0: 24 seeds net -0.315% at bill -0.360%; 2.75 and 2.5 are a cliff](../../experiments/late-layer-exactness/layer-15-row-cut-z-lowered-3-5-3/README.md)
- ← enables: [Score-versus-runtime-risk comparison: a projected 1.0% score gain retains a 46–57% public-failure probability](../../records/research/2026-10-16-score-versus-runtime-risk-comparison-projected-1-0-score/README.md)
- measured by: [Four value-only changes on 49 public networks: MSE +0.318%, bill -0.541%, adjusted -0.223%](../../records/research/2026-10-16-four-value-only-changes-49-public-networks-mse-0/README.md)

[Research graph](../../graph/index.html#node=submission-336570) · [Copy provenance](../../PROVENANCE.json)

