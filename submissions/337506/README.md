# Submission #337506: First-128 readout with fully charged residual-time repair

2026-10-21T10:26:46Z · submission · adopted · measured

First-128 hidden features were combined with fixed runtime changes and removal of diagnostic census work, with all actual operation costs included. All hosted cases passed and the source reached third place.

Recorded measurement:

    Official submission record: {"submission_id": 337506, "date": "2026-10-21T10:26:46Z", "status": "GRADED", "score_public": 1.4092527407367254e-09, "n": 100, "failed_networks": 0, "utilization": 0.10835702051962472, "wall_mean_s": 101.166, "wall_max_s": 114.913, "resid_mean_s": 0.336, "resid_max_s": 0.364, "source_sha256": "f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6", "archive_sha256": "c63a799cc9402f3fb9aecf9b3fb5bf059db65362b9c8dc63803a83d7176f3d1d"}
    Ledger: - Independent hosted audit705564ac executedONCE andpassed all100: NC/hosted337506_runtime_v1/INDEPENDENT_FINAL_HOSTED_v1.jsonc2f476b0. Root rehashed all7inputs, independently verified raw100/five setups, boardthird/quota3used7remaining, and original12Macguards/32finite49bd predictions. ROOT_BEST337506_PROMOTION_v1.json30d87578 at11:11:56Z preserves prior9597222f as K/BEST_GRADED_REFERENCE_before337506_20261021T111156Z.json; newpointer53967437d250d03f9e229ed34157d75c1323b7ef59e2373e337bca133f9b6651. BEST337506=1.4092527407367254e-9, SCI49bd2ba3/RELf9381d09. Oldsecond-fireauthorityremoved, third-onlytradeconsumed. Allnewgainsuse actualpairedlatest337506 plusallbills.
    ### 2026-10-21T15:21:38Z [fable-lead] #337534 FINAL: GRADED score 0.0351 with 2 public TIME_EXHAUSTED failures (idx 3,4) -> excluded from board; BEST stays #337506. Release archive for s1.5 refit prepared NOT fired: tag fable_onpolicy_s1p5_aaec3e8e_v1 archive 7cb0d37a6860 (gates clear). Extended fit relaunched unguarded BLAS4 at 15:20:28Z (guard group_timeout killed the first at 1800 s). Plan per Matt: [instruction removed]) then spend part on a cohort/age cut for >= -8% requests so the net stays <= ~1.404e-9 with margin.
    ### 2026-10-21T17:42:50Z [fable-lead] #337565 GRADED 1.4036387231743654e-9, 100/100 clean, zero failures (hosted -0.40% vs #337506; wall max 111.1 s, mean 97.6 s, residual max 0.370). NEW BEST; J2W gap +0.88%. BEST pointer promoted (prior preserved as BEST_GRADED_REFERENCE_before337565_*). 5 shots remain; next = sub-1.40e-9 with margin; cap4-gate held (FIRE_cap4_gate.sh ready).

Public score: 1.4092527407367254e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [audit_hosted_v1.py](audit_hosted_v1.py)
- [audit_hosted_v2.py](audit_hosted_v2.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1759](../../records/SUBMISSION_LEDGER.md#L1759)
- [SUBMISSION_LEDGER.md#L1777](../../records/SUBMISSION_LEDGER.md#L1777)
- [SUBMISSION_LEDGER.md#L1781](../../records/SUBMISSION_LEDGER.md#L1781)
- [SUBMISSION_LEDGER.md#L1783](../../records/SUBMISSION_LEDGER.md#L1783)
- [SUBMISSION_LEDGER.md#L1785](../../records/SUBMISSION_LEDGER.md#L1785)
- [SUBMISSION_LEDGER.md#L1787](../../records/SUBMISSION_LEDGER.md#L1787)

<details>
<summary>All result files and records (15)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [package.json](package.json) · result
- [validate.json](validate.json) · result
- [validate_package.json](validate_package.json) · result
- [stage_a.json](stage_a.json) · result
- [ROOT_BEST337506_PROMOTION_v1.json](ROOT_BEST337506_PROMOTION_v1.json) · result
- [PREPARATION_FIXTURES_v2.json](PREPARATION_FIXTURES_v2.json) · result
- [PREPARATION_SCHEMA_FINDING_v1.json](PREPARATION_SCHEMA_FINDING_v1.json) · result
- [INDEPENDENT_FINAL_HOSTED_v1.json](INDEPENDENT_FINAL_HOSTED_v1.json) · result
- [PREPARATION_SETUP_SCHEMA_REVIEW_v2.json](PREPARATION_SETUP_SCHEMA_REVIEW_v2.json) · result
- [PREPARATION_FIXTURES_v1.json](PREPARATION_FIXTURES_v1.json) · result
- [PREPARATION_ID_SCHEMA_FINDING_v1.json](PREPARATION_ID_SCHEMA_FINDING_v1.json) · result

</details>

## Related work

- ← shipped in: [First-128 hidden readout: -.367205943% Mac32 and -.4015029588562216% Linux32 versus #337466](../../experiments/on-policy-readout/first-128-hidden-readout-367205943-mac32-4015029588562216-linux32-versus/README.md)
- ← shipped in: [First-128 runtime repair: cold residual -3.863232%, replicated warm residual -4.260674% pooled](../../experiments/wall-and-residual/first-128-runtime-repair-cold-residual-3-863232-replicated/README.md)
- ← shipped in: [Paid first-128 candidate confirms -.591104203% adjusted on 12 known fresh networks](../../experiments/evaluation-methodology/paid-first-128-candidate-confirms-591104203-adjusted-12-known/README.md)
- ← shipped in: [Paid kernel and copy restorations preserve first-128 predictions and reduce three-network residual time 4.330961%](../../experiments/wall-and-residual/paid-kernel-copy-restorations-preserve-first-128-predictions-reduce/README.md)
- ← shipped in: [Unused frozen-object census removed: exact predictions and FLOPs, 13 cold residuals all lower](../../experiments/wall-and-residual/unused-frozen-object-census-removed-exact-predictions-flops-13/README.md)

[Research graph](../../graph/index.html#node=submission-337506) · [Copy provenance](../../PROVENANCE.json)

