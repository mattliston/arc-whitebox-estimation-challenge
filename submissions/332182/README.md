# Submission #332182: upper-block covariance products, lean views, and an epoch-1 refit

2026-09-24T17:24:14Z · submission · adopted · measured

Added upper-block covariance products, lean view handling, and epoch-1 joint-refit literals. The official public score is 3.046329368760037e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332182, "date": "2026-09-24T17:24:14Z", "status": "GRADED", "score_public": 3.046329368760037e-09, "n": 100, "failed_networks": 0, "utilization": 0.1721499846495499, "wall_mean_s": 93.244, "wall_max_s": 98.891, "resid_mean_s": 0.265, "resid_max_s": 0.315, "source_sha256": "acf67b9d59c8b68f6eb7ab0f1b9ce8209f11f5b40b0ac792d223ecce5e89a24a"}
    - 2026-09-24 10:24 PDT FIRE #332182 (Fable): fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py (sha256 acf67b9d59c8b68f6eb7ab0f1b9ce8209f11f5b40b0ac792d223ecce5e89a24a) = Fable S11: S9 #332037 + engineer's --symout v3 (covariance second product as upper-block plain products, G21 mirrored; no symmetric einsum) + --lean v3 (fnp.split quadrants + function-scoped holds; bit-exact) + B2 closure refit EPOCH 1 (params_epoch1.pt). push5p6 kept.
    - 2026-09-24 10:56 PDT GRADED #332182 (S11): adjusted 3.046329368760037e-9, raw 1.7707661381649587e-8, util 0.17215. NEW PHASE-2 BANK, -1.20% vs #332037 (3.0834e-9). Expected ~3.044e-9 (VALID40 -1.29%), so the local prediction held to 0.09 pt.
    - 2026-09-24 13:53 PDT FIRE #332195 (Fable): fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py (sha256 2a5dd324d78c4db6bc457485ce050a1e93d0ec2dea2654d0a6346b7de4119249) = Fable S13, built on S11 #332182 in this order:
    - 2026-09-24 ~14:30 PDT GRADED #332195 (S13): adjusted 2.9742839646322686e-9, raw 1.773872847721236e-8, util 0.16776. NEW PHASE-2 BANK and FIRST SUB-3e-9 (the 3e-9 milestone), -2.37% vs #332182 (3.0463e-9).

Public score: 3.046329368760037e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L724](../../records/SUBMISSION_LEDGER.md#L724)
- [SUBMISSION_LEDGER.md#L728](../../records/SUBMISSION_LEDGER.md#L728)
- [SUBMISSION_LEDGER.md#L732](../../records/SUBMISSION_LEDGER.md#L732)
- [SUBMISSION_LEDGER.md#L743](../../records/SUBMISSION_LEDGER.md#L743)
- [TEAM_LOG_2026-09-22_to_09-26.md#L318](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L318)
- [TEAM_LOG_2026-09-22_to_09-26.md#L328](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L328)

<details>
<summary>All result files and records (5)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [poll.log](poll.log) · log
- [eval_332182.json](eval_332182.json) · result

</details>

## Related work

- ← shipped in: [Smaller request and alias reductions preserve memory while removing about 938 warm round trips](../../experiments/arithmetic-kernels/smaller-request-alias-reductions-preserve-memory-while-removing-about/README.md)
- ← shipped in: [Upper-block covariance and reduced requests improve adjusted error 1.094% on 40 networks](../../experiments/arithmetic-kernels/upper-block-covariance-reduced-requests-improve-adjusted-error-1/README.md)
- supersedes: [Submission #332037: joint closure refit including graph-neural-network trunk weights](../332037/README.md)
- ← supersedes: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../332195/README.md)

[Research graph](../../graph/index.html#node=submission-332182) · [Copy provenance](../../PROVENANCE.json)

