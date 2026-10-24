# Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit

2026-09-25T16:28:57Z · submission · adopted · measured

Added row selection for the third birth contribution, an inner-dimension split product package with census-derived setup prebinding, transformation reuse, and a learned-only refit. The official public score is 2.340045956285692e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332295, "date": "2026-09-25T16:28:57Z", "status": "GRADED", "score_public": 2.340045956285692e-09, "n": 100, "failed_networks": 0, "utilization": 0.1394999356365588, "wall_mean_s": 70.18, "wall_max_s": 77.503, "resid_mean_s": 0.218, "resid_max_s": 0.284, "source_sha256": "9ff80a12de5a5fb8fcde319696db266b42bc81740e0906db036079dc64ff116e"}
    - 2026-09-25 09:28 PDT FIRE #332295 (Fable): fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py (sha256 9ff80a12...) = Fable S27 = S24 #332255 + engineer2's j3cut + the k-split v4 package (engineer; census-derived 372-key prebind list, setup guard 3.5 s) + treuse (engineer E-73, applied last) + covhunter's capacity arm-A refit literals (2 learned-only epochs from loopR3; params 4e9be88d; 28 literals bit-equal; dials = loopR3's). Built by the porter's chain (fire_chain_S27.sh).
    - 2026-09-25 09:58 PDT GRADED #332295 (S27): adjusted 2.340045956285692e-9, raw 1.677886261930439e-8, util 0.13950. NEW PHASE-2 BANK, -1.37% vs #332255 (2.3726e-9); expected ~2.343e-9 (VALID40 -1.235%).

Public score: 2.340045956285692e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L885](../../records/SUBMISSION_LEDGER.md#L885)
- [SUBMISSION_LEDGER.md#L892](../../records/SUBMISSION_LEDGER.md#L892)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1056](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1056)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1075](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1075)

<details>
<summary>All result files and records (7)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332295.json](eval_332295.json) · result
- [eval_332295-79b45dab.json](eval_332295-79b45dab.json) · result

</details>

## Related work

- ← shipped in: [Conservative forward-transform reuse saves 0.09% work and 376-483 requests with identical predictions](../../experiments/arithmetic-kernels/conservative-forward-transform-reuse-saves-0-09-work-376/README.md)
- ← shipped in: [Doubling learned width adds only about 0.08%; extra training improves forty-network error 0.377%](../../experiments/learned-correction/doubling-learned-width-adds-only-about-0-08-extra/README.md)
- ← shipped in: [Fourth birth-direction leg removal saves 0.091% work with approximately neutral pooled error](../../experiments/birth-terms/fourth-birth-direction-leg-removal-saves-0-091-work/README.md)
- ← shipped in: [Leaf-space contraction splitting improves adjusted error 0.807% on all 40 validation networks](../../experiments/arithmetic-kernels/leaf-space-contraction-splitting-improves-adjusted-error-0-807/README.md)
- ← shipped in: [Summing split contractions in leaf space removes repeated unwind work](../../experiments/arithmetic-kernels/summing-split-contractions-leaf-space-removes-repeated-unwind-work/README.md)
- supersedes: [Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder](../332255/README.md)

[Research graph](../../graph/index.html#node=submission-332295) · [Copy provenance](../../PROVENANCE.json)

