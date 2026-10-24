# Submission #332255: joint refit, arithmetic-level fusion, and a Nyström residual-plane range finder

2026-09-25T08:40:22Z · submission · adopted · measured

Added a joint fit of learned values and physics scales, fused arithmetic levels to reduce requests, and a Nyström residual-plane range finder. The official public score is 2.372624767712433e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332255, "date": "2026-09-25T08:40:22Z", "status": "GRADED", "score_public": 2.372624767712433e-09, "n": 100, "failed_networks": 0, "utilization": 0.14098132386730866, "wall_mean_s": 67.189, "wall_max_s": 72.261, "resid_mean_s": 0.212, "resid_max_s": 0.256, "source_sha256": "30810da3b302c5ec5ee9c7060dfd4d3956446d45e0d3ef2bfdf6ca5caebe2084"}
    - 2026-09-25 01:40 PDT FIRE #332255 (Fable): fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py (sha256 30810da3...) = Fable S24 = S23 + the physicist's loopR3 JOINT refit (learned parts + 10 dials, S23-structure card, MPS; CPU-verified VALID16 -0.525%) + the engineer's E-71 level-op fusion (lvfuse2; bit-identical, -3.2k requests/net) + the porter's Nystrom star range finder (48 top-norm columns + one power step).
    - 2026-09-25 02:05 PDT GRADED #332255 (S24): adjusted 2.372624767712433e-9, raw 1.6833305132024634e-8, util 0.14098. NEW PHASE-2 BANK, -0.94% vs #332246 (2.3951e-9); expected ~2.375e-9 (VALID40 -0.854%).
    - 2026-09-25 09:28 PDT FIRE #332295 (Fable): fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py (sha256 9ff80a12...) = Fable S27 = S24 #332255 + engineer2's j3cut + the k-split v4 package (engineer; census-derived 372-key prebind list, setup guard 3.5 s) + treuse (engineer E-73, applied last) + covhunter's capacity arm-A refit literals (2 learned-only epochs from loopR3; params 4e9be88d; 28 literals bit-equal; dials = loopR3's). Built by the porter's chain (fire_chain_S27.sh).
    - 2026-09-25 09:58 PDT GRADED #332295 (S27): adjusted 2.340045956285692e-9, raw 1.677886261930439e-8, util 0.13950. NEW PHASE-2 BANK, -1.37% vs #332255 (2.3726e-9); expected ~2.343e-9 (VALID40 -1.235%).

Public score: 2.372624767712433e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L875](../../records/SUBMISSION_LEDGER.md#L875)
- [SUBMISSION_LEDGER.md#L881](../../records/SUBMISSION_LEDGER.md#L881)
- [SUBMISSION_LEDGER.md#L885](../../records/SUBMISSION_LEDGER.md#L885)
- [SUBMISSION_LEDGER.md#L892](../../records/SUBMISSION_LEDGER.md#L892)
- [TEAM_LOG_2026-09-22_to_09-26.md#L898](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L898)
- [TEAM_LOG_2026-09-22_to_09-26.md#L939](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L939)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332255.json](eval_332255.json) · result

</details>

## Related work

- ← shipped in: [Joint learned-parameter and physics-dial GPU refit passes: CPU validation improves 0.525%](../../experiments/learned-correction/joint-learned-parameter-physics-dial-gpu-refit-passes-cpu/README.md)
- ← shipped in: [Joint refit passes forty-network validation: raw error improves 0.543%](../../experiments/learned-correction/joint-refit-passes-forty-network-validation-raw-error-improves/README.md)
- ← shipped in: [Nyström residual-plane range finder with one power iteration improves native adjusted error 0.320%](../../experiments/third-order-planes/nystrom-residual-plane-range-finder-one-power-iteration-improves/README.md)
- ← shipped in: [Strided level-operation fusion cuts requests 9.5% but increases user CPU 7.9%](../../experiments/arithmetic-kernels/strided-level-operation-fusion-cuts-requests-9-5-but/README.md)
- supersedes: [Submission #332246: cross-term removal and Nyström old-cohort joins](../332246/README.md)
- ← supersedes: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../332295/README.md)

[Research graph](../../graph/index.html#node=submission-332255) · [Copy provenance](../../PROVENANCE.json)

