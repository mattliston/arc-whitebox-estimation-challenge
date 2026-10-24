# Submission #333481: fitted dials, masked covariance correction and three-request operand loads

2026-10-02T07:57:10Z · submission · adopted · measured

S38 is the S37 stack rebuilt on the trainer's dial-only candidate 2 (learned mean correction x 0.8, per-unit kurtosis term 0.75 on both ages, covariance knob k21n -0.048, b-source 0.965, bscale 1.03414, X1 0.59158, column shift 0.656 / 0.6) with the learned covariance edge cut to response rows x alive columns and the three-request operand loads. Qualified by stages A, C (same-window legs S35 / file / file / S35) and D in a 32-minute quiet window; graded 1.907e-9, 4th place; passed the designation test including the slow-setup session.

Recorded measurement:

    adjusted 1.906919202087841e-9 (S37 1.9510e-9, -2.26%; expected about 1.91e-9); final-layer MSE 1.6120e-8 (-1.65%; torch 50 held-out -1.44%, native 10 -2.03%), all-layer 6.377e-9 (-1.25%), utilisation 0.11830070687306943 (-0.60%); 100/100, wall 77.07 / 88.27 s (projected 74.9 / 86.3), residual 0.294 / 0.332 s (projected 0.295 / 0.328); stage C requests 41,862 a network, bill 0.97871 x S35; stage D handles 57,578, group RSS 6.81 GB (footprint 2.20 GB); native ten fresh vs S37: MSE -2.03% [-3.12,-0.99], bill -0.624%, adjusted -2.66%; residual per request 7.03 microseconds; binding phase 17,364 requests (91.4% of a normal worker's); designation: residual room 0.106 s, slow setup +1.83% requests (limit +2%); board 4th (Luna 1.8437e-9, marius_binner 1.6407e-9, J2W 1.6233e-9)

Public score: 1.906919202087841e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py](candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py)

## Evidence

- [SUBMISSION_LEDGER.md#L911](../../records/SUBMISSION_LEDGER.md#L911)
- [SUBMISSION_LEDGER.md#L912](../../records/SUBMISSION_LEDGER.md#L912)
- [TEAM_LOG_2026-10-01_to_10-22.md#L166](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L166)
- [TEAM_LOG_2026-10-01_to_10-22.md#L171](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L172](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L172)
- [TEAM_LOG_2026-10-01_to_10-22.md#L182](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L182)
- [TEAM_LOG_2026-10-01_to_10-22.md#L183](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L183)
- [TEAM_LOG_2026-10-01_to_10-22.md#L201](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L201)
- [TEAM_LOG_2026-10-01_to_10-22.md#L202](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L205](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L205)
- [TEAM_LOG_2026-10-01_to_10-22.md#L216](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L216)
- [TEAM_LOG_2026-10-01_to_10-22.md#L217](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L217)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L464](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L464)

<details>
<summary>All result files and records (13)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary.txt](eval_summary.txt) · result
- [eval_333481.json](eval_333481.json) · result
- [lead_s38_chain.log](lead_s38_chain.log) · log
- [stage_c.json](stage_c.json) · result
- [release_s38_slow.log](release_s38_slow.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333481-32f9d3d2.json](eval_333481-32f9d3d2.json) · result
- [eval_333481-08883bcb.json](eval_333481-08883bcb.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Gradient-directed dial line search lowers error by 1.44% on 50 held-out networks](../../experiments/dials/gradient-directed-dial-line-search-lowers-error-1-44/README.md)
- ← shipped in: [Masked learned covariance correction lowers charged work by 0.69% for 0.02% additional error](../../experiments/learned-correction/masked-learned-covariance-correction-lowers-charged-work-0-69/README.md)
- ← shipped in: [Qualification requires quiet compute windows during instruction-count and worker stages](../../records/research/2026-10-02-qualification-requires-quiet-compute-windows-during-instruction-count-worker/README.md)
- ← shipped in: [Scaling the learned mean correction to 0.8 lowers fresh-network error by 0.84%](../../experiments/learned-correction/scaling-learned-mean-correction-0-8-lowers-fresh-network/README.md)
- ← shipped in: [Submission designation requires successful grading and measured wall and residual margins](../../records/research/2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/README.md)
- ← shipped in: [Three-request operand loads and paired output writes remove 1,373 calls while preserving bytes](../../experiments/wall-and-residual/three-request-operand-loads-paired-output-writes-remove-1/README.md)
- supersedes: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../333435/README.md)
- ← supersedes: [Submission #333528: joint dial solve and third-order remainder carried by row sums cut bill 2.2%](../333528/README.md)

[Research graph](../../graph/index.html#node=submission-333481) · [Copy provenance](../../PROVENANCE.json)

