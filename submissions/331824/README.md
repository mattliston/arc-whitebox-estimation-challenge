# Submission #331824: 1,536-value message-to-covariance context fit improves adjusted error 0.018%

2026-09-22T03:50:28Z · submission · adopted · measured

The sealed message-to-covariance release trains only a terminal 1,536-parameter context fit on top of the frozen #331760 estimator (all other parameters, the moment closure and the bank unchanged) and was uploaded as a single-file archive. It graded as the official bank at the start of the slice: 100/100 networks, zero failures, a 0.018% adjusted improvement over #331760 and a 0.47% lower raw final-layer MSE.

Recorded measurement:

    adjusted 3.5070948899311718e-9 (previous bank #331760 3.507738856975084e-9, relative improvement 0.0001835846595683588); raw 1.9005926183979227e-8 (relative raw improvement 0.0046842288596854464); 100 complete, 0 failed; flops per network min/mean/max 3.81024555148e11 / 4.0238599244434e11 / 4.25364436419e11; predict wall min/mean/max 93.446 / 99.982 / 110.168 s; residual min/mean/max 0.2984 / 0.3179 / 0.3854 s; utilization 0.1845888141453179

Public score: 3.5070948899311718e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [OFFICIAL_RESULT.md#L4](OFFICIAL_RESULT.md#L4)
- [official_best_331824.json#L23](official_best_331824.json#L23)
- [PLAN.md#L4](PLAN.md#L4)
- [TEAM_BRIEF.md#L5](TEAM_BRIEF.md#L5)

<details>
<summary>All result files and records (24)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [official_best_331824.json](official_best_331824.json) · record
- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [PLAN.md](PLAN.md) · record
- [message_covariance_science_pass_release_active_checkpoint132.md](message_covariance_science_pass_release_active_checkpoint132.md) · record
- [eval_331824.json](eval_331824.json) · result
- [TEAM_BRIEF.md](TEAM_BRIEF.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [official_best_331824-cad6cd8a.json](official_best_331824-cad6cd8a.json) · result
- [evaluation_331824_100.json](evaluation_331824_100.json) · result
- [official_completion_331824.json](official_completion_331824.json) · result
- [official_best_331824-5aedc349.json](official_best_331824-5aedc349.json) · result
- [official_result_331824.json](official_result_331824.json) · result
- [evaluation_331824_01.json](evaluation_331824_01.json) · result
- [evaluation_331824_02.json](evaluation_331824_02.json) · result
- [evaluation_331824_03.json](evaluation_331824_03.json) · result
- [evaluation_331824_04.json](evaluation_331824_04.json) · result
- [evaluation_331824_05.json](evaluation_331824_05.json) · result
- [evaluation_331824_100-db2c1e7b.json](evaluation_331824_100-db2c1e7b.json) · result
- [evaluation_331824_101.json](evaluation_331824_101.json) · result
- [evaluation_331824_102.json](evaluation_331824_102.json) · result
- [evaluation_331824_103.json](evaluation_331824_103.json) · result
- [official_completion_331824-d6c63346.json](official_completion_331824-d6c63346.json) · result
- [official_result_331824-cad6cd8a.json](official_result_331824-cad6cd8a.json) · result

</details>

## Related work

- supersedes: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../331760/README.md)
- ← supersedes: [Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer](../331947/README.md)

[Research graph](../../graph/index.html#node=submission-331824) · [Copy provenance](../../PROVENANCE.json)

