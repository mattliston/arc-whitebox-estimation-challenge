# Power-step join sketches failed under fast-kernel rounding despite better offline bases

2026-10-17T22:51:50Z · experiment · rejected · measured

A join sketch built from heavy columns and a power step improved the offline rank-384 basis, but unwhitened and whitened native variants worsened final error. Correcting subsample cores did not repair the effect; measured fast-kernel rounding destroyed the tail directions amplified by the power step.

Recorded measurement:

    Unwhitened rank 35: raw +28.3%; whitened 8 ranks: raw +1.68% (SE .69), bill +0.12%, adjusted +1.80%; full trace at 35/200: adjusted +3.75%/+6.60%; kernel relative Frobenius error 4–8e-6, 10–25x BLAS; noise 3e-7 gives plane error 2.5–4x.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/jsk`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_kernel_noise_v1.py](code/probe_kernel_noise_v1.py)
- [code/make_joinsketch_v1.py](code/make_joinsketch_v1.py)
- [code/make_joinsketch_v4.py](code/make_joinsketch_v4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4991](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4991)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4994](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4994)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4997](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4997)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4998](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4998)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4999](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4999)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5004](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5004)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5005](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5005)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5006](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5006)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5007](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5007)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [JOINSKETCH_v1.md](JOINSKETCH_v1.md) · record
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [GRADE_STORE_v1-262e8e55.json](GRADE_STORE_v1-262e8e55.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-power-step-join-sketches-failed-under-fast-kernel-rounding) · [Copy provenance](../../../PROVENANCE.json)

