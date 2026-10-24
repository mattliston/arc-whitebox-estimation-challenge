# Final hidden-state by weight covariance feature, 2080 columns: five-fold ratio 1.1004, 0/5 folds win

2026-09-20T08:34:49Z · experiment · rejected · measured

The third version reads a signed low-rank covariance feature (sqrt(C_diag)*H14 on active rows, literal signed C_diag*H_a*H_b on cold rows) through the final weight W15 as 2080 fixed upper-pair columns added to the 65-column affine head, fitted closed-form with ridge 1e-4 in five 16-fit/4-held network folds on TRAIN20. The extension raised the adjusted mean from 3.907171531554795e-9 (unchanged) to 4.299374328343792e-9 and no fold beat either comparator, so the fixed value test is negative.

Recorded measurement:

    adjusted means: unchanged 3.907171531554795e-9, affine control 3.893826867750025e-9, extension 4.299374328343792e-9; extension/unchanged 1.100380235068136, extension/control 1.104151384837535; fold ratios [1.0926893539880342, 1.1237916822827347, 1.1168279987675143, 1.0826008188632488, 1.0884944751522365]; gates <=0.95 / <=0.97 / >=4 of 5 folds; 19 new cold parent calls 7,573,603,518,547 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_latent_covariance_information_v3/capture_all.py](code/terminal_latent_covariance_information_v3/capture_all.py)
- [code/terminal_latent_covariance_information_v3/information.py](code/terminal_latent_covariance_information_v3/information.py)
- [code/terminal_latent_covariance_information_v3/qualify.py](code/terminal_latent_covariance_information_v3/qualify.py)
- [code/terminal_latent_covariance_information_v3/unit.py](code/terminal_latent_covariance_information_v3/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/read5_packet_v1/candidate.py](code/read5_packet_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/terminal_latent_covariance_information_v3/capture.py](code/terminal_latent_covariance_information_v3/capture.py)
- [code/terminal_latent_covariance_information_v3/common.py](code/terminal_latent_covariance_information_v3/common.py)
- [code/terminal_latent_covariance_information_v3/feature.py](code/terminal_latent_covariance_information_v3/feature.py)
- [code/terminal_latent_covariance_information_v3/fit_library.py](code/terminal_latent_covariance_information_v3/fit_library.py)
- [code/terminal_representation_v1/capture.py](code/terminal_representation_v1/capture.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/terminal_representation_v1/information.py](code/terminal_representation_v1/information.py)

</details>

## Evidence

- [NEGATIVE_RESULT.md#L5](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L7](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md#L7)
- [NEGATIVE_RESULT.md#L9](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L13](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md#L13)
- [NEGATIVE_RESULT.md#L15](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md#L15)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](../../../records/research/2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2/NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record
- [SCOPE.md](SCOPE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-final-hidden-state-weight-covariance-feature-2080-columns-five) · [Copy provenance](../../../PROVENANCE.json)

