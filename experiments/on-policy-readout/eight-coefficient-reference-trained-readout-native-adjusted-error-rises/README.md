# Eight-coefficient reference-trained readout: native adjusted error rises 2.145536%

2026-09-27 · experiment · rejected · measured

The stable numerical reference produced an eight-coefficient readout fit that was transferred to the original fast estimator. The actual raw reduction was outweighed by charged compute on all six networks.

Recorded measurement:

    Fit norm0.256984,predicted linear raw ratio0.997652843;actual TRAIN6 raw-0.177681%,adjusted+2.145536%,0/6 adjusted wins;17 reference cases and10 exact repeat controls pass.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/assemble.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/assemble.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/assess_train.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/assess_train.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/candidate.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/candidate.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/capture.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/capture.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/check_pair_native.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/check_pair_native.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/common.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/common.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/fit.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/fit.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/launch_capture.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/launch_capture.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/launch_fit.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/launch_fit.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/native_train.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/native_train.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/prepare_targets.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/prepare_targets.py)
- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/screen_pairs.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/screen_pairs.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/verify_fit.py](code/codex_resume_20260926_v2/reference_readout_transfer_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)

</details>

## Evidence

- [RESULTS.md#L1621](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1621)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-eight-coefficient-reference-trained-readout-native-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

