# Shared cross-network 65-feature final covariance tangent readout: extended/parent 1.0253, 0/5 folds, 2/20 nets improve

2026-09-20 · experiment · rejected · measured

One shared additive final-mean linear predictor built from the 65 formal directions of the final covariance response, fitted alongside the 65 affine-head columns with ridge 1e-4 in five 16-fit/4-held network folds on TRAIN20 with all upstream state fixed, failed all three gates; a scalar-only diagnostic with zero added cost still gives extended/parent 1.0001595620850054, so the negative is not an artefact of the provisional 10 GF allowance.

Recorded measurement:

    adjusted means: unchanged 3.9071715315547946e-9, affine control 3.893826867750025e-9, extended 4.005925865907333e-9; extended/parent 1.0252751468818266 (gate <=0.98), extended/control 1.0287889015009244 (gate <=0.99); 0/5 fold wins (gate >=4); 2/20 adjusted network wins; zero-cost diagnostic extended/parent 1.0001595620850054, extended/affine 1.0035922827723478, raw 1.0000785857261936 (10/20 raw wins); 19 parent calls 7573603518547 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_covariance_tangent_readout_v1/capture_all.py](code/terminal_covariance_tangent_readout_v1/capture_all.py)
- [code/terminal_covariance_tangent_readout_v1/information.py](code/terminal_covariance_tangent_readout_v1/information.py)
- [code/terminal_covariance_tangent_readout_v1/qualify.py](code/terminal_covariance_tangent_readout_v1/qualify.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/read5_packet_v1/candidate.py](code/read5_packet_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/terminal_covariance_information_v1/tangent.py](code/terminal_covariance_information_v1/tangent.py)
- [code/terminal_covariance_tangent_readout_v1/capture.py](code/terminal_covariance_tangent_readout_v1/capture.py)
- [code/terminal_covariance_tangent_readout_v1/common.py](code/terminal_covariance_tangent_readout_v1/common.py)
- [code/terminal_covariance_tangent_readout_v1/feature.py](code/terminal_covariance_tangent_readout_v1/feature.py)
- [code/terminal_covariance_tangent_readout_v1/fit_library.py](code/terminal_covariance_tangent_readout_v1/fit_library.py)
- [code/terminal_covariance_tangent_readout_v1/passive_source.py](code/terminal_covariance_tangent_readout_v1/passive_source.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/terminal_covariance_tangent_readout_v1/tangent.py](code/terminal_covariance_tangent_readout_v1/tangent.py)
- [code/terminal_representation_v1/capture.py](code/terminal_representation_v1/capture.py)
- [code/terminal_representation_v1/information.py](code/terminal_representation_v1/information.py)

</details>

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L6](NEGATIVE_RESULT.md#L6)
- [NEGATIVE_RESULT.md#L7](NEGATIVE_RESULT.md#L7)
- [NEGATIVE_RESULT.md#L13](NEGATIVE_RESULT.md#L13)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [COLLAPSED_READOUT_DESIGN.md](COLLAPSED_READOUT_DESIGN.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-shared-cross-network-65-feature-final-covariance-tangent-readout) · [Copy provenance](../../../PROVENANCE.json)

