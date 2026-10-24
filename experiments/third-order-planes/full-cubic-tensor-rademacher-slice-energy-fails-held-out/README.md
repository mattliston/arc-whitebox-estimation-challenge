# Full cubic-tensor Rademacher slice energy fails held-out cumulant-error prediction: ratios .9729/.9934

2026-09-20 · experiment · rejected · measured

Sixteen fixed bilateral Rademacher contractions of the complete rounded CP+star+trace cubic tensor (layers 8 to 15, all cross terms, full bank) were squared into one scalar per coordinate and added as 3 extra columns to a 24-coefficient ridge (1e-3) regression that predicts third and fourth cumulant errors weighted by their marginal mean-response weights, fitted on six TRAIN nets. On the two CHECK networks the extension reduces the loss by only 2.7% against zero correction and 0.66% against the matched control, failing the information gates; the producer itself was qualified at an added bill of 4.70 to 5.06%.

Recorded measurement:

    CHECK ratios loss/zero 0.976335421 (35), 0.968975438 (100); loss/control 0.992143527 / 0.994880545; pooled 0.972870544386 and 0.993425002862 vs <=0.85 / <=0.95; TRAIN 0.966595384229 / 0.994835037719; added FLOPs 4.700622% to 5.063013%; worst contraction relative RMS 6.70339824e-7

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_cubic_slice_energy_v1/capture.py](code/full_cubic_slice_energy_v1/capture.py)
- [code/full_cubic_slice_energy_v1/probes.py](code/full_cubic_slice_energy_v1/probes.py)
- [code/full_cubic_slice_energy_v1/information.py](code/full_cubic_slice_energy_v1/information.py)
- [code/full_cubic_slice_energy_v1/unit_v2.py](code/full_cubic_slice_energy_v1/unit_v2.py)
- [code/full_cubic_slice_energy_v1/finalize.py](code/full_cubic_slice_energy_v1/finalize.py)
- [code/full_cubic_slice_energy_v1/common.py](code/full_cubic_slice_energy_v1/common.py)
- [code/full_cubic_slice_energy_v1/feature_fnp.py](code/full_cubic_slice_energy_v1/feature_fnp.py)
- [code/full_cubic_slice_energy_v1/reference.py](code/full_cubic_slice_energy_v1/reference.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L19](RESULT.md#L19)
- [RESULT.md#L28](RESULT.md#L28)
- [RESULT.md#L29](RESULT.md#L29)
- [RESULT.md#L31](RESULT.md#L31)
- [RESULT.md#L32](RESULT.md#L32)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [LEARNING_EQUATIONS.md](LEARNING_EQUATIONS.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-full-cubic-tensor-rademacher-slice-energy-fails-held-out) · [Copy provenance](../../../PROVENANCE.json)

