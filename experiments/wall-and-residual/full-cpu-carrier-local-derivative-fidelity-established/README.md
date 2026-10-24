# Full CPU carrier: local derivative fidelity established

2026-09-05T20:12Z · experiment · informational · measured

The original large-step v1 fidelity flag remains FALSE and is preserved: the actual discrete-selection card is not globally differentiable. The explicitly diagnosed local branch, rather than a relaxed global threshold, passes the subsequent certificate on both ranks35/100. - Small-width full mean/C/third-cumulant repeated-index plane (K21)/radial maps match the NumPy reference at relative errors below7e-16. - Full CPU P0 mean trajectory maximum discrepancy1.70847e-7/2.23704e-7. - Actual dynamic finite-difference responses at layers4/9/13 agree with the CPU model within0.05% on both networks. - Large early perturbations cross selected memberships, causing7.4%/6.1% half-step changes; identical selected counts were not adequate evidence. - On the exact zero-point selection branch, finite-difference half-step discrepancies are0.316%/0.502%, and full-vector tangent discrepancies are0.307%/0.176% on rank35 and0.451%/0.923% on rank100. - Actual dynamic5e-6 perturbations stay in the same membership cell and agree with the tangent to2.47%/2.79%. Thus a CPU full-depth pilot is now justified, with the actual dynamic forward and local branch gradients.

Recorded measurement:

    The original large-step v1 fidelity flag remains FALSE and is preserved: the actual discrete-selection card is not globally differentiable. The explicitly diagnosed local branch, rather than a relaxed global threshold, passes the subsequent certificate on both ranks35/100. - Small-width full mean/C/K21/radial maps match the NumPy reference at relative errors below7e-16. - Full CPU P0 mean trajectory maximum discrepancy1.70847e-7/2.23704e-7. - Actual dynamic finite-difference responses at layers4/9/13 agree with the CPU model within0.05% on both networks. - Large early perturbations cross selected memberships, causing7.4%/6.1% half-step changes; identical selected counts were not adequate evidence. - On the exact zero-point selection branch, finite-difference half-step discrepancies are0.316%/0.502%, and full-vector tangent discrepancies are0.307%/0.176% on rank35 and0.451%/0.923% on rank100. - Actual dynamic5e-6 perturbations stay in the same membership cell and agree with the tangent to2.47%/2.79%. Remaining finite-difference noise is not hidden by treating the float32 reduction as exact real arithmetic. - Layer9 scalar reverse-mode checks agree to0.0624%/0.1423%. - The frozen scalar-final bank head was ported separately; final RMS discrepancy3.60e-8/4.49e-8, same maximum trajectory discrepancy as above. Thus a CPU full-depth pilot is now justified, with the actual dynamic forward and local branch gradients. This says nothing yet about learnable accuracy gain or global optimization. The all-float32 GPU version failed its baseline precision gate and is not used. Largest offline process RSS was11.47GB in the forward-mode diagnostic; reverse mode was7.70GB. Swap remained5966.50MiB; only one differentiating process at a time.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_full_carrier_bank_baseline_v1.py](code/check_full_carrier_bank_baseline_v1.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/probe_encoder_end_to_end_direction_v1.py](code/probe_encoder_end_to_end_direction_v1.py)
- [code/train_full_carrier_readout_pilot_v1.py](code/train_full_carrier_readout_pilot_v1.py)

## Evidence

- [FULL_CARRIER_AUTODIFF_V1_VERDICT.md#L1](FULL_CARRIER_AUTODIFF_V1_VERDICT.md#L1)
- [FULL_CARRIER_AUTODIFF_V1_VERDICT.md#L19](FULL_CARRIER_AUTODIFF_V1_VERDICT.md#L19)

<details>
<summary>All result files and records (3)</summary>

- [FULL_CARRIER_AUTODIFF_V1_VERDICT.md](FULL_CARRIER_AUTODIFF_V1_VERDICT.md) · record
- [full_carrier_bank_baseline_v1.json](full_carrier_bank_baseline_v1.json) · result
- [FULL_CARRIER_AUTODIFF_DESIGN_V1.md](FULL_CARRIER_AUTODIFF_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-full-cpu-carrier-local-derivative-fidelity-established) · [Copy provenance](../../../PROVENANCE.json)

