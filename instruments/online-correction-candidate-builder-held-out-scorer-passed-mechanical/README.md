# Online correction candidate builder and held-out scorer passed mechanical validation

2026-09-04 · tool · informational · measured

whest validate PASSED on cand_closedloop_op1.py; score_candidate.py supports the six virgin nets (seeds 3016500-3016505, fresh_*_weights.npy / fresh_*_truth4M.npz); panel_candidate.sh scores a candidate on the 8-net panel + 6 virgin nets two at a time (~10 min). Deployed references: the 8-net deployed fitted transport-energy residual correction (F2J)-on numbers (inject_deployed_r*.log) and fresh_*_deployed.json.

Recorded measurement:

    - T-107 PACKAGING PATH CERTIFIED: build_closedloop_candidate.py (estimator_cap.py + the online operator step + in-loop feedback (mean + consistent covariance) + the payload spliced by the sha-pinned splicer, run with the torch interpreter) -> cand_closedloop_op1.py (687 kB, numpy-free); score_candidate.py (grader-style call, phase-2 budget, independent 4M): r35 final 3.2748e-8 at util 0.6003 = the instrument's closed-loop 3.2751e-8. Matt (23:50): PRE-AUTHORIZED to package and fire if the closed loop reaches adjusted < 1e-8 on the panel + virgin gate; no small banks; if it fails, do not give up (levers: state-error input channel, exposure-mixed rows, more iterations, differentiable card).
    
    - T-107b whest validate PASSED on cand_closedloop_op1.py; score_candidate.py supports the six virgin nets (seeds 3016500-3016505, fresh_*_weights.npy / fresh_*_truth4M.npz); panel_candidate.sh scores a candidate on the 8-net panel + 6 virgin nets two at a time (~10 min). Deployed references: the 8-net deployed F2J-on numbers (inject_deployed_r*.log) and fresh_*_deployed.json.

## Programs

- [code/public_g22_operator_features_v1/build_closedloop_candidate.py](code/public_g22_operator_features_v1/build_closedloop_candidate.py)
- [code/public_g22_operator_features_v1/panel_candidate.sh](code/public_g22_operator_features_v1/panel_candidate.sh)
- [code/public_g22_operator_features_v1/cand_closedloop_op1.py](code/public_g22_operator_features_v1/cand_closedloop_op1.py)
- [code/public_g22_operator_features_v1/score_candidate.py](code/public_g22_operator_features_v1/score_candidate.py)
- [code/public_d11_util_halving_v1/estimator_cap.py](code/public_d11_util_halving_v1/estimator_cap.py)
- [code/public_deploy_composition_prep_v1/splice_payload.py](code/public_deploy_composition_prep_v1/splice_payload.py)
- [code/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)

## Evidence

- [LANE_g3_tracker.md#L791](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L791)
- [LANE_g3_tracker.md#L792](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L792)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-04-online-correction-candidate-builder-held-out-scorer-passed-mechanical) · [Copy provenance](../../PROVENANCE.json)

