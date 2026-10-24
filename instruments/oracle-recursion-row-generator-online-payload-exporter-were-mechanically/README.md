# Oracle-recursion row generator and online payload exporter were mechanically verified

2026-09-03 · tool · rejected · measured

PLUMBING CERTIFIED: export_refit_payload.py (state_dict -> payload npz) + RETRO_TRANSPORT_PAYLOAD + the online operator step (_cl_step) applied at the last layer only reproduce the refit runner's r35 number exactly (k22chain refit: 2.8805e-8 vs 2.8803e-8). The closed-loop pipeline is: rows_meanleak (generating) -> run_refit_ext.py G22_ROWS_DIR=rows_meanleak G22_TAG=MEANLEAK -> export -> RETRO_CLOSED_LOOP=1 with the payload (r35/r100, fitted transport-energy residual correction (F2J)-off row = the closed-loop result).

Recorded measurement:

    - T-105a ORACLE-RECURSION ROWS (gen_rows_meanleak.py: the organizers' 1e9 all-layer means fed back into the recursion, mean + consistent covariance; the row's host = the one-step mean before the correction; truthpost/truthpost_rank*.npz adapted from observer_fit.parquet, agreeing with our 32M truth to the noise floor): r35 one-step error per layer L2 6.2e-10, L4 4.1e-9, L8 1.33e-8, L12 2.31e-8, L14 2.22e-8, L15 2.25e-8 (open-loop chain host: 1.0e-9, 4.8e-9, 1.40e-8, 3.08e-8, 3.25e-8, 3.23e-8). The one-step (freshly born) error at the last layers is ~2.2e-8: the closed-loop route's ceiling is its predictability from the tape. 200 rows generating (29 s each); then the frozen-protocol refit on (one-step host, oracle tape) = the teacher-forced operator; then closed-loop inference with its own corrections fed back.
    
    - T-105b PLUMBING CERTIFIED: export_refit_payload.py (state_dict -> payload npz) + RETRO_TRANSPORT_PAYLOAD + the online operator step (_cl_step) applied at the last layer only reproduce the refit runner's r35 number exactly (k22chain refit: 2.8805e-8 vs 2.8803e-8). The closed-loop pipeline is: rows_meanleak (generating) -> run_refit_ext.py G22_ROWS_DIR=rows_meanleak G22_TAG=MEANLEAK -> export -> RETRO_CLOSED_LOOP=1 with the payload (r35/r100, F2J-off row = the closed-loop result).

## Programs

- [code/public_g22_operator_features_v1/gen_rows_meanleak.py](code/public_g22_operator_features_v1/gen_rows_meanleak.py)
- [code/public_g22_operator_features_v1/export_refit_payload.py](code/public_g22_operator_features_v1/export_refit_payload.py)
- [code/public_g22_operator_features_v1/run_refit_ext.py](code/public_g22_operator_features_v1/run_refit_ext.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/jext/run_experiment_ext.py](code/public_g22_operator_features_v1/jext/run_experiment_ext.py)
- [code/public_g22_operator_features_v1/run_gen_meanleak.py](code/public_g22_operator_features_v1/run_gen_meanleak.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L786](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L786)
- [LANE_g3_tracker.md#L787](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L787)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-03-oracle-recursion-row-generator-online-payload-exporter-were-mechanically) · [Copy provenance](../../PROVENANCE.json)

