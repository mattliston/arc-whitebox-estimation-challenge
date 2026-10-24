# Weight-graph recurrent closure: early interior gains later traced to target leakage

2026-09-04 · incident · failed · measured

The later tape audit invalidated the reported interior teacher-forced gains because the features were captured after the oracle correction. GNN CANDIDATE CERTIFIED (build_gnn_candidate.py: estimator_cap.py + _mech_fnp + _gnn_step copied from the instrument + in-loop hooks + the payload literal; 1.63 MB, numpy-free): score_candidate.py r35 1.20517e-7 at util 0.6121 vs the instrument's 1.20518e-7 (same partial model; the features + step add ~1.2 pts of util); wall 41 s. The packaging path for any learned-closure payload is one command.

Recorded measurement:

    - T-114 LEARNED CLOSURE OVER THE WEIGHT GRAPH (Matt: focus 100%): train_gnn_oracle.py - a recurrent message-passing model across layers: node state e^l = MLP([tape20, mech31, gate-weighted messages W^T g1(e^{l-1}), (W∘2)^T g2, (W∘3)^T g3 and their energies]); delta_i = sigma_i rms_l head_l(e_i); J-style loss (per-layer normalized MSE, linspace weights + final term); targets = the one-step error of the oracle recursion (organizers' 1e9 means minus the one-step host); same 160/40 split as the refits (valid-40 reference: 20-feature refit 8.93e-9, 60% removal). First run on the rows available (rows_meanleak_mech partial), 10 epochs, 32 channels, MPS.
    
    - T-114a FIRST LEARNED-CLOSURE RESULT (GNN_V1_PARTIAL: 75 train / 26 valid nets of the protocol split, 51 inputs, 32 channels, 10 epochs, 69 s on MPS): valid final 8.51e-9 vs the one-step host 2.30e-8 = 63.0% removal (tape refit: 60.0%, 8.93e-9); interior removal L8 78%, L12 79%, L14 78% (tape refit ~60%); train final 8.43e-9 = valid (no overfit; still improving at epoch 10). The final layer lags the interior because its mechanistic inputs are zero (no feature dump at the final read) - and it is the scored layer. Next: final-layer features (dump at the final branch + mechfeat row 15), full 200-row training with more epochs/channels, then parity (gnn_parity.py) and the closed loop with RETRO_GNN_PAYLOAD.
    
    - T-114b FINAL-LAYER FEATURES: the retro now dumps the final read layer's per-unit state (pre m/v, final k3 split into the CP3 and star parts, final k4 split into star and CP4 parts, w_in_norm; planes absent -> zeros); mechfeat15.py covers rows 1..15 with zero defaults; gen_rows_meanleak_mech15.py -> rows_meanleak_mech15 (r35: 19 of 31 final-layer features nonzero; rows 1-14 and the host bit-identical to rows_meanleak_mech). Generating (2 workers on 35-134 alongside the 4 workers finishing rows_meanleak_mech); GNN 64-channel / 30-epoch run on the 125 available rows in parallel on MPS.
    
    - T-114c GNN IN-LOOP STEP CERTIFIED (gnn_parity.py, two stages): fnp _gnn_step vs the torch forward max |diff|/rms 5.6e-7 (L2) .. 1.9e-6 (L15); the 16 steps bill 4.14e9 flops = 0.19% of the budget. Exporter export_gnn_payload.py.
    
    - T-114d GNN 64 CHANNELS / 30 EPOCHS (97 train / 28 valid of the protocol split, rows without final-layer features): valid final 7.10e-9 (68.9% removal; tape refit 60%); INTERIOR removal L8 97% / L12 97% / L14 97% (tape refit ~60%; the partial 32-channel run 78%). The freshly born interior error is almost fully predictable from weight-graph messages + mechanistic inputs; the final layer lags only because its inputs are missing (rows_meanleak_mech15 generating). BUT the partial model's own closed loop is bad: r35 5.58e-8, r100 4.00e-8 (operator_1's loop 3.28 / 3.73e-8) - severe exposure bias; diagnosing (own-loop profile, teacher-forcing ladder).
    
    - T-114e GNN LOOP DIAGNOSTICS (partial model, r35): own-loop carried profile L4 4.4e-9 / L8 1.63e-8 / L10 3.03e-8 / L12 4.57e-8 / L15 4.72e-8 (operator_1's loop: 4.1e-9 / 1.47e-8 / 2.26e-8 / 3.27e-8 / 3.16e-8) - the drift starts at L10; ladder oracle-through-L12 + GNN at 13-15: 2.62e-8; damping does not help (mix 0.7 5.91e-8, 0.5 6.41e-8, full 5.58e-8). With 97% teacher-forced interior accuracy the loop's error is off-distribution inputs (drifted host tape + mechanistic features), not unpredictability -> scheduled sampling for the GNN (rows from its own loop blended with truth, alpha 0.5, mech15 features; fine-tune), plus the final-layer features.
    
    - T-114f GNN SCHEDULED-SAMPLING ROWS WORK (gen_rows_gnnmixed.py: the GNN's own feedback blended with truth, alpha 0.5, mech15 features; in-loop features now via mechfeat15 at every layer incl. the final read): r35 one-step error L4 4.4e-9 / L8 1.43e-8 / L12 2.52e-8 / L15 2.53e-8 (oracle 4.1e-9 / 1.33e-8 / 2.31e-8 / 2.25e-8); 19 final-layer features nonzero. Plan: train the GNN on rows_meanleak_mech15 (oracle, with final-layer features) -> fine-tune on rows_gnnmixed (alpha 0.5, then 0.25) -> closed loop.
    
    - T-114g MACHINE INCIDENT: eight row workers + the MPS training + a diagnostic card run pushed swap to 23 GB of 26.6 GB (memory 34% free); rows stalled (workers at 15-40% CPU, paged out) for ~30 min. Stopped the rows_meanleak_mech drivers at 149/200 (its consumer, the hand-designed 51-input F2J refit, is superseded by the GNN's 97% interior removal), the ladder diagnostic and the armed baseline refit. Now: 4 workers on rows_meanleak_mech15 (35-134 at rank ~48, 135-234 at ~147) at full speed + the GNN 64ch/40ep training on the 149 rows as the fine-tuning init. Rule kept: never more than 4 card workers alongside an MPS job; check vm.swapusage before adding load.
    
    - T-114h THE 31 MECHANISTIC FEATURES IN FLOPSCOPE (_mech_fnp in estimator_retro.py, RETRO_CL_MECH_FNP=1; the packaging form): parity vs mechfeat15 max abs diff 0.00 at every layer after fixing a stash aliasing bug (the card reuses buffers in place; the previous-layer stash must copy). The partial model's loop with real final-layer features is 1.2e-7 (it was trained with zero features at the final layer) - a model/feature mismatch, resolved by training on rows_meanleak_mech15.
    
    - T-114i FNP FEATURE PATH CERTIFIED at all 16 layers (max abs diff 0.00 vs mechfeat15) and the closed loop with fnp features equals the numpy-feature loop to the last digit (1.2052e-7 with the partial model, whose final-layer features were zeros in training). The packaging path for the learned closure is therefore: _mech_fnp + _gnn_step (both fnp, certified) + the GNN payload as a literal.
    
    - T-114j GNN CANDIDATE CERTIFIED (build_gnn_candidate.py: estimator_cap.py + _mech_fnp + _gnn_step copied from the instrument + in-loop hooks + the payload literal; 1.63 MB, numpy-free): score_candidate.py r35 1.20517e-7 at util 0.6121 vs the instrument's 1.20518e-7 (same partial model; the features + step add ~1.2 pts of util); wall 41 s. The packaging path for any learned-closure payload is one command.

## Programs

- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py)
- [../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py](../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [../2026-09-04-candidate-parity-bug-exported-variance-head-active-although-instrument/build_gnn_candidate.py](../2026-09-04-candidate-parity-bug-exported-variance-head-active-although-instrument/build_gnn_candidate.py)
- [../../../experiments/flop-accounting/gaussian-closure-training-required-covariance-repair-still-removed-only/code/public_g22_operator_features_v1/gnn_parity.py](../../../experiments/flop-accounting/gaussian-closure-training-required-covariance-repair-still-removed-only/code/public_g22_operator_features_v1/gnn_parity.py)
- [../../../experiments/error-attribution/closed-loop-failure-diagnostics-separated-teacher-forcing-depth-per/code/public_g22_operator_features_v1/score_candidate.py](../../../experiments/error-attribution/closed-loop-failure-diagnostics-separated-teacher-forcing-depth-per/code/public_g22_operator_features_v1/score_candidate.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_gnnmixed.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/gen_rows_gnnmixed.py)
- [../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py](../../../experiments/arithmetic-kernels/third-cumulant-rank-cuts-sharply-worsened-raw-error-while/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_d11_util_halving_v1/estimator_cap.py)
- [../../../experiments/arithmetic-kernels/third-fourth-order-rank-cuts-rank-256-quartic-carrier/code/public_g22_operator_features_v1/cand_gnn_hcap256.py](../../../experiments/arithmetic-kernels/third-fourth-order-rank-cuts-rank-256-quartic-carrier/code/public_g22_operator_features_v1/cand_gnn_hcap256.py)
- [../../../experiments/fourth-order-terms/float32-covariance-transport-saved-5-35-4-adjusted-14/code/cand_gnn_hcap256_f32cov.py](../../../experiments/fourth-order-terms/float32-covariance-transport-saved-5-35-4-adjusted-14/code/cand_gnn_hcap256_f32cov.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [../../../submissions/329754/estimator.py](../../../submissions/329754/estimator.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/cand_gnn_hf1.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/cand_gnn_hf1.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/cand_gnn_hf1_v2.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/cand_gnn_hf1_v2.py)
- [../../../experiments/flop-accounting/removing-dense-quartic-transport-after-co-adaptation-traded-compute/code/public_g22_operator_features_v1/cand_gnn_hlean256.py](../../../experiments/flop-accounting/removing-dense-quartic-transport-after-co-adaptation-traded-compute/code/public_g22_operator_features_v1/cand_gnn_hlean256.py)
- [cand_gnn_hnq_v2.py](cand_gnn_hnq_v2.py)
- [../../../experiments/arithmetic-kernels/third-fourth-order-rank-cuts-rank-256-quartic-carrier/code/public_g22_operator_features_v1/cand_gnn_lean2.py](../../../experiments/arithmetic-kernels/third-fourth-order-rank-cuts-rank-256-quartic-carrier/code/public_g22_operator_features_v1/cand_gnn_lean2.py)
- [cand_gnn_lean2_dbg.py](cand_gnn_lean2_dbg.py)
- [cand_gnn_lean2_v2.py](cand_gnn_lean2_v2.py)
- [cand_gnn_v1p.py](cand_gnn_v1p.py)
- [cand_null_lean2.py](cand_null_lean2.py)
- [../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_force14_test.py](../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_force14_test.py)
- [../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_meanleak_mech.py](../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_meanleak_mech.py)
- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [../../../experiments/error-attribution/fourth-order-state-changes-broke-learned-mean-head-unless/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py](../../../experiments/error-attribution/fourth-order-state-changes-broke-learned-mean-head-unless/code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py)
- [../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_mech.py](../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g22_operator_features_v1/gen_rows_mech.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/run_gen_gnnmixed.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/run_gen_gnnmixed.py)
- [../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/run_gen_meanleak_mech15.py](../../../experiments/flop-accounting/honest-learned-closure-variance-heads-noise-tests-separated-information/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/run_gen_meanleak_mech15.py)
- [../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g28_mech_features_v1/mechfeat.py](../../../experiments/error-attribution/second-stage-correction-removed-only-8-0-closed-loop/code/public_g28_mech_features_v1/mechfeat.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L806](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L806)
- [LANE_g3_tracker.md#L807](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L807)
- [LANE_g3_tracker.md#L808](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L808)
- [LANE_g3_tracker.md#L809](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L809)
- [LANE_g3_tracker.md#L810](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L810)
- [LANE_g3_tracker.md#L811](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L811)
- [LANE_g3_tracker.md#L812](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L812)
- [LANE_g3_tracker.md#L813](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L813)
- [LANE_g3_tracker.md#L814](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L814)
- [LANE_g3_tracker.md#L815](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L815)
- [LANE_g3_tracker.md#L816](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L816)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-weight-graph-recurrent-closure-early-interior-gains-later-traced) · [Copy provenance](../../../PROVENANCE.json)

