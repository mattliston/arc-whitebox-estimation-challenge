# Early oracle-tape residual fits were later invalidated by post-correction feature leakage

2026-09-04 · incident · failed · measured

The later tape audit invalidated the reported interior teacher-forced gains because the features were captured after the oracle correction. PER-LAYER DRIFT OF THE CLOSED LOOP (r35, one-step error under operator_1's own feedback vs the oracle upstream): L2 1.0e-9 vs 6.2e-10, L6 1.28e-8 vs 8.8e-9, L10 3.45e-8 vs 1.82e-8, L12 4.56e-8 vs 2.31e-8, L15 4.51e-8 vs 2.25e-8 - about 2x the oracle's at depth; operator_1's final correction (trained for the oracle upstream) then leaves 3.28e-8. Iteration 2 trains on exactly these rows (DAgger-style); expected to contract toward the oracle one-step errors.

Recorded measurement:

    - T-106 THE TEACHER-FORCED OPERATOR IS IN THE SUB-1e-8 CLASS (REFIT_MEANLEAK_RESULT.json: the F2J operator refit on the oracle-recursion rows (one-step host, oracle tape; frozen 160/40 protocol, 200 rows, 48 s): valid-40 final 8.93e-9 vs the one-step host 2.235e-8 (60% removal); all-200 in-sample 8.99e-9; independent truths: r35 one-step host 2.16e-8 -> 6.64e-9 (4M) / 8.65e-9 (32M); r100 2.13e-8 -> 9.69e-9 / 9.67e-9. At the deployed card's util 0.6065 that is adjusted ~5.4e-9 = the competitor's class, with NO new physics and ~zero added cost. CLOSED LOOP (its own corrections fed back, RETRO_CLOSED_LOOP=1 + payload_meanleak.npz): r35 3.28e-8, r100 3.73e-8 = exposure bias (trained on exact upstream). Cure: iteration 2 (rows from the closed loop of operator_1: gen_rows_closedloop.py -> refit) and/or damped feedback. Running both.
    
    - T-106b damping does not cure the exposure bias (r35, operator_1 closed loop): mix 0.75 3.72e-8, mix 0.5 4.76e-8, mean-only 3.60e-8 (full 3.28e-8). Iteration 2 rows generating (rows_closedloop: operator_1's own feedback, one-step host rows).
    
    - T-106c PER-LAYER DRIFT OF THE CLOSED LOOP (r35, one-step error under operator_1's own feedback vs the oracle upstream): L2 1.0e-9 vs 6.2e-10, L6 1.28e-8 vs 8.8e-9, L10 3.45e-8 vs 1.82e-8, L12 4.56e-8 vs 2.31e-8, L15 4.51e-8 vs 2.25e-8 - about 2x the oracle's at depth; operator_1's final correction (trained for the oracle upstream) then leaves 3.28e-8. Iteration 2 trains on exactly these rows (DAgger-style); expected to contract toward the oracle one-step errors.

## Programs

- [gen_rows_closedloop.py](gen_rows_closedloop.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/jext/run_experiment_ext.py](../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/jext/run_experiment_ext.py)
- [run_gen_closedloop.py](run_gen_closedloop.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L788](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L788)
- [LANE_g3_tracker.md#L789](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L789)
- [LANE_g3_tracker.md#L790](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L790)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [REFIT_MEANLEAK_RESULT.json](REFIT_MEANLEAK_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-early-oracle-tape-residual-fits-were-later-invalidated-post) · [Copy provenance](../../../PROVENANCE.json)

