# Derived plane and residual-carry host reached raw 2.17e-8 at a larger compute bill

2026-09-03 · experiment · rejected · measured

(i) DESIGN A DEAD: the G28 ridge dk3 (leave-one-out (LOO), corr ~0.75) at HALF scale on layers 8-15 with the operator ON: r35 2.762e-8 (+2.1%), r100 3.226e-8 (+16%). (ii) THE DERIVED HOST'S KURTOSIS SHORTFALL IS A UNIVERSAL SCALE: truth-on-derived slope 1.08 (L2), 1.13 (fourth-cumulant Edgeworth operator (L4)), 1.14 (L6), 1.15 (L8), 1.15 (L10), 1.18 (L12), 1.23 (L14), corr 0.97-0.99, the same on r35 and r100 to +-0.03 (k4dump_derived_retro-*.npz). Zero-cost table for this host (k4_scale_derived_{full,half}.json, hook format s_l = derived/truth); testing on both nets. (iii) r150 lacks states_rank0150.npz (the derived-host chain needs it); deferred.

Recorded measurement:

    - T-88 A FULLY DERIVED HOST AT 2.17e-8 (r35, F2J OFF, independent 4M; inject_ekfeedcarry_r35.log): engine-K22 fourth-order planes (forced; the deployable plane chain) + the derived plane-to-distinct feed (RETRO_K21_INJECT) + the residual carry with the relaxed arena (RETRO_RESID_CP window 1-10, RETRO_ARENA_PUSH=30720 RETRO_ARENA_READ=28672 RETRO_READ_RANK_MAX=28672 RETRO_PUSH_RANK_MAX=30720 CP3_CAP=32768): host 8.62e-8 -> 2.170e-8 (-75%; below the deployed FULL card's 2.704e-8 by 20%) at util 0.835 (the carry is +22 pts; the plane-chain port would add ~+10); with the deployed operator ON it collapses to 4.55e-8 (in-span). Its K21 planes are engine-class: rel err 5.3 / 6.9 / 8.8% at L6/10/14 (engine+table 5.1 / 6.0 / 7.5; ek+feed 7.3 / 8.2 / 9.6); through the linear consumer (truth K22/K31 inside) the dv is aligned +0.84 at L8-14 and injected at beta 0.5 with the operator ON: 2.223e-8 (-17.8%; the engine-table supply gave -10.6%). So a DEPLOYABLE-CLASS skew-plane supply at engine accuracy exists (cost: carry +22 pts, port +10). Two candidates now: (A) the derived card without the operator (raw 2.17e-8 on r35; r100 pending; adj ~flat because of util 0.84-0.94); (B) the deployed card + consumer variance correction at half scale (needs truth K22/K31 inside the consumer: the remaining oracle piece).
    
    - T-88b r100 DERIVED HOST (engine-K22 planes + feed + relaxed-arena carry, F2J off, independent 4M): 7.70e-8 -> 2.941e-8 (-62%; 6% above the deployed full card 2.779e-8) at util 0.853; with the operator on 5.86e-8. (RETRO_RESID_CP_LAYERS is not the carry-window env: the 1-6 / 1-10 / 1-15 runs were identical - the default window was used; locating the right env.) Derived no-operator candidate (A): r35 2.17e-8 / r100 2.94e-8 at util ~0.85 before the plane-chain port (~+0.10): raw ~-7% vs the deployed pair offline, adjusted worse. Not the answer alone; a refit or partial operator on top is the open question.
    
    - T-88c THE DEPLOYED OPERATOR IS USELESS ON THE DERIVED HOST (partial scale scan, noise-corrected, independent 4M): r35 host 2.170e-8, + 0.10 x the operator's correction 2.141e-8 (-1.3%), + 0.30 2.261e-8; r100 2.941e-8 -> 2.893e-8 at 0.11, worse beyond. The operator's job (the fourth-order plane defect) is gone on this host; what remains is content its tape does not carry. Candidate (A) = the derived card alone: r35 2.14-2.17e-8, r100 2.89-2.94e-8, util ~0.85 (full-depth carry; RETRO_RESID_LAYERS is the window env) + ~0.10 for the plane-chain port. Testing cheaper carry windows.
    
    - T-88d CARRY WINDOWS ON THE DERIVED HOST (r35, F2J off; RETRO_RESID_LAYERS): 1-6 -> 2.427e-8 at util 0.759; 4-12 -> 2.423e-8 at 0.741; 1-10 -> 2.175e-8 at 0.816; full 1-15 -> 2.170e-8 at 0.835. Window 1-10 keeps the full value at -2 pts. Candidate (A) stands at r35 2.18e-8 / util 0.82 (+~0.10 for the plane-chain port).
    
    - T-88e THE DERIVED HOST IS A CLEAN BASE FOR STATE CORRECTIONS (r100, window 1-10 carry, F2J off, independent 4M): derived host 3.001e-8 (util 0.834); + half-truth v 2.110e-8 (-30%); + half-truth k3 2.294e-8 (-24%: the skew channel HELPS r100 here, unlike with the operator); + half-truth v,k3,k4 8.92e-9 (-70%). With the operator ON every arm is 4.6-5.9e-8 (in-span collapse). => route (A'): derived host (engine-K22 plane chain + feed + carry) + engine-class per-unit state supplies at half scale, no learned operator. Grid running: k4 table, engine-table k3 (mix 0.5), consumer dv (delta 0.5), half-truth k4 ceiling, on both nets.
    
    - T-88f COST CENSUS OF THE DERIVED CARD (estimate, 0.098% util per 1024^3 f32 GEMM-eq; measured pieces marked): base card 0.600 (measured); residual carry window 1-10 +0.216 (measured: 0.816); engine-K22 plane chain port: emission M11/M21/M22 Mehler series (elementwise, ~0.5 GEMM-eq/layer) + push (W2^T diag(k4) W2, W2^T K22 W2, two K31-mixed terms, k4n) ~9 GEMMs/layer -> ~+0.12; the CP4 machinery it replaces -0.11 (census); feed k3-read only 3 GEMMs/layer +0.04 (the full plane read +0.12); f32 covariance chain -0.03. Feasible total ~0.83-0.90 (with the feed's full plane read ~0.95). Adjusted-score break-even vs the bank (1.78e-8) at util 0.85: panel raw <= 2.1e-8. r150 derived-host chain launched (engine ref planes, feed plane read, carry).
    
    - T-88g DERIVED-HOST STATE-SUPPLY GRID (F2J off, independent 4M; derived host r35 2.175e-8 / r100 3.001e-8): frozen k4 table 1.56e-7 / 2.17e-7 (explodes: the derived host's k4 is no longer 40-60% under); engine-table k3 at mix 0.5 1.864e-8 (-14%) / 2.940e-8 (-2%); the old consumer dv as a delta 2.55e-8 (+17%) / 3.49e-8 (+16%: computed against the old card's planes, does not fit this host); half-truth k4 1.755e-8 (-19%, r35). CORPUS REVIEWERS (three reports, 2026-09-03): (1) every refit verdict on record compared a 160-row refit with the 688-row deployed payload (6-8% protocol gap) - no refit with the deployed recipe was ever run on a corrected host; (2) half-scale tests with the operator kept were never run on the G28 ridge dk3 predictor, the cum4 feed, the no-drop carry or any refit; (3) SOL has produced nothing for phase 2 since 08-09; the fourth-order derivation lanes are bounded by T-85/T-80k (do not fund); (4) transported quantities predict what local ones cannot (engine dv-hat lifts held-out dv R^2 0.11 -> 0.44; engine k3u lifts dk3 to 0.80-0.85 at L8-13). Consequence for the derived host: its remaining error is per-unit state content (half-truth v/k3/k4 -30/-24/-19%); the deployed operator is useless on it; a 688-recipe refit on THIS host is the unmeasured big lever, and needs the in-card K22 plane chain (port) to make rows.
    
    - T-88h (i) DESIGN A DEAD: the G28 ridge dk3 (LOO, corr ~0.75) at HALF scale on layers 8-15 with the operator ON: r35 2.762e-8 (+2.1%), r100 3.226e-8 (+16%). (ii) THE DERIVED HOST'S KURTOSIS SHORTFALL IS A UNIVERSAL SCALE: truth-on-derived slope 1.08 (L2), 1.13 (L4), 1.14 (L6), 1.15 (L8), 1.15 (L10), 1.18 (L12), 1.23 (L14), corr 0.97-0.99, the same on r35 and r100 to +-0.03 (k4dump_derived_retro-*.npz). Zero-cost table for this host (k4_scale_derived_{full,half}.json, hook format s_l = derived/truth); testing on both nets. (iii) r150 lacks states_rank0150.npz (the derived-host chain needs it); deferred.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L746](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L746)
- [LANE_g3_tracker.md#L747](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L747)
- [LANE_g3_tracker.md#L748](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L748)
- [LANE_g3_tracker.md#L749](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L749)
- [LANE_g3_tracker.md#L750](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L750)
- [LANE_g3_tracker.md#L751](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L751)
- [LANE_g3_tracker.md#L752](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L752)
- [LANE_g3_tracker.md#L753](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L753)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [inject_ekfeedcarry_r35.log](inject_ekfeedcarry_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-03-derived-plane-residual-carry-host-reached-raw-2-17e) · [Copy provenance](../../../PROVENANCE.json)

