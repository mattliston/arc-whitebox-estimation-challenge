# Third- and fourth-order rank cuts: rank-256 quartic carrier saved compute after co-adaptation

2026-09-04 · experiment · informational · measured

Third-cumulant rank reductions strongly degraded the host and its co-adapted correction, while reducing the fourth-cumulant carrier to rank 256 preserved useful accuracy after refitting. Removing additional cubic or quartic transport components exposed stability and candidate-parity failures; the complete measurements distinguish the unadapted, co-adapted and packaged configurations.

Recorded measurement:

    - T-118 CP3 CAP in the honest loop (r35, 8M frame, un-adapted head): cap 16384 (default) util 0.604 raw 2.49e-8; cap 8192 util 0.546 (-10%) raw 4.46e-8. D11 open-loop cap screen (CAP_SCREEN_*.json): util 0.554 -> 0.542 (8192) -> 0.482 (4096) -> 0.433 (2048) with raw +351% / x67 / x90: the third-order rank cap is NOT where the cost lives (the census shows per-layer cost rising linearly with depth, 7.7e10 at L5 -> 1.25e11 at L14; total 1.32e12 = 0.60). Running the per-source-line FLOP census (K_sitecensus.py) on the D11 card to locate the 0.6.
    
    - T-118 CP4 CAP in the honest loop (r35, un-adapted head; 1e9 raw, u_local): cap 2048 (default) 2.407e-8 @ 0.604 (raw x util 1.454e-8); cap 1024 2.499e-8 @ 0.567 (1.417, -2.6%); cap 512 2.594e-8 @ 0.547 (1.418, -2.5%). The fourth-order bank buys ~10% util per halving at ~4-8% raw un-adapted; with co-adaptation the raw may return to ~2.41 -> ~-9% adjusted. Bar in these units 0.94e-8 (-35%): CP4 alone is not enough; mapping the rest of the util curve (CP4 cap 256/128/0) and the site census to find the remaining cost.
    
    - T-118b CP4 CURVE (r35, 1e9 raw @ util, un-adapted head): 2048 2.407@0.604 (x=1.454); 1024 2.499@0.567 (1.417); 512 2.594@0.547 (1.418); 256 2.639@0.537 (1.417); 128 2.692@0.532 (1.432); off 2.839@0.527 (1.496). The whole fourth-order bank costs 0.077 util (13%) and is worth 18% raw un-adapted; the util FLOOR without it is 0.527 -> the bulk of the cost (0.53) is elsewhere (census running). Co-adaptation chain picks cap 1024/256 (raw x util 1.417) -> rows -> retrain -> loop.
    
    - T-118d CARD CENSUS (card_census.py, D11, r35, 1.319e12 = 0.600): _cp3_winograd_arena_depth4 44.9% (the third-order bank push, 7^4-block Winograd; the cap does not shrink it), _push_quartic 13.7%, _cp4_read_planes 13.0%, _predict 10.1% (covariance push), _push_cubic 9.1%, _cp4_push 4.7%, _g_four_all 3.0%. Third-order machinery 54%, fourth-order 32%, covariance/mean 10%. The arena is the Strassen-optimal dispatch (phase-1 laws) and compression of the bank was CLOSED WITH RECEIPTS (T-95..T-97b: billed low-rank costs more than it saves; youngest cohorts are full rank). So the util floor of this card family is ~0.5 (arena 0.27 + quartic/covariance/cubic) and the fourth-order bank is the only cheap 13%.
    
    - T-118e THE ONE UNTESTED BIG LEVER: the engine K22 chain is a DETERMINISTIC source fix that cut the F2J-off host 7.9e-8 -> 3.3e-8 (T-89); the honest mean head cuts the base host by 68%; if it cut the chain's host similarly -> ~1.06e-8 (below the bar at the card's util, before the chain's own cost). Needs co-adaptation (rows with the chain active; ~10-15 min per row). Running a 60-row version (ranks 35-54, 101-120, 168-187) -> fine-tune from HV_F1 -> loop with the chain on r35/r100 -> 1e9 ranking (chain_k22_small.sh).
    
    - T-118e COST ESTIMATE of the K22 chain if ported: fourth_push_np has ~8 einsums of 1024^3 class per layer (W2^T K22 W2, the 4-index W2/W/W contractions with K22/K31) -> ~2-4e10 flops per layer, ~3-5e11 over 15 layers = 0.15-0.25 of the budget, plus the emission series (columns up to tmax=16). Unless it REPLACES the card's CP4 bank (32% of the budget), the chain would push util to ~0.8, where the bar needs raw < 1.2e-8 (1e9-frame ~1.1e-8). Judge the co-adapted raw first (chain_k22_small), price second.
    
    - T-118c CAP-256 CO-ADAPTED (rows_meanleak_mech15_cap, GNN_HCAP256 valid 65.8%): r35 loop 2.512e-8 (8M) @ util 0.537 vs the full card 2.490e-8 @ 0.604 -> raw x util 1.348 vs 1.504 (8M): -10.4% adjusted. Real but small; the bar needs -35%. Next cost cut candidates: the quartic diagonal push (13.7%) and the cubic push (9.1%) with the head compensating through the weight-graph messages (needs card knobs).
    
    - T-118c CAP-256 CO-ADAPTED, both nets (1e9 raw @ util): r35 2.424e-8 @ 0.537 (raw x util 1.301 vs the full card's 1.454: -10.5%), r100 2.699e-8 @ 0.554 (1.495 vs 1.597: -6.4%). A deployable ~-8% adjusted (bank 1.78e-8 -> ~1.63e-8) = a small bank, NOT fireable under Matt's minimum (adj < 1e-8); stackable. T-118f RETRO_NO_QUARTIC (skip the dense quartic push; fourth-order pre objects from the CP4 bank only) un-adapted at cap 256: util 0.463 (-14%) but raw 2.77e-7 (11x) -> needs co-adaptation to judge (queued after the K22 small chain: rows with CP4_CAP=256 + RETRO_NO_QUARTIC=1). At util 0.46 the bar needs raw < ~1.9e-8 (1e9 frame).
    
    - T-118c CANDIDATE cand_gnn_hcap256.py (honest mean head co-adapted to CP4 cap 256, cap baked via BUILD_CP4_CAP=256): validated; grader frame r35 raw 2.653e-8, util 0.5367, adjusted 1.424e-8 (deployed r35 adjusted 1.623e-8: -12%). The best deployable card so far; a small bank by Matt's rule (minimum adj < 1e-8), held, not fired.
    
    - T-118e K22 CHAIN CO-ADAPTED (60 rows): on chain-active oracle rows the final one-step host is 8.83e-9 (the chain alone cuts the fresh final error 60% teacher-forced) but the co-adapted mean head removes only 17.5% of what remains (7.25e-9 vs 7.35e-9 without the chain): the chain fixes what the head was fixing (in-span, as T-89 found for the patch). LOOP (1e9): r35 2.390e-8 / r100 2.704e-8 vs the plain loop 2.407 / 2.584 -> no gain. DEAD.
    
    - T-118c CANDIDATE DISCREPANCY at cap 256: cand_gnn_hcap256.py r100 grader raw 5.13e-8 (instrument loop 2.82e-8 8M / 2.70e-8 1e9); cand_vs_instrument: the corrections match (corr 1.0000) but the card STATE diverges from L2 (input rel diffs up to 1e-2 by L8, growing) -> estimator_cap.py and estimator_retro.py handle a small CP4 cap differently. The instrument's cap numbers do not transfer until the cap logic is diffed and aligned (next).
    
    - T-118c CAP-256 DISCREPANCY DIAGNOSIS: the _cp4_* functions and the cap logic are identical between estimator_cap.py and estimator_retro.py (the retro's extra energy-ordered cap sits inside the inactive RETRO_CP4_STAR block). The candidate and the instrument still diverge in STATE from L2 at cap 256 (rel 6e-4 at L2 growing to 1e-2 by L8), i.e. the truncation's plane selection differs at the margin (dtype/ordering) and the small-cap card is chaotic to it: r35 agrees, r100 differs 2x. The cap-256 gain (-8% adjusted in the instrument) is NOT reproducible in the deployable card as built; parked (would need the selection made robust, e.g. an energy margin, before packaging).
    
    - T-118f NO-QUARTIC + CAP 256, CO-ADAPTED: the cheaper card's final one-step error is 7.375e-8 (3.3x the full card's 2.235e-8) but the head removes 86.6% of it (65-67% on the full card): teacher-forced 9.88e-9. LOOP (1e9 @ util): r35 3.319e-8 @ 0.463, r100 3.412e-8 @ 0.463 -> raw x util 1.537 / 1.579 vs the full card 1.454 / 1.597: a WASH in adjusted terms at a much cheaper card. LAW: the cheaper the card, the more predictable its error - the learned closure's removal fraction rises with the size of the systematic part. T-118g: the STACK (CP3 cap 2048 + CP4 cap 256 + no quartic; util ~0.3?) co-adapted - if the closure keeps ~90% removal, raw x util could fall toward the bar (chain_cp3low_coadapt.sh re-armed with the stacked env).
    
    - T-118h MINIMAL CARD PROBE (CP3 cap 1024 + CP4 off + no quartic + no cubic): util 0.1809 (BELOW the competitor's 0.22); un-adapted raw 9.6e-7 (meaningless before co-adaptation). At util 0.18 the bar needs raw < ~5.5e-8 grader (~5e-8 1e9 frame). If the closure keeps ~85-90% removal of the minimal card's (large, systematic) error, the bar is within reach. Chains: STACK (CP3 2048 + CP4 256 + no quartic) running -> then MINIMAL (chain_minimal_coadapt.sh). Builder knobs: BUILD_CP3_CAP, BUILD_CP4_CAP, BUILD_NO_QUARTIC, BUILD_NO_CUBIC. The minimal card's plane selection is trivial (CP4 off; CP3 keeps one layer of births) -> candidate reproduction should be robust.
    
    - T-118h UTIL FLOOR of the family with everything cheap: CP3 cap 1024 -> 0.181; cap 512 -> 0.1635; cap 256 -> 0.1636 (the remaining cost is the covariance push and fixed terms). So the cheapest card runs at util ~0.16-0.18; the bar there needs raw < ~5.5-6e-8 (grader). The co-adapted rounds (STACK at util ~0.3?, MINIMAL at 0.18) decide whether the closure keeps ~85-90% removal on such cards.
    
    - T-118g STACK (CP3 cap 2048 + CP4 256 + no quartic), co-adapted: the oracle one-step final host is 1.855e-5 (830x the full card): capping the THIRD-ORDER bank destroys the card (cf. the open-loop cap screen x90); the closure removes 87.8% teacher-forced (2.25e-6) but the loop is 3.09e-7 / 2.96e-7 @ util 0.278 -> raw x util 8.6e-8 = 6x WORSE than the full card. The third-order bank (the arena, 45% of the budget) is essential; the fourth-order machinery is what the closure can replace. Minimal (CP3 1024) round killed as pointless. NEXT: the LEAN card = full CP3 + CP4 off + no quartic + no cubic push (util probe, then co-adaptation).
    
    - T-118i LEAN PROBES (r35, un-adapted HNQ head, 8M): LEAN-2 (full CP3, CP4 off, no quartic, cubic kept) util 0.444, raw 3.53e-8; LEAN (also no cubic) util 0.389, raw 1.09e-7. Bar at 0.444 needs raw < ~2.25e-8 grader; at 0.389 < ~2.57e-8. Observation so far: raw x util stays ~1.45-1.55e-8 across cheap configurations (the closure recovers exactly the cost saved). Co-adaptation rounds for both lean variants launched (chain_lean.sh) with the gate.
    
    - T-118i LEAN-2 (full CP3, CP4 off, no quartic; cubic kept), co-adapted: teacher-forced 86.6% (final 9.28e-9 of a 6.94e-8 host); loop 1e9 r35 3.006e-8 @ 0.453, r100 3.381e-8 @ 0.462 -> raw x util 1.36 / 1.56 vs the full card 1.45 / 1.60 (-6% / -2.5%). The adjusted score is nearly INVARIANT across the cheap family: the closure recovers what each cost cut loses. LEAN (+ no cubic, util ~0.39) running; unless it breaks the tie, the cheap-card line tops out at ~-5% adjusted.
    
    - T-118i LEAN (+ no cubic push): the oracle one-step host explodes to 1.525e-5 (the cubic star push is as essential as the bank); loop r35 5.44e-8 @ 0.389 (raw x util 2.1, worse), r100 2.8e-6 (diverged). DEAD. FAMILY VERDICT: the learned closure replaces the FOURTH-order machinery (quartic push, CP4 bank) at a wash to -6% adjusted, and cannot replace any part of the THIRD-order carry (bank or cubic push). The adjusted score of this family is bounded near -6% below the deployed card; the bar (adjusted < 1e-8, -44%) is out of reach for the moment-closure card + learned closure paradigm. Best deployable: LEAN-2 co-adapted (CP4 off + no quartic; util 0.45-0.46; 1e9 raw x util 1.36 / 1.56 vs 1.45 / 1.60).
    
    - T-118i LEAN-2 CANDIDATE DOES NOT REPRODUCE THE INSTRUMENT: cand_gnn_lean2.py grader r35 1.33e-7 / r100 2.06e-7 (instrument loop 3.15e-8 / 3.51e-8 8M). Isolation: a NULL-payload candidate (heads zero) gives the plain host 3.44e-7 (grader) vs the instrument's plain host 3.11e-7 (8M) -> the CARD PHYSICS agrees under the lean flags; the divergence is in the closure's effect on the state: identical inputs and deltas at L1, but the L2 pre-state differs by 5e-3 rel in the candidate -> the applied correction changes the state differently in the two files only under CP4_CAP=1 / NO_QUARTIC (hook placement relative to the fourth-order context under these flags). OPEN; parked (the line is a small bank at best). All cheap-card numbers above are INSTRUMENT numbers.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/chain_lean.sh](code/public_g22_operator_features_v1/chain_lean.sh)
- [code/public_g22_operator_features_v1/chain_cp3low_coadapt.sh](code/public_g22_operator_features_v1/chain_cp3low_coadapt.sh)
- [code/public_g22_operator_features_v1/cand_gnn_hcap256.py](code/public_g22_operator_features_v1/cand_gnn_hcap256.py)
- [code/public_g22_operator_features_v1/chain_k22_small.sh](code/public_g22_operator_features_v1/chain_k22_small.sh)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/cand_gnn_lean2.py](code/public_g22_operator_features_v1/cand_gnn_lean2.py)
- [code/public_g22_operator_features_v1/card_census.py](code/public_g22_operator_features_v1/card_census.py)
- [code/public_g22_operator_features_v1/chain_minimal_coadapt.sh](code/public_g22_operator_features_v1/chain_minimal_coadapt.sh)
- [code/public_d11_util_halving_v1/estimator_cap.py](code/public_d11_util_halving_v1/estimator_cap.py)
- [code/public_g22_operator_features_v1/build_gnn_candidate.py](code/public_g22_operator_features_v1/build_gnn_candidate.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py)
- [code/public_g22_operator_features_v1/run_gen_meanleak_mech15_k22.py](code/public_g22_operator_features_v1/run_gen_meanleak_mech15_k22.py)
- [code/public_g22_operator_features_v1/score_candidate.py](code/public_g22_operator_features_v1/score_candidate.py)
- [code/public_g22_operator_features_v1/score_vs_truthpost.py](code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L880](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L880)
- [LANE_g3_tracker.md#L882](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L882)
- [LANE_g3_tracker.md#L883](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L883)
- [LANE_g3_tracker.md#L884](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L884)
- [LANE_g3_tracker.md#L885](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L885)
- [LANE_g3_tracker.md#L886](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L886)
- [LANE_g3_tracker.md#L887](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L887)
- [LANE_g3_tracker.md#L888](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L888)
- [LANE_g3_tracker.md#L889](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L889)
- [LANE_g3_tracker.md#L890](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L890)
- [LANE_g3_tracker.md#L891](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L891)
- [LANE_g3_tracker.md#L892](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L892)
- [LANE_g3_tracker.md#L893](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L893)
- [LANE_g3_tracker.md#L894](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L894)
- [LANE_g3_tracker.md#L895](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L895)
- [LANE_g3_tracker.md#L896](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L896)
- [LANE_g3_tracker.md#L897](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L897)
- [LANE_g3_tracker.md#L898](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L898)
- [LANE_g3_tracker.md#L899](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L899)
- [LANE_g3_tracker.md#L900](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L900)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-third-fourth-order-rank-cuts-rank-256-quartic-carrier) · [Copy provenance](../../../PROVENANCE.json)

