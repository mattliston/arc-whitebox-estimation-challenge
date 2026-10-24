# Variance correction required fine fourth-order plane structure and self-consistent application

2026-09-04 · experiment · informational · measured

A HALF-FIXED fourth-cumulant triple-single plane (K31) INSIDE THE CONSUMER IS NOT ENOUGH (r35, engine-table third-cumulant repeated-index plane (K21) + engine fourth-cumulant two-pair plane (K22) inside, K31 := card + 0.5 (truth - card), beta 0.5, operator ON): 2.621e-8 (-3.1%) vs truth K31 -9.0% and half-SCALE truth K31 -7.5%. The consumer wants the truth K31's direction nearly whole. Residual carry (T-51 settings: arena push 30720 / read 28672, no forced drops) improves the card's k3 at depth by ~10-15% of its error at +26-39% util; testing it on the ek+feed card for the K21 supply.

Code availability: The shared retrofit instrument retains the fourth-order truth-forcing and plane-scale hooks. A dedicated launch script for the exact tracker comparison was not identified.

Recorded measurement:

    - T-87 THE VARIANCE ROUTE WORKS WITH ENGINE-CLASS SKEW PLANES (operator ON, independent 4M; deployed 2.704e-8 / 2.779e-8): the engine's K21 pre-planes x the frozen r35 scale table (K21 rel err 5.1 / 6.0 / 7.5% at L6/10/14 on r35, 5.4 / 6.0 / 8.2% on r100; shift alignment with the true K21 error 0.81-0.85 / 0.79-0.83) through the linear pair consumer (truth K22/K31 inside) -> dv aligned +0.86 / +0.88 / +0.90 (r35), +0.87 / +0.89 / +0.88 (r100) at L8/12/14 -> injected at beta 0.5: r35 2.417e-8 (-10.6%; beta 0.4 2.395e-8, -11.4%), r100 2.288e-8 (-17.7%; beta 0.4 2.286e-8). JOINT with the rescaled engine k3 shift at beta 0.35: r35 1.789e-8 (-33.8%, BELOW 2e-8), r100 2.345e-8 (-15.6%; the k3 part hurts r100). DEPLOYABLE-CLASS: the card's own K21 planes with engine-K22 fourth-order planes (ek card; K21 rel err 7.7 / 8.9 / 10.7%, shift alignment 0.44-0.60) through the same consumer -> dv aligned +0.59 / +0.72 / +0.74 -> beta 0.5: r35 2.443e-8 (-9.7%); the truth-fourth-order card's K21: 2.430e-8 (-10.1%). So on r35 the deployable-class supply already matches the engine's variance value; r100 and the consumer with deployable fourth-order planes inside are the two open checks.
    
    - T-87b r100 DEPLOYABLE-CLASS SUPPLY IS TOO WEAK (ek-card K21 planes on r100: rel err 8-11%; consumer dv alignment +0.63 / +0.72 / +0.69 at L8/12/14; operator ON): beta 0.5 2.744e-8 (-1.3%), beta 0.4 2.710e-8 (-2.5%) vs the engine-K21+table supply's -17.7% (alignment 0.87). The deployable route needs the card's K21 planes at engine-class accuracy (~5-6% at depth, from 8-11%): the gap = the engine's exact residual multi-push + F3/feed + the coherent table. Testing the consumer with deployable fourth-order planes inside, and the ek card + feed K21 planes.
    
    - T-87c THE CONSUMER NEEDS GOOD FOURTH-ORDER PLANES INSIDE (operator ON, beta 0.5, independent 4M): with the engine-table K21 supply, replacing the truth K22/K31 inside the consumer by ENGINE K22 + the CARD's K31: dv alignment 0.86-0.90 -> 0.80-0.86 (r35), 0.87-0.89 -> 0.81-0.85 (r100), magnitude 1.0 -> 1.2-1.3; in-card r35 2.829e-8 (+4.6%, from -10.6%), r100 2.593e-8 (-6.7%, from -17.7%); the ek-K21 supply with engine K22 inside: r35 2.817e-8 (+4%). Positive: the ek card + the derived feed has K21 rel err 7.3 / 8.2 / 9.6% (from 7.7 / 8.9 / 10.7) and its dv (truth fourth-order inside) reaches alignment 0.62 / 0.76 / 0.78 -> r35 2.346e-8 (-13.2%, from -9.7%). So the variance route = (K21 accuracy) x (fourth-order planes inside the consumer); testing which fourth-order plane the consumer needs (K22 vs K31) and whether the card's K31 with a frozen ratio table suffices.
    
    - T-87d THE CONSUMER'S BOTTLENECK IS THE K31 PLANE (r35, engine-table K21 supply, operator ON, beta 0.5): fourth-order planes inside the consumer = engine K22 + TRUTH K31 -> dv alignment 0.84-0.89, in-card 2.459e-8 (-9.0%, ~ the truth-both -10.6%); TRUTH K22 + the card's K31 -> 0.82-0.87 but 2.796e-8 (+3.4%); engine K22 + the card's K31 / frozen ratio table -> 0.77-0.85, 3.27e-8 (+21%; beta 0.35: +9%). So K22 can come from the engine chain (deployable) but K31 must be truth-class; the card's K31 (ratio 0.3-0.5, corr 0.55-0.8) is harmful inside the consumer and a scale table makes it worse. K31_pre is 95% inherited/distinct fourth-order content (the plane-supported push keeps 5%; the born block words are 1%) -> the fourth-order tensor carrier problem again, now with a precise target: the (3,1) pre-plane at truth class. Testing what accuracy K31 needs (half-scale truth K31; rank-8 / rank-32 truth K31) and the truth K31's rank structure.
    
    - T-87e K31 INSIDE THE CONSUMER NEEDS ITS FINE STRUCTURE (r35, engine-table K21 + engine K22 inside, operator ON, beta 0.5): truth K31 -9.0%; HALF-scale truth K31 2.500e-8 (-7.5%: scale is forgiving); rank-8 truth K31 (72% of its energy) 2.684e-8 (-0.8%); rank-32 (88%) 2.779e-8 (+2.8%); the card's K31 +3.4%..+21%. Truth K31 rank energy r1/r8/r32/r128 = 0.18/0.37/0.64/0.91 (L6), 0.55/0.72/0.88/0.97 (L14). Alignment at L8-14 (0.81-0.88 for all variants) does not predict these in-card values - the variance channel is sensitive to the plane corrections' CONSISTENCY across K21/K22/K31 (without a K31 correction the K21/K22-driven dv HURTS). The variance route therefore needs the (3,1) pre-plane at truth class including its high-rank part = the inherited distinct fourth-order content = the tensor wall. Stepping back to the corpus reviewers' reports before choosing the next thread.
    
    - T-87f THE CARD'S K31 IS THE DISCONNECTED SHAPE (r35): corr(card K31, v_u C_uw) = +0.98 / +0.99 / +0.99 at L6/10/14 while corr(truth K31, v C) = +0.67 / +0.68 / +0.74; the card's K31 correlates only 0.48-0.71 with the truth's rank-8 part and 0.27-0.49 with the rest; |missing|/|truth| 0.75-0.80. Like K22 (alpha x pairing plane), the card's (3,1) plane is essentially alpha x (v C): the connected fine structure the consumer needs is absent, not under-scaled. Checking whether the CP4 carrier's own (3,1) read (before the alpha assembly) carries that structure.
    
    - T-87g THE CARD HAS NO SOURCE OF THE FOURTH-ORDER FINE STRUCTURE (r35): the CP4 carrier's raw plane reads are 1.5-9% of the truth planes and their fine part (after removing the disconnected shape) correlates 0.00-0.14 with the truth's; the pair-series 'target' planes correlate 0.08-0.30 with the truth (fine part 0.01-0.03). The truth K31's fine part is 67-75% of the plane, K22's 17-27%. Single GEMM-class products of TRUTH third-order planes correlate with the K31 fine part at most +0.47 (k3_u K21[u,w]), +0.42 (m_u K21), +0.41 (k3 k3^T), +0.36 (K21^T K21, K21∘K21^T): the kappa3-squared (second-order Edgeworth) class is PRESENT in the fine structure but no single product dominates. Fitting a linear combination in-sample (r35) and checking it held-out (r100) to bound the kappa3^2-class ceiling before deriving it exactly.
    
    - T-87h KAPPA3-SQUARED CLASS CEILING (K31 fine structure regressed on 19 GEMM-class products of the truth third-order planes and states; fit on r35, applied held-out to r100): corr in-sample +0.43 / +0.55 / +0.65 at L6/10/14, held-out +0.39 / +0.56 / +0.57. So a second-order (kappa3^2) derivation could supply about half of the fourth-order planes' fine structure at depth; the other half is inherited distinct fourth-order content (the tensor wall). Not enough alone for the consumer (which needed the truth K31's fine structure nearly whole; rank-32 truth K31 already failed).
    
    - T-87i THIRD NET (r150, 8M forcing truth, operator ON): half-truth k3 2.621e-8 (-10.4%); half-truth v+k3 1.662e-8 (-43%). The goal-class configuration holds on three nets (r35 -68%, r100 -41%, r150 -43%); the skew channel alone is -32% / -5% / -10%.
    
    - T-87j FIXED DELTAS VS SELF-CONSISTENT MIX: injecting HALF of the (engine-table K21 minus base-card K21) as a fixed per-layer delta at layers 2-14 explodes the card (r35 7.27e-8, r100 5.3e-7; ek-card delta 6.07e-8 / 9.2e-8), although the alignment of the base-relative delta is 0.8. Cause: a delta computed against the BASE trajectory double-counts once earlier layers are already corrected (the corrected card's plane is closer to the supply than the base's), and the plane channel accumulates it. The truth 'mix' forcing that works (forced = current + 0.5 (supply - current)) is self-consistent; a deployable supply computed alongside the card is naturally of that form. New hook RETRO_K21_SUPPLY + RETRO_K21_MIX (pre_k21 := pre_k21 + mix (supply - pre_k21)). Re-testing the engine-table and ek-card K21 supplies in that form.
    
    - T-87k SELF-CONSISTENT HALF FORCING OF THE SKEW PLANE (RETRO_K21_SUPPLY, mix 0.5, layers 2-14; independent 4M): toward the engine-table K21: host 8.62e-8 -> 5.19e-8 (r35, -40%), 7.70e-8 -> 5.21e-8 (r100, -32%); toward the deployable ek-card K21: 6.12e-8 (-29%) / 5.94e-8 (-23%). With the operator ON: 2.90e-8 / 4.60e-8 (engine), 3.07e-8 / 4.58e-8 (ek) - worse than the deployed card. PLANES ARE IN-SPAN (the operator double-counts them), while the per-unit STATES v and k3 at half scale are perpendicular (T-80j). So the deployable supply must be delivered as per-unit (dv, dk3) at half scale in the self-consistent form; testing the engine-table k3 as a self-consistent 'supply truth' (RETRO_FORCE=k3 with the supply file, MIX 0.5) and the consumer dv as a supply v.
    
    - T-87l ENGINE-CLASS SUPPLIES, SELF-CONSISTENT HALF SCALE, OPERATOR ON (supply_truth_engine_rank*.npz: k3 = engine k3u x frozen r35 table; v = base card v + consumer dv (engine K21 + truth K22/K31 inside); RETRO_FORCE with MIX 0.5; independent 4M; deployed 2.704e-8 / 2.779e-8): r35 k3 only 2.347e-8 (-13.2%), v only 2.311e-8 (-14.5%), v+k3 1.802e-8 (-33.4%, BELOW 2e-8); r100 k3 only 3.415e-8 (+23%), v only 2.594e-8 (-6.7%), v+k3 3.820e-8 (+37%). The skew supply must be made harmless on r100-type nets (gentler scale, early-layer window); the variance supply helps both. The engine content is not deployable (full tensor) but the form is now exact: per-unit (v, k3) supplies computed alongside the card, applied as forced := current + mix (supply - current).
    
    - T-87m COMPOSITES (operator ON, independent 4M): variance supply (engine consumer dv) self-consistent mix 0.5 + engine-table skew delta beta 0.35 at layers 2-10: r35 1.850e-8 (-31.6%), r100 2.853e-8 (+2.7%); skew delta at layers 2-7: r35 2.097e-8 (-22.4%), r100 2.722e-8 (-2.1%). The two injection forms differ by net: for v the fixed delta at beta 0.5 gives r35 -10.6% / r100 -17.7% while the self-consistent mix gives -14.5% / -6.7%; for k3 the mix is better on r35 (-13.2% vs -11.1%) and both hurt r100. Still oracle-assisted (engine K21; truth K22/K31 inside the consumer).
    
    - T-87n COMPOSITE, FIXED-DELTA FORM (operator ON, independent 4M): variance delta (engine consumer dv, truth K22/K31 inside) at beta 0.5 + engine-table skew delta at beta 0.35 on layers 2-10: r35 2.052e-8 (-24.1%), r100 2.374e-8 (-14.6%) - both nets improve; r35 at the 2e-8 line. Oracle-assisted by (a) the engine's K21 planes (full tensor) and (b) the truth K22/K31 inside the consumer; the deployable analogs (card+engine-K22 planes for K21; engine K22 + card K31 inside) lose the value (T-87b/c). The deployable gap is now two named objects: the card's K21 accuracy at depth (7.7-10.7% vs the engine's 5-8% with the table) and the (3,1) fourth-order plane's fine structure inside the consumer.
    
    - T-87o A HALF-FIXED K31 INSIDE THE CONSUMER IS NOT ENOUGH (r35, engine-table K21 + engine K22 inside, K31 := card + 0.5 (truth - card), beta 0.5, operator ON): 2.621e-8 (-3.1%) vs truth K31 -9.0% and half-SCALE truth K31 -7.5%. The consumer wants the truth K31's direction nearly whole. Residual carry (T-51 settings: arena push 30720 / read 28672, no forced drops) improves the card's k3 at depth by ~10-15% of its error at +26-39% util; testing it on the ek+feed card for the K21 supply.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L731](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L731)
- [LANE_g3_tracker.md#L732](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L732)
- [LANE_g3_tracker.md#L733](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L733)
- [LANE_g3_tracker.md#L734](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L734)
- [LANE_g3_tracker.md#L735](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L735)
- [LANE_g3_tracker.md#L736](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L736)
- [LANE_g3_tracker.md#L737](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L737)
- [LANE_g3_tracker.md#L738](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L738)
- [LANE_g3_tracker.md#L739](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L739)
- [LANE_g3_tracker.md#L740](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L740)
- [LANE_g3_tracker.md#L741](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L741)
- [LANE_g3_tracker.md#L742](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L742)
- [LANE_g3_tracker.md#L743](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L743)
- [LANE_g3_tracker.md#L744](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L744)
- [LANE_g3_tracker.md#L745](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L745)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-variance-correction-required-fine-fourth-order-plane-structure-self) · [Copy provenance](../../../PROVENANCE.json)

