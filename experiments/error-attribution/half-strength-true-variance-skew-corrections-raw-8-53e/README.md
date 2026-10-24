# Half-strength true variance and skew corrections: raw 8.53e-9 and 1.650e-8 on two networks

2026-09-04 · experiment · informational · measured

T-80j: RETRO_FORCE=v,k3 with MIX 0.5 at every layer (forced = truth + 0.5 (card - truth)), deployed payload untouched, independent 4M: r35 2.704e-8 -> 8.53e-9 (-68%), r100 2.779e-8 -> 1.650e-8 (-41%); hosts fitted transport-energy residual correction (F2J)-off 4.8e-8 / 5.3e-8 (the operator does the rest). Partial beats full (full truth v+k3: -36% / +37%): the operator co-adapts. Singles at half scale: v -25% / -35% / -24% (r35 / r100 / r150), k3 -32% / -5%.

Recorded measurement:

    T-80j: RETRO_FORCE=v,k3 with MIX 0.5 at every layer (forced = truth + 0.5 (card - truth)), deployed payload untouched, independent 4M: r35 2.704e-8 -> 8.53e-9 (-68%), r100 2.779e-8 -> 1.650e-8 (-41%); hosts F2J-off 4.8e-8 / 5.3e-8 (the operator does the rest). Partial beats full (full truth v+k3: -36% / +37%): the operator co-adapts. Singles at half scale: v -25% / -35% / -24% (r35 / r100 / r150), k3 -32% / -5%. Supplies on record at half scale: the full-tensor engine's dv-hat + k3u shifts (corr 0.72-0.82 / 0.76-0.80) r35 2.073e-8 (-23%; k3 alone -11.8%, r100 k3 alone see T-80l); the truth-fourth-order + second-order engine's k3 shift (corr 0.85) -16.7%; the engine-K22-plane card's k3 shift (corr 0.74-0.79) -1.5%; ridge-from-tape dv (corr 0.25-0.45) WORSE. Value ~linear in alignment above ~0.7; off-direction components are punished as iid noise. Alignment levers (T-80k): fourth-order plane accuracy (card planes 0.2 -> engine planes 0.8; truth planes 0.835), the plane-to-distinct feed (+0.07), second-order Edgeworth (+0.02); the engine's ceiling with everything on record ~0.85. What is deployable today: engine-K22 planes + card K31 (k3 shift corr ~0.77, worth ~0). T-82..T-85b (Matt's bar: beat 2e-8 raw): the engine's shifts at half scale fail r100 (k3 +17%, dv+k3 +5%; alignment decays 0.82 -> 0.56 with depth); the engine-K22-plane card's shifts ~0 (early-layer alignment 0.25-0.7); partial plane fixes and partial k4 fixes with the operator collide (in-span). Truth skew planes through the linear pair consumer at half scale: 2.15e-8 / 2.06e-8 (needs ~3% skew planes in >= 32 directions; rank-8 keeps a quarter of the value); with the card's own fourth-order planes inside the consumer only -5.8%. No deployable supply meets the alignment bar. T-86..T-100d (score frame: adjusted = raw x util; a middling competitor is 1.854e-8 at util 0.218 = 4.04e-9): the engine's coherent under-scale table lifts skew alignment to 0.84-0.91; engine-class supplies in the self-consistent half-scale form reach r35 1.80e-8 but hurt r100 (oracle-assisted). The error is born LATE: exact per-unit (v,k3,k4) at L13-15 alone removes 49% of the F2J-off host error, L9-15 86%, L1-4 nothing; with the operator kept, half (v,k3) at L9-15 gives -52% (1.30e-8) and half v alone at L9-15 gives -20% / -29% on r35 / r100. The covariance residual at depth is low-rank and mean-aligned (top-eigen share 0.54-0.61, rank-16 0.81; |<u1,mu>| 0.8; diag of rank-16 E vs dv corr 0.90), Keenan's phase-1 structure - but writing the TRUE low-rank residual (full or diagonal, any scale or window) hurts or is neutral; only the exact per-unit v converts, and no computable direction carries a universal coefficient. A transported state corrector is dead (dv held-out 0.19 even with the oracle previous-layer error). Route left: a trained compensator write through a differentiable card (Keenan's converting mechanism), or a late-layer Tucker tensor engine.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L718](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L718)
- [inject_truth_vk3_mix0.5_r100.log#L1](inject_truth_vk3_mix0.5_r100.log#L1)
- [inject_truth_vk3_mix0.5_r35.log#L1](inject_truth_vk3_mix0.5_r35.log#L1)
- [inject_engine_dv_dk3_b0.5_r35.log#L1](inject_engine_dv_dk3_b0.5_r35.log#L1)
- [inject_dk3_engtab_b0.35_r100.log#L1](inject_dk3_engtab_b0.35_r100.log#L1)
- [inject_dk3_ekcard_b0.5_r35.log#L1](inject_dk3_ekcard_b0.5_r35.log#L1)
- [inject_dk3_engine_b0.5_L8_r35.log#L1](inject_dk3_engine_b0.5_L8_r35.log#L1)
- [inject_dk3_ek_b0.5_L8_r35.log#L1](inject_dk3_ek_b0.5_L8_r35.log#L1)
- [inject_dk3_engtab_b0.35_r35.log#L1](inject_dk3_engtab_b0.35_r35.log#L1)
- [inject_dk3_o4so2_b0.7_r35.log#L1](inject_dk3_o4so2_b0.7_r35.log#L1)
- [inject_dk3_engine_b0.5_r35.log#L3](inject_dk3_engine_b0.5_r35.log#L3)
- [inject_dk3_ek_b0.5_L5_r35.log#L1](inject_dk3_ek_b0.5_L5_r35.log#L1)
- [inject_dk3_o4so2_b0.5_r35.log#L1](inject_dk3_o4so2_b0.5_r35.log#L1)
- [inject_dk3_engine_b0.5_r100.log#L1](inject_dk3_engine_b0.5_r100.log#L1)
- [inject_dk3_engtab_b0.5_r35.log#L1](inject_dk3_engtab_b0.5_r35.log#L1)
- [inject_dk3_engtab_b0.5_r100.log#L1](inject_dk3_engtab_b0.5_r100.log#L1)

<details>
<summary>All result files and records (16)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [inject_truth_vk3_mix0.5_r100.log](inject_truth_vk3_mix0.5_r100.log) · log
- [inject_truth_vk3_mix0.5_r35.log](inject_truth_vk3_mix0.5_r35.log) · log
- [inject_engine_dv_dk3_b0.5_r35.log](inject_engine_dv_dk3_b0.5_r35.log) · log
- [inject_dk3_engtab_b0.35_r100.log](inject_dk3_engtab_b0.35_r100.log) · log
- [inject_dk3_ekcard_b0.5_r35.log](inject_dk3_ekcard_b0.5_r35.log) · log
- [inject_dk3_engine_b0.5_L8_r35.log](inject_dk3_engine_b0.5_L8_r35.log) · log
- [inject_dk3_ek_b0.5_L8_r35.log](inject_dk3_ek_b0.5_L8_r35.log) · log
- [inject_dk3_engtab_b0.35_r35.log](inject_dk3_engtab_b0.35_r35.log) · log
- [inject_dk3_o4so2_b0.7_r35.log](inject_dk3_o4so2_b0.7_r35.log) · log
- [inject_dk3_engine_b0.5_r35.log](inject_dk3_engine_b0.5_r35.log) · log
- [inject_dk3_ek_b0.5_L5_r35.log](inject_dk3_ek_b0.5_L5_r35.log) · log
- [inject_dk3_o4so2_b0.5_r35.log](inject_dk3_o4so2_b0.5_r35.log) · log
- [inject_dk3_engine_b0.5_r100.log](inject_dk3_engine_b0.5_r100.log) · log
- [inject_dk3_engtab_b0.5_r35.log](inject_dk3_engtab_b0.5_r35.log) · log
- [inject_dk3_engtab_b0.5_r100.log](inject_dk3_engtab_b0.5_r100.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-half-strength-true-variance-skew-corrections-raw-8-53e) · [Copy provenance](../../../PROVENANCE.json)

