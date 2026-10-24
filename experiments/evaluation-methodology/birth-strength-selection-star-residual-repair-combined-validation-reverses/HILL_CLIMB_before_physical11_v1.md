# Current Phase 2 hill-climb state

Updated 2026-09-28T06:34:38.683148+00:00. This file is authoritative; prior state snapshots are historical.

All numerical sessions are retired and the completed aggregate/tree guards show zero added swap. No serving, upload, grading, or approval action is pending. The goal remains active and unmet. Current work is a bounded source-only Opus/Fable hill-climb audit; no new candidate build or numerical run is admitted. Derivative/JVP V11 remains parked.

## Official bank and standing policy

S32 **#332738** remains the official bank: adjusted **2.1399773263156476e-9**, graded 100/100 with zero failures. Source `hill_climb_star_projection32_v1/candidate.py`, SHA `f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`; receipt `official_bank_S32_v1.json`. Do not reupload.

User policy remains: bank fully validated aggregate gains of at least 1%; submissions are authorized. Smaller sound components may accumulate. The approximately 1e-9 goal and 2e-9 milestone remain unmet.

## Completed aggregate: broad validation failed

`hill_climb_bds_csr_exact_aggregate_v1/candidate.py`, SHA `23a433b250046b7a6dd9b30c279ccb1e5bc7cc9e4dd63b6d68153c1b60480538`, was one fixed BDS + CSR + exact-cost conjunction.

- TRAIN16 passed: adjusted -0.753127190%, 13/16 wins, CI [-1.335730808%, -0.137624259%]. This training result is preserved.
- VALID40 failed: adjusted **+0.164462482%**, 17/40 wins, CI **[-0.261396856%, +0.602996926%]**; raw +0.244117154%, bill -0.071000214%, CPU -0.804473622%. Maximum adjusted regression was 4.435014913%.
- Receipt integrity passed; adjusted gain, win count, and confidence gates failed. All four native guards retired with zero added swap; maximum broad RSS was 6,339,788,800 bytes. Validation was historically reused, not pristine.
- The source hypothesis is closed. No subsets, coefficients, rescue, or further aggregate evaluation follows. Original standalone BDS/CSR failed-confidence decisions remain unchanged.
- Serving runner/config/assessor are staged and source-reviewed but **unexecuted**. No `validation_admission.json`, `FIRE_CHECKS_v1.json`, packaging, or upload exists for this aggregate.

Evidence: study `train16_assessment.json`, `broad_assessment.json`, `independent_broad_result_review.json`, and source reviews. The independent result review checked 41 non-array pins and scalar arithmetic without reopening predictions or rerunning bootstrap.

## Completed Gaussian-tree birth13: native screen failed

`hill_climb_gaussian_tree_birth13_v1/candidate.py`, SHA `25adf9d26af8c180c2a9189f427049ff314d2fa0630f656a398e2f2da4f63339`, failed TRAIN2: adjusted **+1.574494744%**, 0/2 wins, raw +0.073238519%, bill +1.499867797%.

Independent tiny tree fixtures passed; native layers 0–12 were byte-exact and outputs finite. Unit/native guards retired with zero added swap; native peak RSS was 5,316,575,232 bytes. No follow-up or variant. Historical checkpoint33 local-positive evidence remains distinct from this failed current endpoint. See study `train2_assessment.json`, `RESULT.md`, and `independent_source_review.json`.

## Retained exact bundle: small component, no release qualification

`hill_climb_exact_bundle_v1/candidate.py`, SHA `3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7`, combines unchanged UVP B-cache + nine D3 bindings + 27 extra bindings.

Native2, worker2, boundary2, and setup3 passed: all 16 output bytes exact versus S32, native/RPC bills equal, mean bill -0.390611902%. Setup has 660 bound / 636 fused entries, maximum 3.26786 seconds; serving peak RSS 5.916934 GB, 49,141 handles, zero added swap. Boundary CPU ratio 0.9922638 and residual changes -0.0025/-0.0034 seconds passed. Cache lifetime is bounded and retired after prediction.

This bundle is retained for future composition only. It is below the 1% bank bar and has no worker20 or broad qualification. The failed BDS/CSR aggregate does not revoke this separately measured exact component. Evidence: `retained_bundle.json`, `probe_assessment.json`, `independent_probe_review.json`.

## Current source-only audit and precision conclusion

Continue the bounded Opus/Fable process audit for one evidence-backed next step. Current S32 FP64 charge decomposition accounts for 6.088479 GF (2.201586% of the rank35 bill). That total includes mandatory scalar/reduction/cast work and is not an achievable saving. Keeping closed BD and sensitive Nyström paths fixed leaves only generous gross envelopes of 0.733263%–0.874551%, before required casts and accuracy effects. The current precision ceiling is not a material actionable proposal; **no precision candidate build or execution is admitted**. See `teammate_s32_fp64_consumer_audit_v1.md` and `teammate_s32_fp64_charge_decomposition_v1.json`. Do not turn an upper bound into a gain claim or revive the failed BD precision screen.

Current key-census append forecast remains rejected (0.08547732%); the distinct capacity-neutral forecast remains positive but unexecuted at 0.09729954%, below its planning gate. Preserve those distinctions and all earlier failed candidates.

Root owns serial numerical execution under 7.5 GB / zero-added-swap guards and single-thread BLAS, with only the three existing teammates. Agents perform source/metadata work only unless explicitly assigned otherwise. There is no pending numerical continuation to restart.

Restart receipt: `continuation_aggregate_tree_closeout_v1.json`. Prior full state: `HILL_CLIMB_before_broad_outcome_v1.md`; previous root handoff archive: `HANDOFF_before_aggregate_tree_closeout_v1.md`.
