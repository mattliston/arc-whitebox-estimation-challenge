# Current Phase 2 hill-climb state

Updated 2026-09-28T04:36:05.514485+00:00. Goal active and unmet. All numerical jobs retired. No grading, upload or approval pending. Derivative/JVP V11 remains parked.

## Official bank

**S32 #332738: adjusted2.1399773263156476e-9**,1.548972% better than S31;100/100 complete, zero failures. Source `hill_climb_star_projection32_v1/candidate.py`, SHA `f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`. Receipt `official_bank_S32_v1.json`. Do not reupload.

User policy: **bank fully validated aggregate gains>=1%**. Submissions already authorized; no permission re-ask. Smaller components may accumulate. The2e-9 milestone and approximately1e-9 target remain unmet.

## Measured exact bundle

`hill_climb_exact_bundle_v1/candidate.py`, SHA `3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7`, contains unchanged UVP base B cache + nine D3 setup keys +27 extra setup keys. **Native2 passed**: all16 bytes exact on35/100, bill-0.390611902%, adjusted-0.382585761%. Cold35 saves1,836,555,374F; warm100 saves319,701,476F. Setup660/636;13 cachehits/3 paidcopies per network;337MB cache fully retired. Guard46.341s, peak5.219GB, zero swap. Session17021 reapedexit0. Evidence `hill_climb_exact_bundle_v1/native2_assessment.json` and `RESULT.md`.

**Combined worker/boundary/setup/broad validation is still pending**, and the native2 gain is below1%. Do not claim the sum of component gains or a new board bank. Use this measured source as a prospective exact bundle; keep S32 as the official scientific comparator.

The three isolated components remain preserved:

- UVP base cache `hill_climb_uvp_base_bcache_v1/candidate.py`, SHA0308a99f…: native2 bill-.11583%, actualworker2/boundary2 passed;319,701,476F and296requests saved each, zeroSwap.
- Nine D3 bindings `hill_climb_star32_fused_d3_v1/candidate.py`, SHA61a1f40b…: actualworker2/boundary2/setup3 passed; requests-996cold/-870warm, cold bill-23,761,920F, warm tie. Original equalbill assertion failure and correction preserved.
-27 bindings `hill_climb_s32_prebind27_v1/candidate.py`, SHA1b7f0244…: native2/worker2/boundary2/setup3 passed, all16exact. Cold35 saves1,493,091,978F(-.5399%), warm100 tie; pooledbill-.270478%. Setup651/627,max3.30412s, handles49,069, peak5.844GB,zeroSwap. BoundaryCPUratio1.01142,residualprojection.293451s passes. Session5358 retired. Receipt `retained_component.json`.

## Newly closed accuracy steps

- Consumer-direction star repair `hill_climb_star_consumer_residual_v2/candidate.py`, SHAc0a30ca3…: TRAIN2 passed-.431835%, but **fixed16 not retained**: adjusted-.328423%,10wins,95%CI[-.816288,+.159845], additional14-.31561%. The mean estimate is positive; uncertainty gate fails. All L0 bytes exact, guards zeroSwap/retired,peak5.668GB. No larger panel, conditionalK24 test or amplitude/rank/layer sweep. V1 runtime error from immutable FNP augmented assignment is preserved; V2 made three mechanical fixes only. `train16_assessment.json`, `RESULT.md`, `independent_review.json`.
- Terminal scalar κ3² completion `hill_climb_terminal_k3square_v1/candidate.py`, SHA731e1698…: **TRAIN2 rejected**, adjusted+.991866%,raw+.991472%,0wins; bill+.00006678%. L0–14 bytes exact, guard46.215s,peak5.274GB,zeroSwap/retired. No amplitude/order/scope sweep. This is now a current-native terminal-only failure, distinct from older RF11 recurrent+terminal losses. `result.json`, `RESULT.md`.

## Evidence corrections and existing closures

The inherited600-key prebind list came from old cutoff2.33 TRAIN6 ranks36/45/52/58/64/70, ordered by network incidence plus a0.5 incumbent prior, multiplied by reshape cost (`build_trace_prebind_v1.py`). All27 new rank35 keys were absent from that old census; rank35 was not in its panel. No saved broad current-S32 key census supports an automatic capacity-neutral refresh. Universal new D3 keys are already in the retained nine-key component.

Current S32 paid-cost census remains `hill_climb_s32_bill_census_v1`: exact276.55GF on35, push43.45%,read34.39%,star11.04%. KS chunking partition economics closed; padding only.0263%. Old cache closures were forecasts/old implementation thresholds, not executed failures; the retained337MB cache is actual measured evidence.

Other fixed rejections remain closed: globalstar24/40, birth+32L1–10(fixed16+1.04622%), common192, late negativecut2, terminalcut3, covariance-diagonal64, common256/320L7–9, energy common, no-power star, GN scalar fits, duplicateLoopR8s. Prior state preserved in `HILL_CLIMB_before_exact_bundle_v1.md` and `HILL_CLIMB_before_uvp_bcache_v1.md`.

Root owns serial numerical execution under7.5GB guards, nice10/BLAS1, zero added swap. Use only the three existing teammates. No source candidate is currently admitted for another experiment. Restart receipt `continuation_exact_bundle_v1.json`.
