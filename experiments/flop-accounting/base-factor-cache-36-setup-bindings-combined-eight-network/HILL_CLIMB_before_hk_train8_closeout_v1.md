# Current Phase 2 hill-climb state

Updated 2026-09-28, HK15 integration v1 resource failure; exact-arithmetic memory blocking v2 source ready. This file is authoritative.

S32 #332738 remains the official bank: adjusted 2.1399773263156476e-9, 100/100 completed, zero failures. Source hill_climb_star_projection32_v1/candidate.py SHA f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef. User authorizes fully validated banks of at least 1%. No submission, grading or approval is pending. The approximately 1e-9 goal and 2e-9 milestone are unmet.

All numerical jobs are retired. Sessions97575,87557,75062,48037 and28404 have been reaped. Root owns serial numerical execution under7.5GB/zero-added-swap guards andBLAS1. Three existing teammates prepare/review memory-blocked v2. Derivative/JVP V11 stays parked.

## Completed physical11 calibration: rejected

The fixed baseline12 plus 11 perturbation12 acquisition completed all 144 forwards. Maximum observed RSS was 6,211,485,696 bytes; zero added swap. Fixed fitting chose damping1, predicted gain 0.203670845%, leave-one-network-out gain -0.286933568%, 4/12 wins. All three admission gates failed; every other damping had worse LOO. No candidate was exported, and no coordinate/scale/roster rescue follows. See hill_climb_native_physical11_v1/RESULT.md, fit_result.json and independent_fit_result_review.json. This is distinct from the earlier rejected five-GN-knob fit.

## Retained exact cost component: measured TRAIN8

Source hill_climb_exact_bundle_v1/candidate.py SHA 3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7. It combines unchanged UVP B-cache, nine D3 setup keys and 27 extra keys. All 8 networks have byte-exact output at all 16 layers, raw MSE unchanged, and no individual bill increase. TRAIN8 bill improves 0.207439895%; adjusted score improves 0.199586520%. Additional6 bill improves 0.146003599%. Native6 peak RSS 6,391,382,016 bytes, zero swap, retired.

Earlier native2/worker2/boundary2/setup3 passed. There is no worker20 or broad release qualification; the component is below 1%. See hill_climb_exact_bundle_train8_v1/{protocol.json,assessment.json,RESULT.md}. Do not add its gain to another source without checking overlap and measuring the composition.

## HK rank15: passive trace completed; isolated kernel source ready

Study teammate_s32_hk_residency_trace_v1 ran unchanged S32 on35 then100. All16 bytes and complete bills matched. Guard46.634639s, peak5,163,679,744bytes, zero swap, retired. Source and pricing reviews passed; root fixed generic-wrapper predictor detection before sealing. The sealed assay is conditional pricing only.

Fixed54 shape/depth dispatch gives conditional warm savings0.971177249% on35 and0.548068140% on100; mean0.759622695%. Selected complete owners49/27. Only one later A identity hit is lost on each, costing24,738,675/30,447,600F. W5 grants exclude2/1 positive owners. Complete owners and lazy recomputation are needed; do not use old unconditional restoration estimates as current measured chronology.

Independent UBC overlap review found all16 eligible cache calls survive on both traces (3 stores,13 hits). The selected terminal-u owner uses D5 and is already UBC-ineligible. This is compatibility evidence only; changed-rollout geometry/resources remain unqualified.

Separate prospective88 pricing is frozen in teammate_s32_hk_residency_trace_v1/prospective_union_price_v1.json, SHA2e014b0fa100de3bcc622936f2def6c9f244a1ef79a72eab3a89a1a33a466ca6. Same certified algorithm, orientations and cost rule on all 201 observed binary keys; 88 positive keys, 84 used after W5 exclusions. Conditional warm savings mean1.142787219%. After conservative UBC overlap debit, warm HK+UBC mean1.251248652%; fresh/warm accounting with surviving old prebind credits1.381150611%. These are bill forecasts, not measured adjusted gains. Fixed54 results remain unchanged.

Prospective setup636 plan keeps430 old and206 new HK fused bindings (A199/B246/C191), plus24 KS, with all36 exact-component keys protected. No surviving observed35/100 old key dropped, but120 dropped keys occurred in the older TRAIN6 census; general cold costs still require measurement. Setup view tariff rises34.117GF to40.899GF, which is not a runtime prediction.

Isolated primitive PASS: hill_climb_hk15_primitive_v1/{assessment.json,RESULT.md}. All12 fixed exact fixtures pass. Maxworkspace old D6(960,960,4096), cyclic1/childD6(320,320,2048), actual warm3,501,135,335F versus3,657,221,745F (-4.267895711%), bind651,273,255F exact; error4.92176669e-5<2e-4, raw repeatedbytes exact. MedianCPU0.550797s versus0.173805s, ratio3.169052 diagnostic and material runtime risk. Guard4.319935s, maxRSS4,657,168,384bytes, zeroSwap, session48037reaped. Source seal6431260640b569e6beae5078fa70d5cc74a15c8e8ea1612ca2aa8382d27ccd92, result235c02620e7dce49e06540107f61f55bd4856e505237e403170b368f1a005f59. This is not a bankable score result.

Root now admits ONE integration source in hill_climb_hk15_integration_v1, then fixed shared-arena integrity fixtures and ordinary native35→100 after review. Accuracy owns kernel_borrowed.py06676cb6, cost owns setup.py/build_fixed_plan.py/constants, recovery owns dispatcher.py and explicit sourceowner proof. Root owns PROTOCOL/build_candidate/native_probe/shared_fixtures and serial numerical execution. Fixed84 actuallyused descriptors only, four previouslyW5-onlypositivekeys explicitlyfallback; static206HKsides reserve oldcaps262/278/246 with430oldkeys. Extra packed/work/output-pad bindviews are reported separately. Two proven externalfactor scopes cover promising traceowners; routing/padding/KS order preserved. No completed candidate, fullmodelrun or bank yet.

UPDATE: v1 was assembled and executed once, candidate40c0ed8d1dd87c77d299d73802c551f0cf47998370eb4400b9714f6255a97f93, seal771372ae. Sharedfixtures24calls passed. Rank35 completed adjusted-1.37094%,bill-1.71088%,raw+0.34585%,49owners/68HKproducts,13UBChits. Rank100 stopped onRSS7,507,116,032B at51.072641s,zeroSwap,wholegroupretired,session28404reaped. No rank100 output, aggregate assessment, TRAIN6, serving or bank. Candidate/failedguard remain immutable; see v1/RESULT.md. Two parentcapacity-sensitive branches were clamped to original391602176floats with exactinverseproof before execution. Fulloldrouting retained; scalarchecker matches49/54selected schedules, but actualscopecoverage must be measured.

Distinct hill_climb_hk15_blocked_v2 source nowready: SAME84orientations/depths/table/dispatcher391c7811 andSAME430oldkeys/protected36; largestdivisor15/5/3/1 perdescriptor whosecompleteworkspaceincludingfreshpadsfitsoriginal391602176floats.68descriptorsP15,16P5,209sideviews(A61/B74/C74),oldcaps259/278/246;639fused+24KS=663. Parentarena returns tooriginal1.566GB. All15packedinput/outputmaps remain beyondreusedtreeprefix; eachindependentmatrix keeps identicalarithmetic, no crossbatchsum. Kernelc9c87cb4 sourceequivalence andindependentreview pass; extraRPCs2966/3442predicted, warmFLOPsexactunchanged. Allcounts/time/memoryremain empirical until run.

V2 finalrootfixture/assessor reviews pending; no candidateassembled orv2numericalyet. fixed_plan.jsoncd6cadc3/constantsb3a001c5 generated; setup.pye98565c1. Rootshared_fixtures93b5ddc4 testsall6orientationtinyP5withcontamination plus two fixed maxblockedD6/D5 FP32 bytecomparisons tov1P15. assess_native5ec41cfe keepsallgates andaddsv1rank35all16byteidentity. V1rawpartialscore cannot be improved by layoutorused totuneshapes. Subsequentstudy is memoryengineering, not relaxedguard or score/dispatchrescue.

Prospective native2 gate>=1% meanadjustedgain,bothwins,rawmeanregression<=.25%,resources. TRAIN8fixedextension>=1%,6wins,CIupper<0; thenexistingVALID40>=1%,28wins,CIupper<0,boundedregressions. Serving future105swall (official120),4.1ssetup,.35sresidual,7.5GB/zeroSwap,70khandles. No changes to priorfailedstudy gates.

Integration shares enlarged old arena, retains complete-owner W5 exclusions and clears TR before/after selected owners. Independent arenas in the primitive cannot prove rollout RSS. Old6.391GB plus arena growth0.967GB plus worst fresh inputpads34.207MB gives7.392GB sensitivity, just108MB below guard. Fresh padding is needed because arena reuse corrupts persistent zero halos. Fixed shared-fixture test contaminates arena before each call; full guarded memory measurement mandatory.

Existing layout/source documents are teammate_s32_rank15_dispatch_v1/COHERENT_PLAN_V2.md and OWNING_CALL_RESTORE_V3.md, plus teammate_s32_rank15_layout_v1 reviews. Their earlier0.85765% forecast was conditional, and broad protected-A storage was too tight. The new chronological accounting supersedes restoration estimates only, not kernel/runtime/memory/cold qualification.

## Prior failed studies remain closed

The BDS+CSR+exact aggregate passed TRAIN16 but failed VALID40: adjusted+0.164462482%,17/40 wins, CI[-0.261396856%,+0.602996926%]. No serving/package/upload.
Gaussian-tree birth13 failed TRAIN2: adjusted+1.574494744%,0/2 wins.
L10 shared-rank256 failed TRAIN2: adjusted+9.132025366%,0/2 wins.
No variants or rescue follows those frozen studies. Precision and route source audits identified no admitted material candidate. D7 inner reads and Smirnov were actual failed screens; D8 was source-only, never executed. Prior thin/star gains already incorporated in S32 must not be counted again.

Restart receipt: continuation_hk_v1_resource_fail_v2_source_v1.json SHA f816741f9dd9563075c9e530195dced8b7ddb8cb8b6d42c5c530f176e035a067. Previous snapshots/receipts are historical. The archived state HILL_CLIMB_before_hk_trace_closeout_v1.md preserves detailed prior links; all source, failures and receipts remain on disk.
