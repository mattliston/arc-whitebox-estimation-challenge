# Current Phase 2 hill-climb state

Updated 2026-09-28. User authorizes submission of fully validated adjusted gains >=1%. Goal active and unmet. Root owns serial numerical execution under BLAS1, RSS below7.5GB, zero added swap and bounded wall time. Three teammates handle source work/review. No submission, grading or approval pending.

Official bank S32 #332738: adjusted2.1399773263156476e-9,100/100 complete,zero failures. Source hill_climb_star_projection32_v1/candidate.py SHA256 f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef. Receipt official_bank_S32_v1.json. No later source release-qualified.

## Direct32 Q2 closed

hill_climb_star32_q2_v1/candidate.py SHA256 ac27b278dbcbc3cbe79271b32d134d533451d8ebc629bb3b28a80f352000df2a is retained exact bundle plus only `_LR_Q = 2`. Source review passes, existing K32/RNone/POWERTrue and final factor unchanged. Old randomized K48/R32 Q2 adverse evidence disclosed in PROTOCOL.md.

Native TRAIN2 ranks35,100 completed and session62205 reaped; tag hill_climb_star32_q2_train2_v1, guard hill_climb_star32_q2_v1/native2_guard.json. Guard47.089727s,maxRSS5,687,263,232B,zero swap,whole group retired. Rank35 adjusted1.809374101082264e-9;rank1001.926104497391071e-9. Only1/2 improves, so fixed both-win gate fails. Formal assessment passed integrity and rejected: adjusted+0.0826773197%,raw-0.1508522185%,bill+0.2257281468%,1/2wins; assessor guard complete/zero swap/retired. See native2_assessment.json and RESULT.md. No TRAIN6,VALID40,serving or rescue; Q3/blend/damping/rank/roster grid forbidden. All Q2 numerical jobs retired.

CP/star consistency candidate6c3d5ad0 closed after native2: adjusted-0.0123402%,raw+0.3487049%,bill-0.3678176%,1/2wins. All score gates fail. Guard46.9622s/5.57885GB/zeroSwap/retired;34470reaped. Formal assessor24.9269s/98.9MB/zeroSwap/retired;87179reaped. No extension or rescue. See hill_climb_cp_star_consistency_v1/RESULT.md.

Current candidate: hill_climb_precision_symmetry_bundle_v1/candidate.py SHA256 1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3 (943819B), exactbundle parent transformed by eight reversible precision edits then fixed256-row symmetry fragment. Source/inverse reviews pass. Precision574a0067, symmetry7f73b4d4, inventory31e702bf. Mandatory casts/reductions lower total bill forecast to0.885790% first-use (0.907454% warm), not a measured score; >=1% score gate remains fixed.

Symmetry primitive PASSED: production integer Gram exact, Gaussian child relativeerror3.9311121e-5 versusparent3.4350611e-5 (<2e-4). Warm736,848,416F versus828,796,736F; firstchild796,760,576F. Guard1.56269s,peak1,066,860,544B,zero swap,retired;23023reaped. Full source-bound primitive receipt in symmetry_result.json. It does not establish full-model memory, score or runtime.

Native2 completed/reaped3687: adjusted-0.9656570609%,raw+0.1079910891%,bill-1.0802605597%,2/2wins. Integrity/resources passed (46.6139s/5.08994GB/zeroSwap). Original gain gate FAILED and receipt remains admitted=false. Root explicitly documented a near-threshold extension on identical source and originally selected six ranks; source/old protocol/failed receipt stay immutable. This selection deviation is extension_admission_root.json SHAa116905115395f46b2e3ac45bc59dfe9803223ef3d439e778e7329a6d777daf9, written before extension outcomes. Final >=1%/wins/CI/raw/VALID40/serving gates are unchanged; no source rescue.

Fixed-six extension completed/reaped6267:138.465529s,maxRSS6,557,630,464B,zeroSwap. TRAIN8 extension assessment PASSED integrity but remains admitted=false: adjusted-0.819181720280%,raw+0.081420900811%,bill-0.905745729698%,8/8wins,CI[-1.114302450755%,-0.528279532456%]. Only1%gain gate fails. Assessor guard0.177904s/zeroSwap/retired. No VALID40,serving,package/upload. See RESULT_train8.md and native8_extension_assessment.json. Retain exact source1571846b as an empirically positive component; not a bank or retroactive pass.

# Precision + core carry: closed

TRAIN2 ranks35,100 passed integrity, finite-all16 and L0–9 byte parity to precision parent. Adjusted -0.961509948488%, raw +0.111934480963%, bill -1.079988987103%, 2/2 wins versus S32. Original 1% gate fails. Increment is slightly worse than precision parent (-0.965657060929%), so do not retain the core addition or extend it.

Native guard 46.675471s, peak4,984,553,472B, zero added swap; session37898 reaped. Assessment guard0.186665s, zero swap, retired. No TRAIN6, broad validation, serving or upload. Original exact-parent core source remains unexecuted.

New prospective broad extension: hill_climb_precision_broad_v1/admission.json pins unchanged precision1571846b and four existing S32 VALID40 orders. Selected after TRAIN8; original failed gate preserved. Score gates >=1%/28wins/CIupper<0/maxreg10; serving required before submission. New policy hill_climb_policy_20260928_v2.json reports raw changes without arbitrary0.25% veto. Broad serial runner active session14099; blocks1/2 complete (228.06s/6.190GB and235.06s/5.649GB, bothzeroSwap), block3 running. Run script run_blocks.py checks each guard before next block; all four fixed blocks planned without score-based peeking.

Derivative/JVP remains parked.

## HK15 family closed

Primitive passed isolated arithmetic/cost/error but CPU3.169x. Integration v1 candidate40c0ed8d completed only rank35 then rank100 hit7.507GB RSS guard; no whole-screen result.

Memory-only blocked v2 candidate6649ddcf restored original arena, passed P5/P15 fixtures and all16rank35 byte identity with v1. Native2 passed1.596807568%adjustedgain,2/2wins,raw+0.132522538%,peak5.762GB,zero swap.

Fixed TRAIN8 rejected v2: adjusted-0.47415648380658704%,raw+0.35514332969242%,bill-0.8489700965794489%,5/8wins,95%CI[-0.9991990484410722%,-0.05484216617511207%]. Gain,wins,raw gates fail. TRAIN6 guard177.7093s,peak6.23457GB,zero swap. All HK jobs retired. No VALID40,serving,package/upload or table/depth/orientation/precision/roster rescue.

See hill_climb_hk15_blocked_v2/{native8_assessment.json,RESULT.md} and continuation_hk15_v2_train8_closed_v1.json SHA256 d195e6a768aa9c9aee6f035ec818e1974e3cd81c394382ea9e66c36b2a3c6198. Separate source/JSON runtime projection exceeded105s prospective gate; not a remote measurement.

## Retained exact component and earlier closures

hill_climb_exact_bundle_v1/candidate.py SHA2563651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7 combines UBC B-cache and36 extra setup keys. TRAIN8 all16 outputs byte-exact8/8;adjusted-0.199586520%,bill-0.207439895%,no individual bill increase. Native6 peak6.391GB,zero swap. Earlier native2/worker2/boundary2/setup3 passed; no worker20/broad release qualification. Below1%; retained for measured composition only.

Physical11 completed144 forwards but predictedgain0.20367%,LOO-0.28693%,4/12wins failed;no export. BDS+CSR+exact aggregate failed VALID40 atadjusted+0.16446%,17/40wins. Gaussianbirth13,L10rank256 failed TRAIN2. D7inner,Smirnov actually failed screens;D8 source-only. Existing thin/star gains already inS32. Detailed older state archived in HILL_CLIMB_before_hk_train8_closeout_v1.md. Source/failures/receipts remain immutable. Closed families do not silently reopen.

## Next fixed composition prepared

hill_climb_precision_g4_symdiag_v1/candidate.py SHA09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd (946324B): GaussianR4d3379e66 plus exact symdiagbbbc4bc7 suffix, which refines diagonal256 blocks to[128,64,64] atD6 and adds13setupkeys. Bound673/fused649,sides200/259/190 underexistingcaps. Buildb14b262b and protocol04645d4f pin source/inverse/history. Accuracy adapting assessor; recovery review; cost portableprimitive. No numerical execution of new components/composition yet. Standalone R4 andsymdiag remain unexecuted, not empirical passes.

Run prospective primitive first; then native2component adjusted<0 versusprecision157; fixed6 yields TRAIN8component<=-.05%,6wins,CIupper<0. Finalbank>=1% versus S32 onfixed40/28wins/CIupper<0/maxreg10 andserving. No raw0.25%veto. Estimated arithmetic saving alone is not score/runtime evidence; extra12.9kexplicitFNPcalls forecast requires actualservingchecks.
