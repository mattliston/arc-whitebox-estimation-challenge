# Precision VALID40 closeout

The fixed precision source `1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3` improved pooled adjusted score by **0.7988805173%** versus S32, with **38/40 wins**, but **failed the unchanged 1% bank gate**. It is retained only as a positive working component. No serving qualification or upload is admitted; S32 #332738 remains the bank.

| Reported measure | Delta versus S32 |
| --- | ---: |
| Adjusted score | −0.7988805173% |
| Paired 95% bootstrap interval | [−0.9471885614%, −0.6557746230%] |
| Raw MSE | +0.0777467023% |
| Full charged FLOPs | −0.8711184501% |
| Native CPU | −2.5601573814% |
| Worst individual adjusted regression | +0.3538102780% |

Receipt integrity, win count, negative CI and maximum-regression gates passed; only the 1% adjusted-gain gate failed. Raw error is reported without a veto under the pinned prospective policy. The earlier TRAIN8 failure remains false. The admission discloses the outcome-informed extension and historical reuse of VALID40; it does not claim pristine validation or a retroactive training pass.

All four native blocks and the assessor completed and retired with zero added swap. Candidate block times were 228.057s, 235.058s, 232.399s, 232.498s; maximum native guard RSS was **6,190,284,800 bytes**, below 7.5 GB. The numerical assessor verified all 16 finite F64 prediction layers and retired UBC caches. This independent closeout checked source/JSON pins, exact commands and rosters, guards, row arithmetic and decisions. Prediction files stayed opaque; no arrays, model, targets or bootstrap were executed during this review.

Evidence: [assessment](broad_assessment.json) SHA256 `82bef436a90b6a81ea76fb37f9191955bc3841e31b4042bfdd131c1cca254b5c`; [admission](admission.json) `a6e223a7cff1cd8acbc562d86380ddf44427690919afb33c7feae5d3b393b2b9`; [assessor](assess_broad.py) `d776a3764648c115ee53edd2e62570f7e628ef04581d0ed36dc8afd09ad48177`; [assessor guard](assess_guard.json) `05f010a8f968630a9a8aee422de58279ebe89d0af652c5e115ba153442d84987`. The assessment pins all four native guards, source, policy, manifests, rows and prediction receipts. A separately admitted Gaussian4/symdiag composition is distinct work; this closeout does not claim its result or retirement.
