# Current c2/OpenBLAS independent14 result — September12,2026

Actual current candidate c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea.
No refit; whole-source gain-slice inverse and native/RPC35 pair qualified.
This is new current-source evidence, not inherited old4d98 accuracy.

The15 native calls (public8,virgin6,repeat35) all passed:105 artifacts,
30 closed ledgers, all16 domains/all47 selectors, own repeat, current35
same-backend payload/full-cost replay. Watch elapsed950.9141654596s,
max combined RSS7,017,709,568B, completed interval max.4951261664s,
zero swap growth, complete cleanup and exit0.

Scoring used original independent4M FP32 truths widened to FP64, with
unclipped mean per-neuron MC variance subtraction. Per-network raw error
times its own complete max(.1,F/2**41), then group mean; not product of means.
Public35/100/200 were TRAIN and150 VALID: independence refers to truth draws,
not all networks being unseen learned-model holdouts. No new fit occurred.

| Group | Current NC raw | Current complete F mean | Current NC adjusted | Historical APUSH adjusted | Ratio |
|---|---:|---:|---:|---:|---:|
| public8 |2.2965935292537643e-8|402804969761.625|4.21257761153528e-9|4.580621791764777e-9|.9196519169316311|
| virgin6 |1.786913418264998e-8|400502686802.3333|3.26171945931551e-9|3.594875562995286e-9|.90732471879995|
| combined14 |2.0781591959728645e-8|401818277064.7857|3.805066974869664e-9|4.158159122292138e-9|.9150845032530077|

Combined observed raw3.907247219476308e-8 and observed adjusted7.143868809770681e-9;
mean MC variance1.8290880235034444e-8. Parent combined NC raw2.0303779033431815e-8,
completeF449007784422. APUSH is historical Accelerate, current candidate is
production OpenBLAS: objective comparison, not backend-matched causal ablation.
Both >=3% group improvement gates and combined NC adjusted<4e-9 PASS.
Scorer/watch operational PASS, no estimator calls or parameter updates.
Watch elapsed8.395029375329614s,maxRSS170393600B,interval.31273954175412655s,
zero swap growth, complete cleanup/exit0.

Native result/completion/finalization:

    9563919f22a938ec084751444cf1c4e643e18fd4aba1cf8b87c3526f2d3f6608
    fe75d8c7943c14a58a66025f96ecd4e97b8f89d44d12759a0d1ff2233f8bbf4c
    3578b8eb6a9c9a1c183d9c87c3dd32c3a9fa322bfa70209ecc3a6c0f4aeeb36e

Native watch:

    6fe4acd894a5dd13f0bdb558ef48c522622917413b400a0b1ca43850236474fa
    6d58d72298d7cda2049b995b3faf4c0bd090394a03b3b61d3cda35ca6a81c040
    e9bf477c760ca54805e6610407d0b5d2133035679ef989cb5e35ca40a71ea088

Score result/completion/finalization:

    8f3b8ca3d327ee6ed4d035aadefe9b28d91ab8e430ba6348561df30a0cbb4637
    ab6573d40b8633a62cbd6345aabe0ab0b3bc469846b2fdd850d39b98469de522
    ae23ebd494031f44a071166145f2dcb7df441d566d279fb5a4b14acd3517553e

Score watch:

    f75fb48bf3d2b4d05f6321509e566c11aa6ba04dc313e22954a400d8e79467b2
    45811ab69c049c0581b40eb0010e07b4786d7243dcf9ef1108b4b96b823589cf
    ba7f6025617742e5c151327f8dfd79040d47e4224cef2716b67544fcd6fa6fe1

No official score/submission claim. Official confirmed bank remains#330390
4.53886349950395e-9. Fresh current full40+score, production-five and release
checks remain required. User now requests banking below4e-9, then continued
work toward<1e-9 with<3e-9/<2e-9 milestones. Old negative receipts preserved.

