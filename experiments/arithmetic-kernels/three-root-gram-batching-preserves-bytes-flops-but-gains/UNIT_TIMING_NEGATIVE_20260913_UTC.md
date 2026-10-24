# Root-batch Gram: exact algebra, insufficient measured time gain

Actual2026-09-13 UTC unit2790 retired completely. Operational PASS, fixed
timing gate FAIL. No model call, raw/adjusted score, full-card change or upload.
Official bank330390 unchanged. Preserve V1 inert failure and all V2 sources.

Same fixed synthetic1024 Gram/three512x1024x512 D5 roots, one fresh paid RPC
context, original serial tree versus three-root batching. Actual block times:

| Role | Seconds | Complete helper F |
| --- | ---: | ---: |
| serial first |0.274143542163074|955654144|
| batch first |0.264488167129457|955654144|
| serial warm |0.2271282086148858|955654144|
| batch warm |0.220575999468565|955654144|

Warm ratio0.9711519357886954, a2.884806% measured component reduction; required
<=.90. Retain this small exact-component observation, but do not call it a
validated whole-card speedup or use it to pass the45s release gate. No repeat,
shape/depth/batch scan or full-card port follows this fixed question.

All four served FP32 matrices are byte-identical, including repeats, SHA
1e2eac2824c2172fce6bb7fe6ca1c989af805b4c21c9b9a245d44607d7909cc7.
Against the one denseFP64 product of exact serializedFP32 operands, maxabs
7.996167403032106e-6 and relativeFrobenius1.0127841677084965e-5; original gates
5e-5/2e-5 passed. Scalar arithmetic/helperF unchanged; fewer operators did not
produce the required time reduction.

Actual COMPLETE context8957101063F, including both bindings, RNG, ingress,
four helpers, independent FP64 reference, every copy/finite/error check.
Context1.210216583s, backend0.868979454s, FlopScope overhead0.316766112s,
residual0.024471017s. All480 tracked array refs retired before context close.
Whole4.664638210s, peak tree1267597312B, max completed observation0.419345s,
zero added swap, all groups/owners reaped. No numerical retry authorized.

Worker result/completion/finalization:
d7fd38f7b6b8d1ec70739e6a385ed2a40f1743104554a26ef837c54802238f93
d3c6abc0241039f030ae4f574a08a7f4007732896ef5c90dc4500019676d8258
dda30cf78ac2ab63205af98cca17d77cf28c7ba521a0d0d70f7161a0e8427911

Watch result/completion/finalization:
476501d2137232a196c40933b713dec3f7b560868543fc4edad6c14b1b3bc77b
d5a5dad23e61e2d7afab7d855e35385a04dc8fa7cebb57ff3e1006932c2f9168
a68412ed54d8869d3ab43eb295204840f4647d1083b0d27230ffda37932799eb
