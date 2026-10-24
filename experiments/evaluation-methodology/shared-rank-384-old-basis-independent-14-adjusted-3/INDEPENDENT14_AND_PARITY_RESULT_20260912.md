# Actual independent14 and failed cross-backend parity

September12,2026. Candidate remains
4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3.
No submission or package was made. Official bank remains #330390,
4.53886349950395e-9 adjusted. Latest user policy requires a validated <4e-9.

## Independent14: scientific PASS, not release qualification

Fixed public8 (35,100,150,200,250,300,400,500), virgin6
(3016500..3016505), plus unscored repeat35. Same current heads and own complete
FLOPs. Score each final output against the original independent4M truth using
FP64 arithmetic and the original unclipped mean(mc_se²) correction. Average
per-net raw times its own max(.1,F/2^41), not a product of aggregate averages.

| Frame | APUSH NC adjusted | SHARED NC adjusted | Ratio |
| --- | ---: | ---: | ---: |
| public8 | 4.580621791764777e-9 | 4.196175435152373e-9 | .9160711418472539 |
| virgin6 | 3.594875562995286e-9 | 3.256346849302149e-9 | .9058301997493701 |
| combined14 | 4.158159122292138e-9 | 3.793391755502277e-9 | .9122767176382641 |

Combined SHARED NC raw2.0718239669315202e-8, mean completeF401818277064.7857.
Combined APUSH NC raw2.0303779033431815e-8, mean completeF449007784422.
Uncorrected observed SHARED raw3.9009119904349645e-8 and adjusted7.132193590403293e-9;
APUSH observed raw3.859465926846627e-8 and adjusted7.888222942018369e-9.
Both original per-panel .97 ratio gates and combined NC<4e-9 pass.
This is a Monte Carlo noise-corrected LOCAL estimate, not an official score.

All15 native calls/105 artifacts passed domains, selectors, repeat and own
closed bills. Native watch889.9932182924822s, peak6851330048B,
max completed interval.403456749394536s, zero added swap, all groups retired.
Actual native result/completion/final:
ceae3540a59b10f9ebf90885ff2b9b70f8154fc50a6cfe22e29ad5ff01d4e911;
8f31681349242356655de980eeaa44d3401812e941d99f5202b145dce8bb2983;
1ff6639648aca4e8fb60793751bcc7b281c97376d0f2d10b500bdb475d1630fc.
Watch result a7bb3ea80bbe1c65131dfa0e53c496a7e431bfc907cfcd9a83fb90c669b6fbd2.

Scorer v1's namespace-mutation TypeError occurred before any numeric/truth
access; old negative seals are preserved in SCORE_PANEL_V2_PROTOCOL.md.
Scorer v2 isolates the metadata namespace only, with the original numerical
body and all gates unchanged. No model rerun/refit under that repair.
Actual scorev2 result/completion/final:
3fe222ee2b37eb22aa8df4a82e89ecd320afd64e8b16ff52f07dd2701edec098;
aea08b595e70aaf805143888dc0c3362e8edc63aef3d4389fba9cfef979f5f3f;
df969e0e0b03a33d65e8bc026a46130640c428b6e1802cab7ad83592824ee66c.
Watch result57b8ae4e88afccc548d4b3a8ae38867b7f2a1053da654043971d2869f9ee7020;
whole6.46646125056s, peak154566656B, maxinterval.31750187464s, no swap.

## Fidelity v2: real numerical FAIL, before truth

All native/production inputs, artifacts and own bills authenticated. The
original all16 maxabs<=1e-5 gate then failed, before the four truth reads or
observed-MSE ratio checks. No result is relabelled. Source707f4b3d and
watch1f409b90 remain immutable. Worker1.8525523329153657s/92913664B;
watch2.407314791344106s/118194176B/.29168470948934555s/no swap.

Negative fidelity result/completion/final:
fcffdc67d39025333f5cb1c15cd1ee5cbd2c1df42cec01355b93d67ce0c381c9;
ec07d8e9b44dad72b36d96a589068e4f248cc7495febfd6cbbc1e5d43af79354;
0281a990041ee93dd1925fefbd51a4dde216409606789e5822b156fe4e511a0a.
Negative watch result/completion/final:
1dccf7ea43149a8f19d8b5fc79bddb93a9ff1d88bf3a555b42d87159e20100f3;
107fa0e8cbe1b3a646e92b15e707b6b762ab0e4742472cabf682777d335f15c0;
b6ec1d725247d61c553531585076175aabbefbbb59ac2b9276f89a6281c772e6.

## Target-free scalar diagnosis, no rerun or gate relaxation

diagnose_saved_parity.py dfca98ca995ca8086199131c4391c7608913cab30cdbe05739b28652b5fc7f5a
decoded only the five saved pairs using stdlib struct. It authenticated exact
file/payload hashes and physical fixture identities; no NumPy, model or truth.
Operational PASS means all five pairs were diagnosed, not fidelity PASS.
Actual0.29963541589677334s, combined60391424B, maxinterval.17220108304172754s,
no added swap and all observation groups retired.

| Rank | All16 maxabs | L15 prediction-to-prediction MSE |
| --- | ---: | ---: |
|35 (also repeat)|3.8504600524902344e-5|6.081371436460466e-11|
|100|4.082918167114258e-5|6.78433905134167e-11|
|500|3.528594970703125e-5|7.643768773744793e-11|
|3016500|3.6776065826416016e-5|5.605790184334638e-11|

These MSEs measure two predictions against each other, NOT error against
truth. Both backend repeats are byte exact. Difference is ~1e-6 atL2,
~2e-6 atL8, grows atL9+, and crosses the original vector gate atL11/L12.
Each production bill is15090F below its own corresponding native bill.
Source inspection identifies Accelerate local BLAS versus production
OpenBLAS, with identical NumPy2.4.6 QR implementation; this is a plausible
cause, not yet a demonstrated one. Further work must establish faithful
same-configuration score and original fidelity before release.
