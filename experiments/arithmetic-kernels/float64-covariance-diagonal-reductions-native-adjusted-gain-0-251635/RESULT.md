# Small positive native result, below advancement threshold

Replacing only covariance diagonals with float64 reductions improves
pooled adjusted error by0.251635% on development ranks35/100, with2/2 wins.
It does not meet the prospectively required1% TRAIN2 improvement. No
TRAIN6, precision/layer/strength sweep, refit, composition, validation or
submission follows this fixed test. This positive point estimate is not a
validated population gain or evidence of reaching the user’s score target.

| Rank | Raw MSE change | Full native bill change | Adjusted change |
| --- | ---: | ---: | ---: |
| 35 | -0.431480% | +0.043155% | -0.388511% |
| 100 | -0.166642% | +0.043276% | -0.123438% |
| Pooled | -0.294540% | +0.043215% | -0.251635% |

Layer0 uses the exact input weight-norm variance. Hidden layers reuse the
original rounded first covariance product and reduce its outer diagonal
in float64. The final layer already used this reduction and is unchanged.
The covariance cast and fast first product remain approximate; this does
not establish exact hidden variances. All casts/products/reductions are
charged, adding121634816 FLOPs per network on these two cases.

Candidate_v2 SHA170c974c6898bcda3fc6cfb0bc936177a28b3edaf44d74258e335416d6c56aff.
Small signed/asymmetric native-helper checks agree with independent scalar
float64 summation within5.56e-17; off-diagonal and star output bytes are
unchanged for identical inputs. Both real layer0 outputs match the exact
Gaussian ReLU formula after the original output cast. Every final MSE and
adjusted score is independently reconciled with math.fsum and the complete
native FLOP counts. The two development networks and1B targets were reused.

The first synthetic attempt stopped on an unsupported flopscope.free call
before any network run. All original sources/seal/failure receipts are
preserved. V2 removes that call and retains the explicit deletion of the
temporary references; it changes no arithmetic or scientific gate. See
IMPLEMENTATION_FIX.md. Four groups retired, three successful; peak sampled
RSS5010571264 bytes, zero added swap. Session51816 reapedexit0.

Official S31#332517 remains2.1736465056908115e-9 adjusted. The source and
archive are unchanged. This candidate has no held-out or actual-worker
qualification and is not selected for release.
