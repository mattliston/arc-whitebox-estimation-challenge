# Lazy coefficient rotation: fixed fidelity gate failed

Phase 2 only. The exact preserved lazy-rotation fragment was ported to current
Weight5 and executed on native35/100/35 plus independent cold100. Complete
compute cost fell, but finite-precision prediction changes failed the fixed
all-layer maximum-absolute-difference limit. Reference means were not opened.
There is no measured raw or adjusted accuracy improvement or worsening.

## Reason for the port and its scope

The previous component's dense algebra was positive, while its older BOX
parent took 51.229 seconds at the real client. Weight5's measured local client
sequence now takes 39.4935/37.5752/38.2081 seconds. That changed runtime premise
motivated this fixed port; no old deployment qualification was inherited.

The exact numerical fragment SHA256 is
`e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92`.
Candidate SHA256 is
`b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c`.
Weight5 is an exact byte prefix. Six reversible substitutions change only the
last shared-state, push and read definitions. Rank384, age4, all QR/core updates,
all reconstructed rows, physical/learned consumers, kernels, arena and Weight5
cache ownership remain. Rotations are deferred only at layers8/10/12/14 and
consolidated at9/11/13, with the last map reused at15. At most two ordered factor
groups and one map are retained per prediction.

Dense rank384 tests pass 13 comparisons and ten rejection cases. Maximum
relative Frobenius error is 6.524844351582119e-7 against the unchanged 2e-4
limit. These bounded noncancelling fixtures certify the identities and guards,
not accuracy on the production signed factor state or packed-kernel billing.

## Complete native evidence

| Call | Own complete FLOPs | Saved versus matching Weight5 call | Native seconds |
| --- | ---: | ---: | ---: |
| Cold35 | 398080917697 | 7466832638 | 29.62456625 |
| Warm100 | 388883126034 | 7465448812 | 27.72145717 |
| Repeat35 | 377556934779 | 6337803914 | 29.18781596 |
| Cold100 | 400893063472 | 7682859772 | 28.00971583 |

Own cold FLOP ratios are 0.9815882775 and 0.9811960046. Context history affects
view binding and the measured saving; it is not one constant saving per call.
All four outputs are finite, nonnegative FP32 with shape16x1024. Own repeat and
warm/cold100 are byte exact. The first eight served layers match Weight5 exactly.
All 15 Weight5 cache creators and39 hits remain, and caches retire each call.

| Network | Largest all-layer absolute difference | Final-layer RMS difference | Values exceeding 1e-5 |
| --- | ---: | ---: | ---: |
| 35 | 2.6047229766845703e-5 | 5.091128473248542e-6 | 115 |
| 100 | 2.2977590560913086e-5 | 4.8973679118009544e-6 | 104 |

The first changed layer8 has maximum difference 1.1920928955078125e-7 on both
networks; differences grow through the subsequent recursion. This records
finite-arithmetic divergence, not proof of an error in the real-arithmetic
identities. Both violate the predeclared <=1e-5 fidelity bound.

The scorer stopped at that assertion before its prediction seal and before
any reference-mean load. Its exit1 and watcher remain preserved. A separate
target-free diagnostic inspected only the already saved predictions, hashes,
bills and weights; it neither reran a network nor loosened a gate. No wider
panel, client test, contract/package, parameter fit, precision/window/schedule
change or upload follows this fixed failure.

Four full predictions total 1,565,414,041,982 billed FLOPs. Maximum native
self-reported RSS is 6,204,063,744 bytes; external monitored peak is
6,044,827,648 bytes. All owned groups retired, and additional swap was zero.
The failed scorer is an expected qualification failure, not an OOM or timeout.

Weight5 remains qualified locally at NC14 adjusted3.883207214854186e-9 with its
exact archive question pending. Official best remains graded #331584 at
4.015535762022727e-9. The approximately1e-9 objective is active and unfinished.
