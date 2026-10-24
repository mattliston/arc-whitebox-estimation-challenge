# Nonlinear historical pair memory: fixed information test failed

Phase 2 only. September 6, 2026. Judgment finished 21:36:54 UTC / 14:36:54 PDT.
This is a qualified negative, not a failed acquisition or noisy MC estimate.
No new candidate or submission follows. Official bank #329926 remains
adjusted 4.982231460183304e-9; that is not a fresh board poll.

## What was tested

Frozen protocol `NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md`, with
the source-only directed-correlation correction documented in
`PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md`. Twenty TRAIN networks from ranks35–58;
VALID exactly37,39,42,48. Only post12 -> pre13 offdiagonal bank third action
was scored. Actual previous bank lower moments and represented K21 history
were supplied for free; this was an optimistic conditional information test.

The predictor used the exact three-product coordinate-star transport plus
two shared coefficients multiplying normalized K and K U. U came either
from the fixed nonlinear persistent history or the matched immediate-source
RESET. Current pre13 K21 was loss-only, decoded after predictor freeze.
TRAIN-only joint covariance/next-mean normalization and one two-by-two ridge
solve per arm preceded all VALID action decoding. No Monte Carlo truth,
current-query diagonal, private labels, neural fit or endogenous rollout.

The covariance consumer was the full frozen r=0..2 cubic response, preserving
the same directed rho for both K21 orientations and the original variance
denominator. The next consumer was the leading pre14 Gaussian variance-to-mean
response. Neither quantity is final raw MSE or the complete head Jacobian.

## Held-out result

Ratios below are residual squared response error versus zero correction,
pooled by equal network/coordinate weighting across the four VALID networks.

| Consumer | Zero-correction MSE | History MSE | History / zero | RESET / zero |
| --- | ---: | ---: | ---: | ---: |
| Covariance | 7.55420332890e-8 | 6.83127896854e-8 | .9043017074 | .9044486819 |
| Next leading mean | 2.22084983478e-8 | 2.20466193732e-8 | .9927109446 | .9906854096 |

History / RESET pooled next-mean error is **1.0020445794**. History loses on
all four networks; per-net ratios are37:1.00295073,39:1.00007517,
42:1.00219598,48:1.00275726. All four fixed gates fail: covariance <=.50,
next mean <=.50, history/RESET next <=.80, and >=3/4 next-mean wins.

History coefficients are[-.00287137328019,-.000228368440528]; RESET
[-.00308812824208,+.000182416869501]. Normal-equation condition numbers
are3.65577 and2.41839 respectively, with finite positive TRAIN energies.
There is no singular-solve or near-zero-denominator ambiguity.

Do not extend this fixed source, decay, normalization, layer, rank, feature,
ridge, or coefficient family. This does not prove impossibility of every
nonlinear recurrent matrix architecture; it rejects the precise tested lead.

## Numerical and resource qualification

Original capture v1 rows35–37 passed numerical parity, but37 exceeded the
fixed6GiB cap:6,459,342,848 bytes versus6,442,450,944. The v1 panel driver
correctly stopped before38; all sources, rows and failure logs remain.

An additive v2 resource-only replay accumulated the OFFLINE physical CP read
in fixed2048-column float64 chunks. Bank state arithmetic was unchanged.
All24 v2 captures passed at4,955,635,712–5,534,760,960 bytes peak RSS.
The three existing v1 rows were compared with v2: all original bank arrays
byte-identical; worst physical-post-plane relativeL2 difference3.9588e-15
against the fixed1e-10 limit. No source or target-dependent parameter changed.

The response v2 tiny certificate passed ten symmetric/asymmetric/active-floor
fixtures, worst error1.14926e-13 against2e-11. The v1 symmetric formula's
asymmetric witness was1.25808e-4; its frozen sources/certificate are preserved.

The final judge passed all provenance, chronology and finite-value checks;
wall20.9817s, peak1,326,891,008 bytes. One nice10 NumPy-only worker, no Torch
or MPS. Two BLAS threads were requested; backend instrumentation was empty,
so the effective thread count is not independently certified. Swap remained
840.75MiB. Floored cold post-variance units are the explicit protocol choice,
not a PSD repair or grounds to relabel this negative as inconclusive.

## Frozen receipts

- Judge source SHA0485c62b96a16aa712f2ca8f262af77029475a2d9c9e1efe394643f8f5fa9ac9.
- Result `nonlinear_pair_memory_information_v1/result.json`,
  SHA3d0c7e788143b704fca3793ecf2e219bec1cd945c5f7ed374134e86676ba95eb.
- Coefficients SHAa1b7581883901b7923f5d5770588157f260880b988b741e5586f919e50de8888.
- Three-row replay qualification SHAaf8c2b0cb66b6730af9d93b5aa0e56efcc97ba3e825093c4926f4bbd3805f328.
- Response certificate v2 SHA846254792751b6b01bd658fcb477ea67605c6ed671685e2c3910c87d4b3e8e58.

The fixed test is complete and closed. The <1e-9 official adjusted goal
remains active and unmet. No threshold bank has been earned by this result.
