# Fixed terminal representation information test

Parent is frozen Read5 source
d7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b.
Its archive stays unchanged and unsubmitted pending explicit approval.

Question: does the final shared encoder discard information useful for its
remaining mean error? Its actual terminal input has126 channels, two SiLU
layers have128 channels each, and the final linear encoder compresses to64.
The earlier H15-quadratic test and covariance-energy test failed. This test
instead accesses actual intermediate nonlinear features before that64-channel
compression. It changes no physical state, weights, gates, ranks or head.
It is not a longer fit or rescue of either failed feature family.

Instrument gnn_step only at layer15 to retain its existing input, first and
second SiLU activations, final64 state and sigma. Source removal must restore
the original function byte for byte. No numerical operation is added inside
the prediction. Require full16 output and paid-bill equality with the saved
Read5 calls on35/100/35, plus byte equality of every observed array on repeat.
Use the existing nice10/BLAS1/8GB/256MiB-swap guard, serial processes only.

Then collect the fixed TRAIN20 ranks35,36,38,40,41,43,44,45,46,47,49,50,51,
52,53,54,55,56,57,58, reusing cold35 qualification. Every remaining network
gets its own cold bill. Capture all features/outputs before opening targets.
Join to the original1B target manifest and exact weight hashes. These networks
have prior training/exploration history: the result is exploratory network
cross-validation, not independent generalization evidence.

Five folds are ordinal modulo5, fixed in ascending rank order. Baseline
control X is sigma32 times [1,H15]; extension Z is sigma32 times the126
encoder-input,128 first-SiLU and128 second-SiLU channels (382 total).
Fit the raw residual truth-minus-served, weighted by each parent's actual
cold cost multiplier. All normalizers are TRAIN-fold-only weighted RMS,
floored at1e-12. The affine control uses ridge1e-4 times total sample weight.

To distinguish additional information from duplicate-column ridge effects,
project normalized Z off the normalized control span with a weighted SVD
pseudoinverse, relative singular cutoff1e-7. Orthogonalize on TRAIN only and
apply that same projection to held networks. Drop residual columns whose
TRAIN weighted RMS is <=1e-6, then normalize surviving residual columns and
fit one ridge1e-4 residual correction atop the unchanged affine control.
There is no penalty/rank/feature/seed/epoch or layer-window selection.
Save the complete projection, coefficients, rank and training orthogonality
check per fold. No nonlinear optimizer or recurrent/backbone derivative.

Seal all OOF served FP32 nonnegative predictions before aggregating scores.
Charge1MF beyond each own-cold parent for the affine control and3MF for
the extension. These are conservative information-screen readout reserves,
NOT qualified deployment bills. The projection can be folded into readout
coefficients offline; it need not run as a382-by65 inference multiplication.

Useful information requires pooled adjusted error <=.95 of unchanged,
<=.97 of affine control, and at least4/5 fold wins against both. A negative
closes this fixed span without a fit, CHECK opening, larger-feature/penalty
sweep or deployment. A positive permits a separately specified native
readout qualification and new-fit-held CHECK4 before any bank claim.
The approximately1e-9 goal is not replaced by this information screen.
