# Polynomial fusion on the qualified Lazy parent

Declared 2026-09-20 UTC, before building or running this candidate. Phase 2
only. Parent is the submitted Lazy source with SHA256
`b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c`.
This is a separate current-parent experiment, not a new interpretation of
the September 13 component gate. That older loop-based port passed numerical
integrity but failed its separate terminal-block timing condition. Its source,
protocol, results and failed verdict remain unchanged. The present parent
already passes full runtime and has a vectorized series and scalar final
layer; the actual estimator has fourteen three-channel pair evaluations.

Change one polynomial helper and remove its now-unused T31 construction/cast
in the two callers. Fuse Gaussian orders 1..3 with their local third/fourth
and T31 corrections. For k=1,2 the derivative change is
`t3 F[k+3]/6 + t4 (4 k F[k+2]+F[k+4])/24`; for k=3 it is
`t4 F[5]/2`. Accumulate `(F+DF) G + F DG`, without a DF DG term.
Keep Gaussian orders 4..6 and all T21/T22 terms. Keep physical covariance
asymmetry and existing precision, initialization, kernels, state, learned
parameters and heads. The real-arithmetic identity assumes symmetric rho;
changed FP32 association, T31 rounding and asymmetric rho can change outputs.
No byte-equivalence claim or hidden covariance repair is made.

First certify the symmetric identity against the independently executed parent
helper in FP64 on fixed synthetic fixtures, both one and three channels,
absolute error <=2e-12. On realistic FP32 low-correlation fixtures require
max error <=2e-6 and RMS <=2e-7 for symmetric and small-asymmetric rho;
these are integrity bounds, not network error gates. A fixed full-width
three-channel fixture must use fewer FLOPs than the parent. Repeat every
fixture in fresh contexts and require byte-exact output/bill repeat. All
fixture input arrays stay unchanged. No reference means are opened here.

After unit pass, run exactly 35/100/35 and separate cold100, on the entire
candidate using the inherited native harness. Require finite nonnegative
16x1024 FP32 outputs, exact first layer, exact repeat and warm/cold parity,
unchanged source/weight hashes and retired numeric caches. Seal predictions
and complete own bills before reference means. Use the existing ordinary1B
development truths. Require lower cold bills, both adjusted errors improved,
pooled adjusted ratio <=.999, and neither raw error ratio >1.01. These are
prospective cost-component gates; no coefficient, rank, schedule, layer,
precision, association or threshold alternatives will be selected afterward.

Only if development passes, run the fixed existing public8/fresh6 panel;
reuse these two cold predictions and perform twelve new cold calls. Recompute
paired noise-corrected errors from existing 4M references and saved Lazy
outputs, not old raw scores. Require improvement in both groups, all14
adjusted ratio <=.998, at least12/14 wins, and no individual ratio >1.01.
Then run the unchanged candidate cold on fixed confirmation ranks
[37,39,42,48,59,63,66,67] against saved Lazy cold results and ordinary1B
truths. Require pooled adjusted ratio <=.998, >=6/8 wins, maximum ratio <=1.01.
These are previously used networks, not globally untouched validation.

Only after those score checks, run the actual NumPy-free client35/100/35.
Require exact native/client outputs and bills, repeated-output identity,
each complete call <45s, pooled time <=1.05 times saved Lazy client time.
Complete tiny contract and archive validation. A fully qualified archive may
be submitted under the user's standing board authorization; the official
result is separate from any local projection. Failure closes this candidate
without a follow-up sweep, fit, panel expansion or upload.

One numerical process group at a time, nice10, BLAS1, external 8GB groupRSS
and 256MiB added-swap guard; zero swap growth required for qualification.
Keep every source and receipt versioned; preserve the current bank and all
old negative results. No new Monte Carlo datasets or training. An incremental
cost gain would not achieve the approximately1e-9 goal, which stays active.
