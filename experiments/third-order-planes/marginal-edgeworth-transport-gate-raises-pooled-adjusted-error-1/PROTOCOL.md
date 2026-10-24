# Non-Gaussian gate transport of the existing cubic factor state

Phase 2, 2026-09-19. Parent: exact Weight5, SHA256
a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9.
This round changes the inherited CP sector's diagonal transport, not its
rank, basis metric, birth coefficients, pair map, or learned readout.
The prior radial-L4-squared and pair-local-L3-squared negatives stay closed.

For alpha=mu/sigma, gamma3=k3/sigma^3 and gamma4=k4/sigma^4, integrating
the first-order marginal Edgeworth density over the positive half-line gives

    gE = Phi(alpha) + phi(alpha) *
         [gamma3*(alpha^2-1)/6 - gamma4*(alpha^3-3*alpha)/24].

Use gCP=clip(gE,0,1), with no tuned coefficient. Clipping is a declared
probability projection of an approximate density; no exact-law or complete
second-order cumulant claim is made. The retained CP tensor is transformed
by gCP on all three legs. Its inherited repeated plane and shared-old basis
commit use the same gCP. The existing coordinate star is rebuilt from the
original target plane minus this changed CP plane, unchanged newborn and
unchanged trace. Thus the represented nonlinear target plane is restored in
real arithmetic; future all-distinct content is changed. Original FP32
factor storage and FP64 plane assembly remain the parent's conventions.

Keep the Gaussian gate in all learned features, Gaussian newborn source,
BD/NODE modification, masks, trace/radial updates and all other consumers.
This is an isolated approximation to CP inheritance, not a replacement of
every nonlinear derivative by an Edgeworth one. Existing BD's skew gate
jet changes the NEWBORN base; it does not update historical CP factors.
No new state array persists between predictions. The parent first Gaussian
layer has zero pre-cumulants, so the first two served layers should remain
byte exact before changed inheritance is first consumed at layer2.

First certify the probability formula against independent high-order
Gauss-Legendre integration of the explicit Hermite density; absolute error
<=2e-12 on the fixed alpha/gamma fixtures. Verify zero-cumulant identity,
probability bounds, dense three-leg gating and exact repeated-plane star
restoration (<=1e-12 absolute). Algebra does not forecast final accuracy.

Then one fixed native35/100/35 and separate cold100, serial nice10 and all
BLAS threads1 under the existing8GB/256MiB-swap guard. Seal all outputs,
complete own bills, source and weight hashes before reading the two existing
1B final means. Require finite nonnegative all16 outputs, exact own repeat,
warm/cold100 and first-two-layer parent equality. Advance only if both
own-cold adjusted scores improve and their pooled ratio is <=.99. J is raw
MSE times max(.1, complete own FLOPs/2^41). These are reused development
networks, not independent validation. A failure closes this exact test,
without strength/order/layer-window/mask/rank/precision changes or fitting.
Only a pass can justify wider validation and client qualification. No upload
is selected by a development result, and no existing archive is modified.
