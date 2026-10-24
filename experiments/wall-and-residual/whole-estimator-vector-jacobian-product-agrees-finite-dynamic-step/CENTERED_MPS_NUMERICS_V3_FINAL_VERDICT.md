# Repaired native-MPS bank: forward and local VJP fidelity established

September6,2026. Phase2 only. This is a numerical infrastructure result,
NOT a new physical approximation, fitted model, raw/adjusted improvement
or deployment candidate. Official bank329926 is unchanged.

## Cause, reproduced independently

Installed Torch2.7 MPS scalar clamp_min bound reuse was the dominant late
forward failure. In same-shaped tensors the first small scalar bound was
reused for later1e-10/1e-12/1e-30/0 bounds. Three tiny widths and different
bound orders reproduced it: first correct, all subsequent different bounds
wrong. Explicit tensor bounds were correct. This is an observed local
backend defect; no upstream-version claim or public forum post was made.

In the model, a1e-12 post-variance floor became1e-10, giving post sigma1e-5
instead of1e-6 for negative-variance neurons. Post-correlation feature
columns27/28 differed by up to.556, changing the learned head by~1e-5.
Resetting every incoming MPS state to rounded CPUf64 still reproduced this;
the moment host itself differed<8e-7. Centered scalar/pair algebra alone
did NOT fix the GPU failure. The old failed records remain unmodified.

## Versioned repair and forward certificate

ExplicitBoundBankFullCard in centered_bank_autodiff_v3.py preserves all
carrier mechanics, selectors, frozen P0 and actual scalar finalQ. It uses
the previously certified centered identities and a fixed32-node native
Gaussian CDF, since direct MPS erfc is unsupported. Every scalar clamp_min
in the numerical maps, prefix, feature builder and learned heads becomes
an explicit tensor-bound where expression, preserving equality derivative1.
No CPU fallback or f64-on-MPS claim. Thirteen prefix/head clamps and one
marginal clamp are transformed; old modules/payloads are untouched.

Full1024x16 dynamic MPS bank versus frozen NumPy baseline:

| Net | Max mean difference | Final difference RMS | Memberships |
|---|---:|---:|---|
|35|1.39839787394e-6|2.38354724806e-7|all30 equal|
|100|1.39716386938e-6|2.63501890630e-7|all30 equal|

Both pass the unchanged2e-6 maximum gate.1.988seconds for both in the
audited replay; process RSS890,011,648B is NOT total GPU memory.
Output SHA48736c9b55493164ab012a54ad73fb29efebc314fd6b341eac9771e71cfc96a0.
CPUf32 BLAS baseline still fails7.5e-6; do not claim that path was repaired.
CPUf64 centered control passes~2e-7. Small scalar/pair value and reverse
gradient certificates pass CPUf64/f32 and MPSf32, including alpha0.

## Local derivatives versus finite steps: keep the distinction

One fixed random final-layer projection per net, all16x1024 mean-control
VJPs, original CPUf64 bank versus repaired MPSf32:
relative2.65834453e-6 /5.39764156e-6. Max per-layer relative9.701e-6.
All finite. This is strong local point-Jacobian agreement with an already
independently certified CPU derivative, not global smoothness.

The dynamic finite-step composite remains FAILED: some layer13 perturbations
at the fixed2e-4/1e-4 amplitudes change selection membership. A separate
zero-point fixed-membership AND order certificate also remains FAILED on
rank100large (5.7781% FD-vsAD; half3.6287%). Rank35 both pass.
No amplitude, direction, tolerance or old pass flag was changed.

The same steps through the ORIGINAL CPUf64 bank explain the latter issue:
rank100 CPU FD-vsAD6.0065%/2.8912%. CPU-versus-MPS finite-step discrepancy
is only.1703%/.1211% on35 and.2150%/.7173% on100, all below the fixed1%
backend-consistency diagnostic. Thus the finite step itself is not a good
linearization in that case, even with the original CPU card. No invented
GPU derivative fix is indicated by that discrepancy. This does not promote
the failed finite-step gates or authorize a blanket training sweep.

Peak mixed-worker process RSS7,866,630,144B; largest observed GPU driver
allocation6,504,054,784B, not independent additive peak totals. Swap stayed
5910.50MiB. All workers terminated. No target means loaded by these tests.

## Research implications and limits

A fast native numerical implementation and a trustworthy local VJP now
exist. Any new raw lever still needs a distinct, preregistered hypothesis,
actual dynamic forward validation and held-out accuracy gates. This does
not reopen the failed CPUf64 readout/encoder-direction pilots: those fits
used correct derivatives and independent NumPy evaluation. The MPS frozen-K3
shadow may have been affected but also omitted real K3 transport; do not
declare it accurate from this repair or automatically rerun it.

The tempting generic learned bidirectional pass was also checked against
the wider Phase2 record: public_full688_backward_adjoint_state_cv_v1 already
executed and closed. LANE_direct_diagram_adjoint.md section121.2 records
3.23376600912e-8 versus3.23349582295e-8,330wins/358losses. It must not be
launched as if novel merely because its design README sounds unexecuted.

Implementation hashes:
- centered_bank_autodiff_v3.py45c88f1e73d694f0781c4f0a01053511addbe465a6746ef11ac38378b093983c
- centered_torch_numerics_v3.py18df077006603d1b434216b2a4ffebc1b1d1e631983163fa0a9213eb0a11412e
- tensor_clamp_min_v1.pyc68768a7dc17a09c6e9105817a1b9d958b2409d076dab16f5ea427d5439e9c6b
- gaussian_cdf_quadrature_v1.pyf6f96fbc8f47bd4ffab93ba9cafc805b2ea79435f5a2106649d94a3bc0754398
