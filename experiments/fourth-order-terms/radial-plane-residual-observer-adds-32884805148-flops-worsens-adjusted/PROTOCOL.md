# Fixed radial-plane residual mean observer

Phase 2 only. Prospective before any numerical work for this study. The
parent is exact official 331684, SHA256
facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
Its official adjusted score is 3.5844144212297072e-9. The 4e-9 bank is
retained; 3/2/~1e-9 and the full goal remain unmet. Qualified future uploads
are already authorized. No changes to the official source or archive.

The current nonlinear map computes a complete postactivation K22 matrix,
then reduces it to a radial scalar. Test one natural-strength mean observer
of a specified part discarded by that reduction. It runs alongside the
unchanged physical/learned trajectory, with no fit, amplitude selection,
covariance write, parameter change, or feedback to the parent.

For each original map at layers 0..14, BEFORE the edge source and mean hook,
let S=sym(C_map), Q=sym(post22_map), v=diag(S), and define

    r = 3 sum(Q) / [(sum(v))^2 + 2 sum(S*S)]
    D = Q - (r/3) [outer(v,v) + 2 S*S].

This helper's own symmetric projection leaves sum(D)=0 in exact arithmetic.
It does not change the parent's radial coefficient or omitted-response
radial correction. Q and S are approximate map outputs, not true moments.
Cast D to FP32 once; no clipping, PSD repair, new normalization or weights.
Interpret D as the symmetric fourth tensor whose iiii and iijj entries are
D_ii and D_ij, with all 31/211/1111 entries zero. The diagonal convention
is important: there are six permutations for iijj and one for iiii.

At the next affine transition with the actual FP32 weight W, compute

    delta4_j = 3 sum_i W_ij^2 (D @ (W*W))_ij
                 - 2 sum_i D_ii W_ij^4.

Use ONE classical FP32 FNP matrix product D@(W*W); compute the final
products/reductions in FP64. All formation, casts, reads, products and
copies are inside predict and charged. No production NumPy/SciPy calls.
This is an exact real-algebra ray of the rounded coordinate-pair tensor,
up to the declared FP32 product arithmetic. It is not a full fourth-state
carrier or a new 31/211/1111 supplier. For isotropic Gaussian weight
queries the unrounded ray has zero mean because sum(D)=0; it retains
query-specific variation that the radial scalar discards.

Initialize the observer mean shift to zero. For layers 1..15 use the
original parent alpha, sigma, pdf and gate:

    shift_l = gate_l * (W_l.T @ shift_(l-1))
        + [(alpha_l^2-1)*pdf_l/(24*sigma_l^3)] * delta4_l.

The first nonzero step omits the known-zero matvec. The matvec and scalar
response use FP64. Report the original parent's per-layer mean plus this
shift, with its original final nonnegative/FP32 conversion. The first
row is exact. The physical means, covariances, CP/star/trace banks, births,
selectors, hidden states and learned heads all continue from the original
parent state. This Gaussian-gain first-order mean observer omits covariance,
GNN and higher-cumulant feedback; it is not the full native tangent.

Distinct prior scope: the older G29 fourth-plane chain replaces the fourth
state with plane-supported transport and loses other collision sectors.
This observer retains the entire original parent and adds only its declared
residual ray to a separate mean recurrence. Older scalar/fourth-tree,
T-times-C, trace, graph, sphere and learned mean-observer failures remain
unchanged. A later complete joint carrier is not implied by this test.

Qualification before reading targets:
- Preserve the whole parent byte prefix. Append the helper and exact copied
  function derivatives: two maps expose post22 by reference, and the final
  predictor calls the helper and adds only the output shift. Prove every
  whole-function edit inverse and no changed learned literals or formulas.
- Six fixed tiny signed/nonsymmetric/zero/unequal-variance cases certify D,
  radial-plane reconstruction, exact double-trace cancellation, explicit
  dense fourth tensors, rays, positive coordinate rescaling of a supplied
  tensor, output permutation and the independently expanded mean recurrence.
  Exact FP64 reference errors <=2e-11; actual FNP ray/recurrence relative
  L2 errors <=3e-5 for nonzero references (absolute for exactly zero).
  Repeated bytes/bills and input preservation required.
- Disabled native35 must equal all16 parent output bytes and complete bill.
  Enabled35/100/35 and independent cold100 must preserve every underlying
  parent output byte and exact parent bill after subtracting measured helper
  work, preserve own repeat/warm-cold output and shift bytes, and retire all
  numeric caches. Compare to authenticated matching cold/warm parent calls.
- Research wrappers independently check each source, ray and mean recurrence
  in NumPy FP64, without feeding those results back to the candidate. Require
  source relative <=2e-6, each ray and recurrence <=3e-5, finite outputs,
  exact source/ray/shift repeats, and helper overhead <=10% of each full
  parent bill. Such research time is not production client qualification.
- Seal all predictions, bills and qualification before decoding ordinary
  existing 1B reference means for reused development networks35/100.

Score the actual candidate raw mean-squared error times its complete cold
max(.1,F/2^41), using scalar fsum plus an independent NumPy reduction.
Advance only if BOTH adjusted ratios are <1 and the pooled adjusted ratio
is <=.99. No raw-only substitute for this objective. A pass requires a
separately frozen wider panel and production/client/contract/package checks
before upload. Failure closes this fixed observer: no amplitude, layer,
source, precision, gain, training or blend rescue. No official forecast.

One serial nice10 BLAS1 group, 8GB RSS limit, existing 256MiB emergency
swap cutoff and ZERO added swap for qualification. Unit90s, disabled180s,
enabled450s, cold100180s, qualification60s, score60s. Artifacts <=15MB;
no retained wide trajectory, GPU, new samples, downloads or external writes.

