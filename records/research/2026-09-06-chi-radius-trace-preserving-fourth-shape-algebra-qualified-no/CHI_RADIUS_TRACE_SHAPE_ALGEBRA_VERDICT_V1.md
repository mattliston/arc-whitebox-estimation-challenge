# Chi-radius trace-preserving fourth shape: algebra qualified, no value

September6,2026. Phase2. No target arrays, wide test, fitting or candidate.

The exact coefficient/plane closure in CHI_RADIUS_TRACE_SHAPE_DESIGN_V1.md
is not identical to the executed noncentral metric, Sym(C,D), or direct
angular-card experiments. Its distinct component is the fixed
Sym(mu_A,T_A) fourth tensor, with the angular scalar chosen only to
preserve the current pre-frame sum of the bank's K22.

An important algebraic cancellation is certified: the known pure radial
`(2/n)Sym(C_A,C_A)` contribution is absorbed completely by that scalar
fit. The final new tensor is `eta Sym(C_A,C_A)+R`, where R contains the
mean–third and the fixed mean–covariance terms. It is not an additional
independent radial fourth signal.

The fixed four-dimensional dense certificate passes:

- K22/K31/marginal fourth maximum error1.11e-16.
- Trace relative error1.81e-16; angular scalar recovery1.39e-17.
- Both plane diagonals agree within5.55e-17.
- Radial cancellation and zero-mean/zero-Gaussian-input identities exact.
- The new nonlocal K4-to-K3 source identity agrees within1.69e-21.
- Runtime0.0182seconds, peak38,125,568bytes; no worker remains live.

Receipt: chi_radius_trace_shape_certificate_v1.json.
Helper SHA73de07f079c969604920158131df5ebaa7a3303adc44c898cd6ebba90c63125f.

## Cost boundary, not a production claim

The plane constructor and fixed trace fit require only O(n^2) work.
An80*n^2/layer conservative elementwise estimate gives1.26e9 FLOPs over
15 pair layers, below.3% of a490e9 bank bill; no measured flopscope or
runtime result exists. Exact supplied K31 must replace the old radial
K31 formula in every pair response that consumes it.

However, the new Sym(m,T) also emits a one-leg derivative of the ENTIRE
third tensor and a new vector–matrix trace term. A straightforward exact
mixed-CP carrier needs an additional rank-R factor push and mixed plane
products, O(n^2 R), plus ownership of new trace-source descendants.
No cheap complete full-carrier schedule is established. Replacing the
gate by g+p*d*m would introduce unowned higher-order terms and must not
be presented as exactness. Keeping the old CP/trace feed unchanged is
an explicitly selective pair-only intervention, not the complete
Gaussian moment closure induced by the new fourth tensor.

A fixed local plane-only information test could be separately designed
and labeled as such. This audit does not launch one, assert a value gain,
or license an angular rerun, source/gate sweep, fit, port or submission.
