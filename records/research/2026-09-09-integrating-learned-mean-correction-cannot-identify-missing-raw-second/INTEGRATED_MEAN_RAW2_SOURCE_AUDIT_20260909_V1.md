# Integrated learned mean response: source-only no-go

September 9, 2026. Phase 2 only. Root and fresh_efficiency inspected the
current P1 source, not new targets or numerical outputs. This is not a
numerical closeout of an executed integrated-head estimator.

For a fixed centered scalar law, d_mu E[ReLU(X+mu)^2] = 2 E[ReLU(X+mu)].
The current unfloored cumulant scalar formula already obeys this identity:
m = mg + k3*e/6 + k4*f/24 and raw2 = raw2g + k3*d/3 + k4*e/12.
P1 then supplies delta_P while preserving the raw second-moment matrix.
The corresponding diagonal covariance update is -2*m*delta_P-delta_P^2.

An identity-consistent ADDITIONAL marginal raw2 correction would instead be

    delta_raw2(mu) = 2 integral[-infinity, mu] delta_P(t; fixed centered law) dt.

It is not identified by delta_P(mu), a local derivative, 2*mu*delta_P, or
the alternative convention that holds centered covariance fixed. Existing
two-SiLU P1 features include post-covariance row statistics, post-moments,
global mean and recurrent messages. Clipped features and biases do not
enforce an integrable negative tail or a calibrated virtual-shift law.
Exact shifted feature reconstruction cannot simply reuse the unshifted
post-covariance statistics. Reusing them defines a new surrogate premise.

For Q quadrature nodes at each of layers1..14, an implementation must pay
Q times shifted covariance-row feature and encoder work, plus integration,
copies/guards and any changed downstream trajectory. Incoming message
projections could be reused. No additional CP contraction is inherently
required, but this is not a zero-cost local head rewrite. The scalar
identity also supplies no off-diagonal raw moment or coherent k3/k4 update.

Closest preserved receipts, all different from an integrated mean curve:

- CENTERED_FEEDBACK_ROUND_V1_CLOSED.md: matched two-net raw gains about
  1.72%/1.69%, failed its original both-net8% gate.
- MEAN_RESPONSE_TRANSPORT_PILOT_V1_CLOSED.md: matched20/4 raw ratio
  1.00525677 and0/4 wins, despite useful derivative/gate proxy results.
- CUMULANT_TILT_MEAN_RESPONSE_V1_CLOSED.md: value-weighted oracle error
  ratio1.1402001, failed; this was a distinct covariance-response ansatz.
- QUADRATIC_RESPONSE_MOMENT_IDENTITY_AUDIT_2026_09_08_V1.md: local
  response-jet derivation, not a calibrated integrated-P mean curve.

Current owner: current_d50_native_objective_p1_v1/build_v1/estimator.py,
SHA0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435,
functions features, gnn_step, _nonlinear_pair_batch and _predict_binary.

Decision: no native write, fit or endpoint run selected from this identity
alone. Missing evidence is a boundary-conditioned virtual-shift mean curve
whose integral predicts the current raw2 error, together with a complete
cost margin. The original receipts remain unchanged. No new score or bank.
