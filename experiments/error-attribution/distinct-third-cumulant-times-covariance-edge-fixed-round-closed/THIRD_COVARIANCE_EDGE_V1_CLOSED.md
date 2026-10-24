# Distinct K3 times covariance edge: fixed round closed

Phase 2 only, September 5, 2026. No estimator, fit or score.

The three primary augmented diagrams pass at widths8/16: coefficient
relative errors 2.13e-16 / 2.06e-16; full and low-rank query errors below
7.1e-19. This is a genuinely omitted term, but omission does not establish
useful alignment with the current error.

The first fixed wide cell was rank35 L12 -> next pre-K21 L13, repeated
indices0,32,...,992, all1024 partner indices. Current source-v2 coherent8M
oracle previous means/covariances; frozen source inputs before reading
targets; original covariance target byte-exact. Pre/post CP20352/21568.
Existing K21 diagonal discrepancy on queried coordinates was .03784%.

Natural coefficient1 gives unweighted plane-error MSE ratio1.00254042;
leading covariance-consumer-weighted ratio1.01336446. Weighted correction
cosine with the missing plane is -.04847, versus the predeclared >=.4
gate. Correction norm is7.69% of weighted baseline error. Full source
fails the information gate, before any approximation is relevant.

Fixed top4 covariance-edge approximation also fails: unweighted source
relative error83.30%, weighted84.76%, versus15% gates. Top4 covariance
trace fraction14.576%. This differs from projecting the whole third tensor,
but the tested edge truncation is still inaccurate for this source.

Stop: no rank100, other layer, q8/q16 sweep, damping, fitting, rollout,
candidate or submission. This closes the specified source test, not all
third-order improvements. Whole run11.036s, peak4.702GB. All jobs complete.

Receipts: `third_covariance_edge_certificate_v1.json`,
`third_covariance_edge_r35_l12_v1.json` and `.npz`, plus all32 query NPZs in
`third_covariance_edge_r35_l12_progress_v1/`. Preserve all artifacts.
