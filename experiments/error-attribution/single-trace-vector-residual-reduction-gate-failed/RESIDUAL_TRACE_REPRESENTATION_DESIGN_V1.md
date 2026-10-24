# Can the residual sector reduce from four matrices to one vector?

Phase 2, September 5, 2026. No new sampling or deployable fit.

The one-high residual oracle passed a strong single-cell gate. Before
designing an expensive four-matrix sidecar, test a strictly smaller shape:
Sym(b,C), meaning b_i C_jk+b_j C_ik+b_k C_ij, with its coordinate-repeated
entries removed. C is the already-frozen post-layer covariance. This does
not repeat the closed whole-third/cold-CP trace replacements: the complete
parent stays fixed; only its error is approximated.

Fit b to the four source-slice residual matrices separately in stream A
and stream B, never to next-plane scoring rows. This is an ORACLE
representation fit, not training a deployable network. Exact linear least
squares with no damping/regularization sweep. Certify the normal operator
against an explicit small-width design matrix first. Evaluate the fixed
next W query and independent stream using the previous opportunity gates:
weighted signal error ratio<=.7, cosine>=.6, both cross-stream ratios<1.

No extra samples, metrics, ranks, layer selection or basis tuning. Failure
closes this vector/covariance trace reduction, not the already-passed
four-matrix residual opportunity. A pass still needs a target-blind source
that predicts b and a complete cost/recursive/independent-score evaluation.
No candidate or submission is licensed by an oracle fit.
