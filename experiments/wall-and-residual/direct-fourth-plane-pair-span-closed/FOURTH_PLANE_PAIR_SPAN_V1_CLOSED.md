# Direct fourth-plane pair span — CLOSED

2026-09-06 03:57:06 PDT, Phase2. One-cell coefficient-span oracle only;
no new sampling, cross-net training, state write, or candidate.

The fixed14 analytic features have numerical rank9 on this actual pre13
state (five dependencies, no rank search). Retained span reconstruction
6.18e-16 relative. Small adjoint/dense ridge certificates passed. All four
full-width ridge normal solves converged in71/80 iterations with residuals
6.44e-8..9.79e-8, within the unchanged1e-7 gate. No solver sweep.

Independently fitted A/B coefficients use the already qualified two8M
streams at rank35/pre13. All six response targets are byte-identical to
the preceding trace oracle receipts. Replica signal/noise is37.9..47.3.

| Object | Residual/baseline signal cross-energy |
|---|---:|
| Standardized K22 plane | 0.9445027152 |
| Standardized K31 plane | 0.8889136689 |
| Standardized marginal k4 | 0.8575401834 |
| Isolated covariance response | 0.7899170215 |
| Isolated next-mean response | 0.8830418241 |

Required covariance and next-mean ratios<=0.20, with both ordinary
cross-replica errors<=0.30. Actual ordinary ratios are0.7929/0.7967 for
covariance and0.8893/0.8827 for next mean. Decisive FAIL, no expansion.

This does not establish an unregularized universal capacity ceiling, or
cross-network generalization. It rejects this fixed analytic pair-field
span as an information route worth expanding. No feature/power/rank/
regularization/iteration/sample/net sweep follows. No raw/adjusted score.
Elapsed1.368s; peak hostRSS686,899,200B. Bank329926 remains unchanged.

Receipt fourth_plane_pair_span_verdict_v1.json/.npz. Helper SHA256
b5fc7af86f8590c3fa85cff277244c855d3c703b5f36ae5b5ba59f291245d0dc;
script6813da6cacb0f746b4489950b6cb9403a52c6e0b16a7bc0538103340ef669a2f.
