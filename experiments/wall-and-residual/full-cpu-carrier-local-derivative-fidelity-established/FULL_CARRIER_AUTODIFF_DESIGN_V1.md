# Full-carrier automatic differentiation: fidelity before optimization

2026-09-05, approximately 12:52 PDT. Phase 2 only. Target remains official
adjusted below1e-9; no milestone or new upload. No forum posting.

The old mean/covariance shadow froze K3/K4 and failed early-layer response
fidelity (approximately30% error). Its late-only training pilot was flat.
That closes that approximate training route, not differentiation of the
actual signed-cube, coordinate-star and covariance-trace third carrier.

Implement the complete current NumPy full-pair P0 host in torch, including
the CP factors and coefficients, star, radial fourth state, b trace vector,
all pair moments and the pre-correction GNN features. The CPU reference
retains float64 scalar/moment arithmetic and float32 carrier products.
An optional MPS float32 implementation is only an offline derivative model;
it must earn its own response certificate. Nothing changes in deployment.

No labels are needed for this round. First compare nonlinear maps at small
synthetic widths and compare current full-host trajectories at ranks35/100.
Then reuse the archived target-free actual-carrier responses to fixed
layer4/9/13 mean perturbations, at amplitudes2e-4 and1e-4, rather than
regenerating them. Birth/live top-k selections follow the same rounded
threshold rule and are piecewise constant; report any changed selection
counts. No gradients are claimed across a selection boundary.

Gates before training: synthetic map relative error<1e-10 in CPU float64;
CPU bank-path maximum mean discrepancy<2e-6; response relative error<=10%
and finite-difference half-step disagreement<=5% for all six archived
responses; one scalar adjoint/finite-difference check per net<=5%.
The fidelity gate, not baseline closeness alone, determines whether a
full-depth training pilot is justified. If it fails, diagnose the actual
equation/precision problem before any fit. No speculative expanded fit.

At most one differentiating card process while memory is measured; use
layer checkpointing to avoid retaining every pair-kernel intermediate.
Stay within the established nice10/CPU/MPS resource limits. Preserve failed
versions and write new receipts. No training targets, sealed data, MLP
sampling, candidate changes, commits or submissions in this fidelity round.
