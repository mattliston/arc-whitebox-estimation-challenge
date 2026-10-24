# Multi-state physical residual pilot

The selected model has six unnormalised physical residual states.  Its main
state is transported by the exact local tangent `Phi * W`; three latent
states also use `Phi * W`, and two use the centred operator
`sqrt(n/8) * (W**2 - 2/n)` with bounded local gates.  Local innovations carry
the physical `sigma * layer_residual_scale` factor.  Reported corrections at
L0 and L1 are exactly zero, while the L1 latent seed is multiplied by 0.1.

Protocol: train ranks 0:20, select only on ranks 20:25, then evaluate the
frozen checkpoint once on ranks 25:35.  The H32 checkpoint at epoch 224 was
selected before held targets were opened.

| split | host final MSE | corrected final MSE |
|---|---:|---:|
| development 0:20 | 9.0783e-8 | 3.5900867850e-8 |
| validation 20:25 | 9.0631e-8 | 3.6557376906e-8 |
| held 25:35 | 8.9395150090e-8 | 3.8573099343e-8 |

The held result removes 56.85% of the host MSE but misses the requested
`3.82e-8` threshold by `3.73e-10` (0.98%).  It is effectively tied with the
prior one-state H48 (`3.8579886884e-8`) and H96 (`3.8597182203e-8`) pilots.
The extra state capacity is real on validation--the H48 explicit model's
main-state-only ablation is `4.6662e-8`, versus `3.7335e-8` with its latent
block--but it does not produce a material held improvement.  No analytic
blow-up row is present in ranks 25:35, so tail-included and tail-excluded
scores are identical for this pilot.

Artifacts:

- `explicit6_h32_accum.pt`: frozen selected checkpoint
- `explicit6_h32_accum_DEVELOPMENT.json`: development and selection trace
- `explicit6_h32_accum_HELD.json`: one-use held result
- `run_pilot.py`: deterministic streaming trainer/evaluator
