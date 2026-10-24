# Sampling-free closure accuracy target below 1e-8 raw error using up to the full 2^41 budget

2026-08-24T03:27Z · decision · adopted · derived

The research authority set the primary objective as a deterministic analytic closure with final-layer raw MSE below 1e-8 on 1024x16 networks, using the whole legal envelope (2^41 FLOPs, 8 GiB, 120 s, 0.4 s residual) when it improves raw accuracy. The earlier 219,902,325,555-FLOP screen was only the 10% score-multiplier floor and was retired as a design ceiling, so every branch closed only by that cost was reopened (full-K21 actions at high CP rank, spectral PSD correction, structured K4 transport, TT residual carriers, future-weight action states, fused CP3 + K21 + K4 consumers). Closure variants were to be ranked by raw MSE, not adjusted score, and no sampler, observer or hybrid promoted as the primary result.

Recorded measurement:

    measured starting point: cap-16384 full-birth CP3 carrier with recursive reported-mean carry, final-layer raw MSE 9.340196551564291e-7, all-layer 5.293963582531012e-7, 620,090,560,514 metered FLOPs per MLP (28.198453970344417% of 2^41); layer 1 exact to the reference noise floor, layer 2 below 1e-7, layer 3 the first material failure

## Evidence

- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L7](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L7)
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L22](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L22)
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L31](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L31)
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L64](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L64)

<details>
<summary>All result files and records (1)</summary>

- [FULL_BUDGET_CLOSURE_DIRECTIVE.md](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md) · record

</details>

## Related work

- ← rejects: [Width-equivariant residual observer with 32 channels over four weight powers: design deferred](../2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/README.md)
- ← enables: [Full-budget closure design: six mechanisms projected at 1.9092T FLOPs; dense third-order transport excluded](../2026-08-24-full-budget-closure-design-six-mechanisms-projected-1-9092t/README.md)
- enables: [Full-birth third-cumulant carrier with recursive mean feedback: final raw error 9.340196551564291e-7](../../../experiments/third-order-planes/full-birth-third-cumulant-carrier-recursive-mean-feedback-final/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-sampling-free-closure-accuracy-target-below-1e-8-raw) · [Copy provenance](../../../PROVENANCE.json)

