# Phase 2 full-budget pure-closure directive

Status: active research authority, recorded 2026-08-23.

## Objective

The primary scientific objective is a deterministic, sampling-free analytic
closure with final-layer raw MSE below `1e-8` on width-1024, depth-16 Phase 2
networks. This is the target for the closure itself. Sampling, Kerdock/MUB
cubature, control variates, learned residuals, and structural compaction are
downstream tools only after the closure reaches the target class.

## Resource authority

Use the complete legal Phase 2 envelope when it improves raw closure accuracy:

- metered FLOPs per MLP: `2^41 = 2,199,023,255,552`;
- solution-process memory: less than `8 GiB`;
- wall time per MLP: at most `120 s`;
- residual wall time: at most `0.4 s` and only for control/bookkeeping;
- all numerical submission-time work through FlopScope 0.12 operations.

The earlier `219,902,325,555`-FLOP screen was only the 10% score-multiplier
floor. It is retired as a closure-design ceiling. A branch closed only because
it exceeded that floor is reopened for resource and accuracy review under the
full `2^41` budget.

## Measured starting point

The strongest measured pure closure is the cap-16384 full-birth CP3 carrier
with recursive reported-mean carry:

- final-layer raw MSE: `9.340196551564291e-7`;
- all-layer raw MSE: `5.293963582531012e-7`;
- metered FLOPs per MLP: `620,090,560,514` (`28.198453970344417%` of `2^41`);
- the carrier preserves every CP birth term through layer 16, so residual error
  is not explained by CP rank truncation alone.

Layer 1 is analytically exact up to the reference noise floor. Layer 2 remains
below `1e-7`; layer 3 is the first material failure. The full-rank mean-carry
result therefore identifies missing joint-state transport—especially K21 and
quartic action—not insufficient scalar mean recursion.

## Reopened closure families

Re-evaluate every branch previously closed only by cost, including:

1. full-K21 forward and transpose actions at high CP rank;
2. correlation-interior or eigenspectral PSD correction without global
   off-diagonal shrinkage;
3. diagonal and structured off-diagonal K4 transport;
4. centered TT3 residuals and quartic TT/CP residual carriers at ranks selected
   under the full budget;
5. future-weight action states that store only contractions required by the
   next affine/ReLU map;
6. fused CP3 + K21 + K4 consumers that preserve exact layers 1 and 2 and begin
   correction at layer 3;
7. high-accuracy covariance repair paths formerly rejected by the 10% floor.

The catastrophic cap-512 recursive K21 result closes only its fixed global
correlation-shrink consumer. It does not falsify K21 transport. Any successor
must separate the raw third/fourth-order correction from PSD enforcement and
must preserve the corrected diagonal exactly.

## Promotion rule

Rank closure variants by raw MSE, not adjusted score, while they remain within
the hard Phase 2 limits. Do not promote a sampler, learned observer, ridge
readout, or hybrid as the primary result. After an independently validated
`1e-8`-class closure exists, use it as the pathwise control variate for a
separate Kerdock/structured-sampling residual stage.

