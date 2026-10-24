# Phase 2 forum research audit

Snapshot date: 2026-08-23.  Forum results below are participant evidence unless
explicitly identified as organizer statements.

## Organizer status

- [Phase 2 official start date and details](https://discourse.aicrowd.com/t/phase-2-official-start-date-and-details/18196):
  the dataset and software were made public, but evaluator security review was
  still pending and submissions were not open.
- [Phase 2 starting?](https://discourse.aicrowd.com/t/phase-2-starting/18186):
  the organizer said the launch and submission deadline would shift.  The old
  deadline must not be treated as final.
- The pinned software and dataset are therefore our public-prelaunch authority;
  the eventual launch announcement can supersede them.

## Sampling-free propagator and every reply

[Topic 18177](https://discourse.aicrowd.com/t/bitter-lesson-pilled-sampling-free-propagator-yielding-1-5e-5-raw-mse/18177)
contains four posts, all reviewed.

Keenan's Phase 1 system carried mean/second moment, an `O(N^2)` compressed
third-cumulant plane, and a recurrent per-neuron latent trained through the
whole 32-layer recursion.  It achieved `1.50e-5` raw MSE.  Its expensive
learned attention closure was nearly inert; removing it produced a leaner
bit-identical system.  Even true mean and covariance at every layer left an
`8.1e-7` Phase 1 floor, identifying state fidelity rather than lack of model
capacity as the remaining problem.

Two replies thanked Keenan for the public datasets.  The final reply supplied
the width/depth crossover experiment we need for Phase 2.  It compared an
uncorrected order-2 Hermite/Mehler chain with FLOP-matched Monte Carlo and used
median per-network MSE ratios because both methods have heavy-tailed errors.

Analytic MSE / matched-MC MSE (`<1` means analytic wins):

| depth \ width | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.11 | 0.05 | 0.02 | 0.01 | 0.01 |
| 8 | 0.49 | 0.21 | 0.11 | 0.05 | 0.03 |
| 12 | 0.61 | 0.60 | 0.24 | 0.14 | 0.07 |
| 16 | 0.64 | 0.83 | 0.37 | 0.21 | **0.12** |
| 24 | 2.70 | 1.20 | 0.74 | 0.44 | 0.25 |
| 32 | 3.24 | 2.68 | **1.04** | 0.64 | 0.34 |
| 48 | 2.42 | 1.51 | 1.07 | 1.33 | 0.78 |

The Phase 1 point was at median parity.  The intended Phase 2 point favors the
zero-fit analytic chain by about 8.3 times.  This is the strongest reason to
put analytic closure ahead of learned feedback in the new graph.  It remains
an external result that we will independently replay.

## Combined frontier thread and every reply

[Topic 18193](https://discourse.aicrowd.com/t/phase-1-solution-ideas-for-phase-2-and-the-honest-public-frontier-all-methods-combined-and-explained/18193)
contains five posts, all reviewed, plus the linked field guide.

The important corrections and shared conclusions are:

- Late activations are dominated by a common mean, but centered residual
  covariance is not effectively rank two.  At the Phase 1 scored layer, about
  56 of 256 directions were needed for 99% of centered variance and 155 for
  99.99%.
- Exact post-ReLU marginal mean and standard deviation at every layer did not
  solve Phase 1 when cross-neuron dependence remained approximate.  The
  reported floor was `7.07e-7` with a higher-cumulant readout.
- Exact final mean/variance alone gave about `9.33e-7`; adding exact K3/K4 to an
  Edgeworth readout gave roughly `1.5e-8` to `2.2e-8`.  Thus the scalar readout
  can be nearly solved while recursive joint-state transport remains wrong.
- Batch whitening was the article author's highest-priority Phase 2 sampling
  experiment, with reported Phase 1 gains of roughly 2.06--2.34 times.
- Antithetic results conflict across participants; no automatic promotion.
- The corrected spectral analysis strengthens our existing conclusion: a
  small participation ratio does not authorize a tiny covariance carrier.

These are Phase 1 measurements.  Wider/shallower Phase 2 networks can move the
dependence floor, so every numerical conclusion is a hypothesis to re-test.

## Other recent threads incorporated into the migration

- [Technique census](https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157):
  analytic closure was not competitive at depth 32; this is now a historical
  conclusion, not a Phase 2 veto.
- [Research atlas](https://discourse.aicrowd.com/t/phase-1-research-atlas-how-i-reached-public-16-and-why-the-next-56-experiments-failed/18159):
  our Phase 1 causal and tensor experiments remain the reusable evidence map.
- [Method-class ceiling](https://discourse.aicrowd.com/t/write-up-submission-326725-measuring-the-ceiling-of-a-method-class-how-we-closed-54-families-of-estimators-without-building-them/18182):
  motivates oracle ceilings and meter-first gates rather than blind variants.
- [Korobov hybrid](https://discourse.aicrowd.com/t/our-current-approach-korobov-hybrid-trunk-e-offline-ridge-residual-corrector-g-16-graded-gain-what-we-learned/18185):
  supports a measured hybrid comparator lane.
- [Meter/residual trade-off](https://discourse.aicrowd.com/t/measured-the-metered-flop-vs-residual-wall-time-trade-off-and-what-it-implies-for-a-phase-2-cap/18184):
  Phase 1 wall-time economics are superseded by the new hard 0.4-second
  residual gate, but the call-overhead lesson remains relevant.
- Phase 1 reports 18183, 18188, 18181, 18180, 18179, 18176, 18175, 18174,
  18171, 18170, 18169, 18166, 18164, and 18163 were included in the recent-topic
  inventory.  Their numeric results remain comparison ideas, not Phase 2
  authority.

## Decision

Start Phase 2 with a clean analytic ladder, not a transplanted learned model:

1. diagonal Gaussian;
2. order-2 Hermite/Mehler;
3. CP3 and K21;
4. centered TT3 residual with a newly selected rank and non-cubic PSD path;
5. teacher-state attribution;
6. end-to-end learned correction only for the residual that remains;
7. whitening/QMC at matched official cost as comparator or hybrid support.

