# Fixed-count cohort retention: closed on the information prerequisite

Phase 2, September 6, 2026, 11:16:46 PDT. No target access, samples, carrier,
fit, changed estimator, score or submission. The worker finished in 3.9366s;
a subsequent `ps` check found no remaining worker. MPS was released to root.

Uniform fixed-count sampling within each birth cohort is not independent
Bernoulli sampling. Its exact finite-population local output variance and
the optimistic continuous allocation across cohorts were certified before
the wide screen. All 120 tiny joint patterns agree with the formula to
4.40e-16 relative; the continuous optimum satisfies the interior KKT
condition to 2.22e-16 on the wide cases.

One coherent repaired-MPS current-bank trajectory per rank supplied BOTH
pair norms and cohort sums. The older NumPy cohort sums were deliberately
not combined with MPS norms. Same frozen P0/Q, complete signed pairs,
unchanged nonnegative final clipping and the full diagonal-plus-rank-one
final k3 Jacobian. Full-forward max differences 1.398398e-6 / 1.397164e-6
pass the unchanged 2e-6 gate, all 30 active/birth memberships agree, pair
identities are 5.61e-16 / 5.25e-16, and J reverse errors are
2.34e-8 / 2.05e-8.

Results (rank35 / rank100, current bank, local frozen final consumer):

- Half retained totals: 5968 / 5936 of 11936 / 11872 pairs.
- Best CONTINUOUS cohort allocation: added MSE 2.11939241137e-8 /
  2.00296372747e-8. Both FAIL the fixed <=1e-9 gate by over 20x.
- Uniform half per cohort: 8.82130343779e-8 / 7.92292938489e-8.
- Coherent independent uniform half: 8.82063052238e-8 /
  7.92275660941e-8, reproducing the previous independent receipt.
- The removable common cohort component is only 0.1515% / 0.1598% of
  total squared pair effects. Fixing the count therefore adds essentially
  no cancellation benefit.

All acquisition, earlier-layer errors, feedback, runtime and costs were
FREE. Integer allocation cannot beat this continuous optimum. Close this
uniform-within-cohort fixed-count proposal: no count/budget sweep, actual
sampling, coadaptation, bank, panel or upload follows. This does not prove
a bound on arbitrary direction-matched dependent rounding or deterministic
compression, and is not a new reason to reopen those without new evidence.

Separate shape-only receipt `uniform_half_cp_shape_cost_v1.json` confirms
the potential scale, not feasibility: uniform half per cohort rounded UP
to 32 would retain 51.206% / 50.674% of pairs and save an optimistic
39.399% / 39.755% of measured first-context bank FLOPs, preserving the
already-banked shared newest-pair push. Accuracy failed before a carrier
could realize that saving. Acquisition and changed non-GEMM costs are
omitted; this is neither measured candidate utilization nor runtime.

Files: `cohort_fixed_count_thinning_v1.py`, tiny and wide JSON receipts,
wide NPZ, and the preregistered design. Source SHA256
9fbc16ffe12e019b2b6f320e143017a69ad5cbbaa011dee352d40d82cc2663ed.
Wide NPZ SHA256
da7f6948221839b7966653c082f85eba00121b8e4d08f1e184be753a407f5055.
Peak process RSS 2,071,281,664 bytes; largest observed GPU driver memory
6,478,872,576 bytes. These are not independent additive peaks.

