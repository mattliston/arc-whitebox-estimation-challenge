# Fixed d50 all-P coadaptation attempt — not admitted

2026-09-09 UTC / 2026-09-08 PDT. Phase2 only.

Confirmed official bank remains #330390: adjusted4.53886349950395e-9,
raw2.120838555441651e-8. No new estimator, fit result or score was produced.

## Executed result and decision

- CPU target-free forward stage completed on TRAIN35 and100. Same-backend
  null all16 served bytes and physical selector memberships exactly matched
  the unmodified frozen d50 BirthDirectionCard. All four fixed P perturbations
  were acquired on each net, without truths.
- MPS target-free forward stopped at TRAIN35. Null also matched its own frozen
  parent bytes; all five attempted CPU/MPS all16 maximum output differences
  were <=1.1920928955078125e-6 (limit1e-5).
- At fixed amplitude.025, the synthetic directional secant relative difference
  was0.004903593533971467 (0.4904%). At.05 it was0.13046692087695158 (13.0467%),
  exceeding the prospectively frozen5% gate. Original CPU/MPS .05 secants were
  -4.095735528453215e-5 and -3.56137752532959e-5.
- Therefore this fixed qualification FAILED. Rank100 MPS forward, both full-P
  gradient stages, final qualification judge and six-epoch fit DID NOT RUN.
  No TRAIN or VALID target arrays were opened. No gradient comparison or raw
  optimization result exists. No threshold, amplitude, seed or backend rescue.

This closes the fixed numerical attempt, not the mathematical possibility that
autonomous current P optimization could reduce raw error. Small absolute primal
agreement is not substituted for the failed prescribed directional criterion.

## Bounded saved-output diagnosis

The separate saved_scalar_audit_v1 read only already sealed tiny FP32 outputs
and the original cotangent. Zero card calls, gradients, weights or truth reads.
Using common math.fsum FP64 scalar evaluation for both backends leaves .05
secant discrepancy7.535934252587385%, still above5%. The signed original gap
5.34358003123625e-6 decomposes into saved-primal contribution3.0865193659246337e-6
plus original-scalar-evaluation contribution2.2570606653116165e-6.

The .05 full-vector secant relativeL2 is2.5695032518565402%, final-layer2.0932948741223905%.
At.025 the common scalar discrepancy is0.9431244177238573%, vector4.839780092359662%.
A cancellation-sensitive scalar projection can have larger relative error than
its vector; that different metric does not replace the frozen scalar gate.
Common math.fsum is high-accuracy evaluation, not a rigorous exact-real enclosure.

All47 selector memberships agree CPU/MPS on all five saved rank35 cases.
Several ordered index lists differ. Their causal numerical contribution is
unproven; these receipts exclude changed memberships, not ordering effects.

## Resources and preservation

CPU stage147.60s, supervisor148.83s; child peakRSS6.515933184GB.
MPS stage9.91s, supervisor10.64s; peakRSS1.135640576GB and maximum sampled
driver allocation6.478888960GB. Both stayed within the preset900s/12GB/16GB
limits. MPS supervisor passed=false because the child numerical gate failed,
not because a resource limit was exceeded. Neither supervisor terminated a job.
Swap stayed760.75MiB; no numerical processes remained at the final inventory.

All sources, failure artifacts and prior receipts are retained. No board
submission, promotion, commit, deletion or forum post occurred in this attempt.
train.py and build_terminal.py are reviewed but UNEXECUTED and NOT ADMITTED.
native40.py and NATIVE40_IMPLEMENTATION.md are preserved UNREVIEWED, UNEXECUTED
downstream drafts; judge_terminal.py was never created. Do not continue this
attempt from a draft or treat its CPU PASS as complete fit admission.

The next scientific decision needs a separately justified raw-accuracy route
or a genuinely different qualified numerical implementation, with its own
prospective complete-objective test. This failure is not permission to reopen
closed raw families, substitute a passing metric, or run an unqualified fit.
