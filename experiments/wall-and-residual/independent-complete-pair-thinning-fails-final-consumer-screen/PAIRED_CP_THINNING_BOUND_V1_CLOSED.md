# Independent complete-pair thinning fails the final-consumer screen

September6,2026, finished10:34:57 PDT by the actual runtime clock.
No sampling draws, fitting, carrier changes, candidate or submission.

At half the expected signed-pair count, the mathematically optimal
independent Bernoulli probabilities yield linearized added final-MSE:

| Net | Original pairs | Expected retained | Optimal added MSE | Uniform added MSE |
|---|---:|---:|---:|---:|
|35|11936|5968|7.97996804392e-9|8.82063052238e-8|
|100|11872|5936|7.94569647108e-9|7.92275660941e-8|

Both miss the declared1e-9 allowance by about8x. These are expected
LOCAL perturbation variances at the frozen bank's actual final readout,
not raw errors against truth or adjusted scores. Earlier-layer error,
state feedback, probability acquisition and all runtime costs were free.
Close this independent-pair proposal: no budget/probability sweep, draws,
whole carrier, coadaptation, panel or upload. It is not a universal bound
on deterministic or correlated compression, or a coadapted estimator.

The numerical prerequisites pass. Full MPS baseline max-mean errors
1.39839787394e-6 /1.39716386938e-6 are below the unchanged2e-6 gate;
all30 active/birth memberships match. Complete-pair summation agrees with
the original signed cubes to5.61e-16 /5.25e-16 relative. The exact
diagonal-plus-rank-one final Jacobian agrees with independent reverse
derivatives to2.34e-8 /2.05e-8. Tiny exhaustive Bernoulli enumeration
matches the variance formula exactly at reported precision.

One nice10 worker,3 CPU threads, one MPS job;3.381seconds. Peak process
RSS2,062,761,984B; largest observed GPU driver allocation6,478,872,576B.
These are not independent additive peaks. No threadpoolctl or MPS fallback.
No target labels loaded.

Source SHA2cb6e5aa04aecec14c3fd9ffa27da55981e8649e28266125162ebd9f88da7b2d.
NPZ SHAb5497fccc0a1a79eab699b9912e1c4210fa5a91edf7221ae84f3fdd3f4974f81.
Evidence: paired_cp_thinning_bound_v1.py/.json/.npz and the preregistered
PAIRED_CP_THINNING_BOUND_DESIGN_V1.md. Preserve every artifact.

Additional archive check after this result: omitting CP output rows near
the ReLU linear tails is NOT novel. RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md
already records its exact-linear certificate,200-row matched fit and
failed3% raw-worsening gate on rank35. Do not rebrand that proposal as a
new saturation-aware read strategy or repeat its threshold experiment.
