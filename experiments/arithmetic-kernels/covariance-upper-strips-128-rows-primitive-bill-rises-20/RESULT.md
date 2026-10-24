# Smaller upper-triangle strips increase the native bill

The fixed128-row circuit fails its primitive cost gate. Computing fewer
output entries does not compensate for its additional input transformations.
No full estimator candidate, endpoint run, width selection or submission
follows.

| Warm second covariance product | Parent |128-row strips |
|---|---:|---:|
| Complete native FLOPs |828796736|999639808|
| Median CPU seconds |.067598|.025703|
| Native operation count |172|204|
| Relative error versus identical-input float64 product |2.93494e-5|2.52192e-5|

The bill rises20.6134%, despite CPU time falling61.9767%. The fixed gate
required at least20% bill reduction and no more than25% CPU increase.
Actual full-estimator and worker effects were not measured.

Small signed/asymmetric orientation and input-ownership checks pass.
All three warm repetitions have exact repeated output bytes and full bills.
Candidate-parent relative Frobenius difference2.79204e-5 is within the
declared1e-4 bound. This is not an exact symmetry or bit-parity claim.

The original seal inadvertently pinned its own then-empty file. The first
run stopped at the seal assertion before setup or arithmetic. Keep its
source, seal, log and failed guard. `bench_v2.py` changes only the names
of the repaired seal and output receipt; candidate and gates are unchanged.
The authoritative result is `result_v2.json`; see HARNESS_FIX.md.

Both guarded groups retired, one completed and one pre-arithmetic harness
failure. Peak sampled RSS1,048,903,680 bytes; zero added swap. Session57047
reaped with exit0. No network weights, targets, held-out arrays or fits.

The source review also recovered already-executed alternatives: E-43's
two-versus-three512-block covariance products; old minimum-cost depth-policy
savings under0.5% with runtime costs; and the S31 spherical tail diagonal
variant's0.02322% TRAIN2 adjusted gain. None was rerun as an untested lead.

S31 remains the official bank at2.1736465056908115e-9 adjusted in its saved
receipt. At unchanged raw1.6931451014556843e-8, even reaching the0.1 scoring
floor gives1.6931451014556843e-9. Thus reaching approximately1e-9 requires
a substantial raw-error improvement as well as, or instead of, lower cost.
No fresh official poll; submission authorization remains in force.
