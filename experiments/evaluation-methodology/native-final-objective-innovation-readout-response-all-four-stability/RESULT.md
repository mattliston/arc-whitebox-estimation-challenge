# Native final-objective readout response: qualification failed

All four sampled direction/network cells fail both response-stability and
curvature requirements. The finite responses are resolvable, but halving the
step changes the secant by80.35–95.93%, versus a5% requirement. Curvature
ratios are0.8347–1.0334, versus a0.10 requirement. No final reference mean was
read, no probe accuracy was scored, and no final-objective coefficients were
fit. There is no new candidate or board submission.

| Retained direction | MINI | Relative secant gap | RMS(h*r_half) | Curvature |
|---:|---:|---:|---:|---:|
| 0 | 57 | 0.829581 | 4.58234684e-06 | 0.834714 |
| 0 | 68 | 0.884665 | 4.32802613e-06 | 0.922003 |
| 7 | 57 | 0.803451 | 4.33726376e-06 | 0.897991 |
| 7 | 68 | 0.959258 | 4.55528216e-06 | 1.033376 |

This experiment keeps the checkpoint47 hidden covariance encoder and its
normalizers, but discards its output weights and output bias. The32-channel
covariance basis is projected and conditioned on42 saved TRAIN cells. Eight
directions retain93.9481771% of its covariance Gram energy; each is scaled to
the discarded predictor's mean squared relative-covariance output. The selected
directions0/7 and h=1/8, h/2 were fixed before native probes. Bias stays zero.
All non-output model parameters remain exact. Earlier16-output native
calibration/SPSA failures and the checkpoint47 candidate failure are unchanged.

The zero readout matches all16 parent output bytes on both reused TRAIN
MINI57/68; it still incurs the helper's added bill. All sixteen nonzero network
calls preserve first9 means, finite/nonnegative outputs and seven injections.
The actual FP32 readouts equal the sealed directions times the declared step.
Independent Gram/eigenvector/normalization identities, model replay, source
pins, control checks and scalar response arithmetic pass. This is a limited
two-direction/two-network qualification; it establishes neither a global
derivative nor a general impossibility of final-objective optimization.

The preceding component diagnostic found frozen learned-correction raw ratio
1.0319499852 and live ratio1.0345932982 even on training networks. The privileged
remainder gives0.5730885661, versus the full innovation oracle0.5881857638.
The learned component changes the final mean without enough alignment with
its residual error. Its own feature feedback accounts for little of that
regression here. These observations motivated this different objective; they
do not make the failed readout calibration usable.

All12 response-study guards complete and retire. Combined with the five
component-diagnostic guards, all17 complete, peak5.539GB,zero added swap.
No radius, direction, basis-rank, coefficient or probe-sign rescue follows.
The next research question is what causes the native response instability at
these scales; selection changes and numerical sensitivity are hypotheses,
not established causes. No next numerical experiment is selected here.

S31#332517 remains the saved official bank at2.1736465056908115e-9 adjusted.
Its source/archive hashes are unchanged. No fresh official poll, approval,
worker or grading is pending. The2e-9/~1e-9 goal remains active and unmet;
qualifying board submissions remain authorized.
