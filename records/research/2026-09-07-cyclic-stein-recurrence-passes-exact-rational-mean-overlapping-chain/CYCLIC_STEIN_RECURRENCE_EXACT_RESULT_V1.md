# Cyclic correction: exact bias and variance certificate

September 7, 2026, 05:29:25 PDT execution; receipt timestamp 12:29:25 UTC.
Phase 2 only. This is a mathematical construction result, not a raw-error,
complete-compute or adjusted-score measurement. No network run is admitted.

## Result

The fixed rational, width2/depth3 affine recurrence passed every declared
gate in CYCLIC_STEIN_RECURRENCE_EXACT_PROTOCOL_V1.md. One short nice10,
standard-library process completed with exit0. No MPS, network weights,
truth arrays or fitting were used. The script and protocol were read back
before execution; the full result JSON was read back afterward.

- N=3,5,6: exhaustive enumeration of 8,32,64 independent-sign states
  respectively gives the analytic mean at EVERY layer of EVERY chain.
- N=5 and N=6: the full overlapping-chain covariance formula equals the
  enumerated variance exactly, as rational numbers.
- N=6: the disjoint lag-three covariance is exactly zero in both outputs.
- The same-row negative control is biased by 247/4050 and 7307/176400
  in the two toy coordinates (approximately .0609877 and .0414229).
- N=2, which reuses a row across three sites, is also biased.
- The mean transfer matrices do not all commute; the certificate is not
  limited to commuting scalar transfers.

Exact final target means are 36509/70875 and 1129/15750. At N=6 the
coordinate variances of the cyclic average are respectively
17314027579/482233500000 and 67492777307/2333772000000.
These fixture numbers say NOTHING about Phase 2 error magnitude.

Receipt naming caveat, preserved without rewriting output: the N=2
negative control's fields named variance/covariance are computed around
the analytic target, not its own biased expectation. They are therefore
target-centered second moments, not centered variances/covariances.
Only its failed mean equality is used. For the positive N=3/5/6 cases,
the exact mean gate proves target=expectation, so their variance checks
are correctly centered. Applicability flags must be respected: no doubled
covariance formula is claimed for N<2L-1.

## What was proved, and what independence is required

For L sites and independent COMPLETE rows Z_i, define

    D_l^i = S_l(Z_((i+l) mod N))
              + A_l(Z_((i+l) mod N)) D_(l-1)^i,  D_input=0.

When N>=L, the current row is independent of every preceding row in that
chain. Induction gives the desired expectation recurrence. All-site
statistics can be acquired once per row; separate full network draws for
every site of every chain are unnecessary.

Complete row randomness includes the actual input, Mehler draw and probes.
Within-row correlations across layers are allowed. A globally shared
random probe is NOT covered by the finite-range variance claim. Reference
means, slopes and controls must be deterministic given the weights, or
independently acquired with an appropriate conditional argument and bill.

For N>=2L-1, cyclic stationarity gives, coordinatewise,

    V_LR = Var(D^0) + 2 sum_(k=1)^(L-1) Cov(D^0,D^k),
    Var(average_i D^i) = V_LR/N.

The existing efficiency agent independently checked both the construction
and the N=2L-1 boundary. The certificate does not validate the scalar Stein
implementation, floating-point control means, covariance estimation,
finite-sample confidence bounds or any target-network result.

## Revised paid leading circuit

With n=1024,16 sites and r probes per independent row, two actual forwards,
two probe JVPs, one static-control JVP, and cyclic mean-error transport have
the classical FP32-convention leading subtotal

    c(r) = (94+92*r)*n^2,
    c(1) = 195,035,136 FLOPs per row.

The first preactivation probe product is shared by BOTH actual JVPs and
the control JVP. The zero-input recurrence product is skipped. All-site
deterministic linearized control norms can be supplied with a classical
60*n^3 = 64,424,509,440-FLOP setup subtotal. Relevant FP64 arithmetic would
be billed at its own higher rate. These are declared circuit counts, not
an optimality lower bound or complete native measurements.

Still charged and not yet fully specified: reference trajectory/heads,
control-slope acquisition, scalar gates, normal functions and stable
Stein tails, RNG and Mehler mixing, norm extraction, gathers/copies,
initialization, buffers, output reductions and resource qualification.
Storing all S/G rows costs O(N*L*n); streaming requires a real wrap-record
lifetime schedule. No current bank setup is free, and no cost is borrowed
from an incompatible candidate.

For the DECLARED one-probe circuit, with exact unbiased arithmetic and
V_LR averaged across output coordinates, the scoring multiplier implies
the optimistic necessary condition

    V_LR < 1e-9 * 2^41 / 195035136
         = 1.1275010752688173e-5

for adjusted <1e-9, even granting setup and every omitted operation free.
This is a coupled-estimator variance requirement, not the preceding
single-site Vp bound. No existing receipt measures this V_LR. Different
circuits, controls or sampling dependence require their own analysis.

## Decision

The prior same-path product-bias obstacle has a concrete unbiased solution
with shared row acquisition and a correct full-variance formula. The
remaining decisive obstacle is statistical efficiency at COMPLETE cost.
No favorable actual variance magnitude, complete bill or numerical/runtime
qualification follows from this exact certificate. Do not launch a full
network sampler, direction/probe sweep or fit solely because it passed.

Official bank remains #329926, adjusted4.982231460183304e-9. Retained
independent14 response component remains4.83424098592912e-9. No milestone
crossed, no submission and no reopening of the response exception.

## Immutable artifacts and resources

- Protocol SHA090f1b0754d4f227ef3b811c189efa7c41a1b58eff1277690e67d0338c81d933.
- Script check_cyclic_stein_recurrence_exact_v1.py,
  SHA365c18cf1d499566da17dfbe9bda8d8a56c73dfdc079281609a10f05eee4e638.
- Result cyclic_stein_recurrence_exact_v1.json,
  SHA07e810718fd39bf6b64b4f43c02e079bdcc34d8c9ac80060f3432dccb5ae5938.

Pre-execution resource check: no Python/Whest workers; swap784.75MiB of
2048MiB. Initial sandbox sysctl was denied; the authorized read-only
escalated check succeeded. The sole tiny process completed, no live handle.
No candidate/frozen-file/root-estimator edit, upload, post, commit,
promotion or deletion. All existing work remains preserved.
