# Resummed pair-correlation consumer: one fixed candidate

Phase2, 2026-09-20. Prospective before numerical work. Parent is exact Lazy
b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c;
official #331599 remains the bank. Future qualified uploads are preauthorized.

The current pair map uses Gaussian correlation order6 and first-order
third/fourth-cumulant correlation order2. The old radial_mehler screen tested
Gaussian6 and Edgeworth2 on an older one-star host; it did not evaluate this
resummation. A bounded source search found no higher-correlation-order or
exact-correlation pair consumer in the inspected Astra files. This is not a
claim that all historical work has been exhausted. Higher *cumulant* orders,
radial mixtures, L3-squared and missing all-distinct sources are separate
closed experiments, not what is changed here.

Retain exactly the present marginal means/variances/k3/k4, cumulant supplies,
CP/star/trace births, masks, rank384, age4, Lazy rotations, learned parameters,
final scalar head, and matrix kernels. Replace the interior pair consumer
and first Gaussian pair consumer by the same first-order Edgeworth law with
its Gaussian correlation dependence evaluated through conditional Gaussian
boundary moments. No new cumulants, parameters, training, source strengths,
projection, PSD repair or correlation clipping.

For standardized X,Y of means a,b and correlation rho, compute P(X>0,Y>0)
from Phi(a)Phi(b)+integral_0^rho phi2(a,b;t)dt, with exactly32 Gauss-Legendre
nodes, streamed in fixed chunks8, FP64. Obtain raw truncated moments M_ab
(a,b in0..2) by Gaussian integration-by-parts recurrences. Derivatives of
ReLU powers are ordinary powers/indicators or delta/delta-prime/delta-double-
prime boundary moments. Conditional univariate moments give those derivatives
analytically. All normalized mixed cumulants and factorial coefficients agree
with the parent's formal L3+L4 map. Pure-local contributions subtract their
Gaussian marginal-product derivative; mixed cumulants do not. Then use the
parent's exact existing post-centering, diagonal restoration, radial reset
and aux construction. This remains a formal first-order cumulant closure,
not a positive probability law or complete higher-moment propagation.

The correlation integral is numerically approximated, not symbolically exact.
Require finite positive prevariances and absolute off-diagonal rho<=.85 in
the new consumer, with no fallback/repair. Kernel has an explicit disabled
path returning the original helper, preserving the full parent source prefix.

Qualification first, no challenge targets:

- Deterministic small signed-correlation fixtures, means spanning[-8,8],
  including rho0 and +/-.85. Compare32-node probability to128-node reference
  with maxabs<=2e-11. Compare raw M11/M21/M22 to independent200-node direct
  conditional integration, scaled error<=2e-10.
- For moderate means[-2,2], |rho|<=.4, independently sum80-term Hermite
  derivatives. Every required (3,0),(0,3),(2,1),(1,2),(4,0),(0,4),(3,1),
  (1,3),(2,2) derivative for AA/BA/BB must agree within2e-10 scaled. Compare
  the assembled L3+L4 pair covariances against that independent series too.
  Include a nonsymmetric rho array and retain its literal entry in both
  conditional boundary orientations; no symmetry repair of native C.
- Actual FNP helper replay must match the FP64 research implementation within
  scaled2e-10; domain rejection and zero-cumulant symmetry also checked.
  Freeze GL literals only from this fixed node construction, before native
  candidate/accuracy work. No quadrature-order/tolerance/precision rescue.

Then build exactly one enabled candidate by appending helpers and wrappers to
the unchanged parent. Preserve a disabled/null source. Verify the source prefix
and copied marginal/post-centering segments. Disabled cold35/100/35 must
reproduce all16 parent bytes, complete original bills, repeat and cache
retirement. Enabled cold35/100/35 plus separate cold100 must be finite,
nonnegative, correctly shaped, exact repeated/warm-cold outputs, unchanged
first returned row, and every new-domain check must pass. All operations,
casts, quadrature and setup are billed. Save predictions/bills before opening
ordinary1B means. These reused development networks are not fresh validation.

Advance only if enabled own-cold adjusted/parent<=.90 on BOTH35/100. This
fixed gate incorporates any extra compute. A failure closes this consumer:
no order, window, coefficient, blending, refitting, precision or kernel rescue.
A pass would require the existing NC14/confirmation/native-client/runtime/
contract/package gates before a milestone submission. No upload on dev alone.

One owned numerical group at a time, nice10, six BLAS controls1,8GB group RSS,
zero added swap required. Existing supervisor emergency stop256MiB swap.
Unit90s, each native sequence240s, cold100100s. <=30MB new artifacts besides
the two ~0.8MB source copies; no new weights, dense tapes or physical samples.
The full approximately1e-9 goal remains active regardless of this test.
