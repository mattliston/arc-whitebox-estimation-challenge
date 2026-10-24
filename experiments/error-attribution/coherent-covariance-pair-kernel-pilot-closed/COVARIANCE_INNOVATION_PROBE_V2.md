# Coherent 8M state protocol, September 5, 2026

The v1 row generator completed ranks 35/100/150/200, then failed before
any fitting because 1e9 mean files do not exist for 250/300/400/500.
No model, verdict, or eight-row completion receipt was produced. Preserve
all v1 outputs, including incomplete directories.

For v2, use the existing **same 8M sample** post means and post covariances
from each mps_cov2 file for all eight teacher trajectories. This makes
the empirical teacher state internally coherent. It changes the estimand
from v1 and is not a 1e9-mean or noise-free test. Regenerate all eight into
a new directory; do not combine v1/v2 rows. All files were checked to
contain P=8,000,000 and the required 16-layer arrays before generation.

The fixed basis, six training nets, held-out 400/500, layers 8..14,
100-epoch objective, and >=25% per-held-out-net two-metric gate remain
exactly as predeclared in COVARIANCE_INNOVATION_PROBE_V1.md. No fit has
been inspected. The v2 trainer requires the completed eight-row receipt.

These are existing public research labels; no sealed dataset targets are
opened and no new samples are generated. Successful information gates
would still require independent-noise validation and deployable student
state tests. Failure closes this specific cheap response pair kernel.
