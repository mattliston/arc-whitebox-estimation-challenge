# Whole-bank gradient certificate after the explicit-bound repair

September6,2026. New MPSv3 full forward passes both35/100 at max1.399e-6 /
1.398e-6, exact active/birth memberships, unchanged2e-6 gate. Native fixed
CDF and explicit tensor-bound min operations passed small value/gradient
certificates. This is no training or score improvement.

Before any fit, run one fixed target-free reverse-mode objective per net:
random final-layer linear projection, PCG64DXSM930096+rank, unit Euclidean
norm. No truth vector, output-error residual or chosen favorable neuron.
Compare ALL16x1024 mean-control gradients of the original CPUf64 BankFullCard
against the repaired MPSf32 card. Both use checkpointed full dynamic
forward, identical frozen P0/Q. Reference CPUf64 local full-carrier
derivatives already have independent NumPy finite-difference certificates.

Require finite gradients, both forward maximum errors<=2e-6, every30
active/birth membership matches the previously frozen reference for MPS,
and aggregate gradient relative error<=1%. Report each layer separately.
The projection is on unclipped means; tensor-bound clipping's finite-value
and equality derivatives have a separate exact certificate. This checks
a local Jacobian, not global smoothness across discrete membership changes.

As an independent numerical check, perturb mean control at layer13 in the
fixed archived direction_13, amplitude2e-4 and1e-4, plus/minus, using the
repaired MPS forward. Require all selection MEMBERSHIPS unchanged from
baseline; compare the two scalar central differences and their reverse
derivative. Each relative discrepancy<=5%, normalized by the reverse
derivative magnitude (floor1e-12). If a membership changes or the signal
is lost to roundoff, record failure; do not tune amplitude/direction.

Single differentiating worker, three CPU threads and one MPS job, no
threadpoolctl. Record process RSS separately from GPU driver allocation,
and check swap before/after. No parameter update or new target access.
