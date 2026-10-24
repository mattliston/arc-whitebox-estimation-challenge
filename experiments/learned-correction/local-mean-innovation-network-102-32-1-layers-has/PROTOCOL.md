# Fixed supervised local-mean innovation observer

Phase2, 2026-09-20. Prospective before new captures, source labels or fitting.
Parent is accepted Lazy b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c,
official331599 adjusted3.8164492657795035e-9. Full~1e-9 goal stays active;
qualified submissions are preauthorized. No old failed experiment is reopened.

New information: checkpoint56's current-Lazy ledger finds a substantial
formal endpoint component from late mean innovations, with separate full
off-diagonal covariance components. It does NOT supply a causal predictor.
The old frozen-trajectory observer trained a recurrent final correction on
20 final-mean labels; the old matrix observer learned315 covariance-source
coefficients from final labels. Here learn a different target: local mean
innovations after subtracting measured incoming mean/covariance transport.
Existing hidden states/features and the old Gaussian-gain tangent are reused;
their names or availability are not claimed as new information.

TRAIN is exactly150,200,250,300,400,500. CHECK is35,100. These eight public
networks and the two CHECK references have been reused in past research;
this is network-disjoint fitting, not pristine generalization. No fresh6 or
confirmation targets are used. No new samples. Use existing coherent8M
references ONLY for TRAIN labels. CHECK8M labels need not be opened; CHECK
ordinary1B final means are accessed only after final forecasts are sealed.

Passively capture original parent35/100/35 first, then six independent cold
TRAIN calls. Add only Python reference bindings, including native features30,
outgoing H64, sigma, post7..14 internal means/maskedC, pre8..14 C and Gaussian
response data, final internal mean/response and served means. Require all16
prediction bytes and COMPLETE original bills exact, repeat captures exact,
cache retirement, and common fields byte equal to existing LCET/LFTO captures
on35/100. Expected TRAIN cold outputs/bills come from the accepted Lazy panel.
No altered native numerical expression, recast or estimator mutation.

For each TRAIN layer l8..14 define e_l=true post mean minus internal parent
mean, E_l=sym(true centered post covariance minus captured maskedC). Let
T_l be exactly lazy_covariance_error_transport_v1/common.py's Gaussian-gain
tangent. Labels are d_l=e_l-(T_l(e_(l-1),E_(l-1)))_mean. At15 use true
post mean minus saved served FP32 output, and subtract the tangent's final
mean times the captured internal-parent positive indicator. Thus final
labels also include the fixed served rounding/clipping defect. This is an
auxiliary signed-error model, not a coherent alternate probability law or
the full native Jacobian. Verify each TRAIN reference's post-to-pre affine
mean/covariance provenance<=1e-4 on all eight target layers.

Predict d_l/sigma_l from102 inputs: the30 current native features,64 outgoing
native H entries, and fixed8-way one-hot layer8..15. Normalize only the94
continuous inputs by global TRAIN coordinate mean and population standard
deviation, floor1e-6. Divide labels by one positive global TRAIN RMS; store
normalizers/scales as FP32 for the deployed arithmetic. These normalizers
do not use CHECK, true moment features, neuron IDs, rank IDs or network seeds.

One shared MLP102->32->1 with ReLU,3329 trainable scalars. Use CPU Torch
float32, seed2026092057, first weights Kaiming-uniform with a=sqrt(5), hidden
bias uniform[-1/sqrt(102),1/sqrt(102)], output weight/bias initially zero.
Exactly100 epochs; each epoch one fixed seeded randperm over49152 TRAIN rows,
minibatch2048 (24 updates/epoch,2400 total). AdamW, betas(.9,.999),epsilon1e-8,
weight_decay1e-4, foreach=False; epoch LR .0003+(.003-.0003)*.5*(1+cos(pi*(ep-1)/99)).
MSE of normalized local labels, global grad cap5 with nonfinite rejection.
No final-loss fine-tuning, checkpoint selection, extension, seed/LR/width/
feature/normalizer/source/window/rank sweep, blending or parameter rescue.

Before TRAIN labels or updates, qualify the exact MLP on deterministic small
fixtures: independent NumPy equations, CPU Torch outputs and parameter
finite differences, actual FNP outputs, zero head, shapes/dtypes and repeated
metering. Require output scaled error<=2e-5, gradient relative<=2e-4 on
smooth fixtures. Source seal the model, trainer, label preparation and all
hyperparameters before decoding any new label. No MPS is needed.

At the predetermined terminal epoch, retire optimizer and training leaves,
restore a nontrainable payload, and record TRAIN local-label MSE/zero-predictor
MSE. Require<=.50 to advance. Failure closes the study before CHECK scoring.
This is a source-fit screen, not an estimator score or information bound.

If that gate passes, use only causal captured fields to forecast on all8
networks. Auxiliary e7=E7=0. At8 add predicted d8 to zero e/E. At9..14
apply the fixed tangent to e/E then add predicted d_l to e only. At15 apply
the same positive-indicator final tangent, add predicted d15, and return
max(0,parent_served_final+e15). Parent physical state/H stay fixed throughout.
E contains covariance response generated by mean errors, with no learned
fresh covariance source or diagonal-only replacement. All predictions must
be sealed before opening CHECK1B means. Both CHECK raw ratios must be<=.85.
These reused development scores only license actual native compilation;
they are not adjusted scores or bank qualification.

A subsequent FNP integration, if licensed, must measure own complete cold
costs and native/client fidelity. Both own-cold adjusted ratios<=.90 and all
existing broader panel/confirmation/runtime/contract/package gates are then
required. New training on public rows must be disclosed in any panel report;
fresh6 and confirmation remain additional checks. No upload on this pilot.

Paid deployment hypothesis:8 small MLP evaluations and an auxiliary e/E
tangent alongside the unchanged host. Initial zero state avoids layer8
congruence; layers9..14 require12 full matrix products, final15 variance
requires one matrix product plus column reductions. That is13 square-product
equivalents (27.904 billion conventional FP32 operations at n1024), plus all pointwise,
copies, setup and source evaluation. This is circuit planning, not measured
native FLOPs or runtime. Possible FP32 fast products require their own
fidelity qualification. No cost is credited before a complete native bill.

One numerical group at a time, nice10, six BLAS variables1,8GB group RSS,
zero added swap. Existing256MiB emergency guard retained. Capture qualification
240s, each TRAIN cold90s, units60s each, label preparation120s, CPU fit300s,
prediction120s, scoring30s. At most2GB new artifacts (roughly1.2GB captures),
no weights/reference copies, no GPU, external write or old-file changes.
