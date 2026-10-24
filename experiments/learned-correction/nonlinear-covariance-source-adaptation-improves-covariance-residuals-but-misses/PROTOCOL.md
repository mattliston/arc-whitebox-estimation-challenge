# Current-state nonlinear covariance-source adaptation

Prospective Phase 2 study after checkpoint80. Official331684 remains
adjusted3.5844144212297072e-9; 4e-9 is banked, 3/2/~1e-9 unmet. The previous
turn made progress by completing the positive-leaf catalog search. The full
goal and authorization for qualified uploads remain active.

The accepted edge source was trained on older, oracle-input states. The
current-native innovation study reused its nonlinear encoder and changed
only sixteen output weights. It failed its covariance/next-response gate.
That result does not measure whether adapting the nonlinear features can
transfer on the current autonomous states. This new study explicitly admits
training those features; the older source, fit, failed gates and stopping
decision remain preserved. This is a distinct bounded capacity test, not
a claim that the old fit succeeded. The exact bank's use of the older edge
model already has a native benefit; no further benefit is assumed.

Reuse all eight authenticated current-native captures and coherent8M
mean/covariance references from full_lower_covariance_innovation_supervision_v1.
TRAIN=35,100,150,200,250,300; VALID=400,500. They are reused research networks,
not pristine holdouts. No new estimator call or dataset is needed for this
information test. Keep the original normalized FP32 inputs F(14), R, S,
sigma/pdf/gate envelope, sixteen tanh channels and cov_factor. No new input,
layer indicator, rank, layer policy or normalization is introduced.

Use the same Gaussian-gain local covariance innovation U as the prior study:
actual covariance error minus its declared transport of previous mean and
covariance error, at layers8..14. This is an approximate local source label,
not an exact derivative of the full cumulant program. Reproduce its FP64
energy and next-Gaussian-mean response energy against the old saved statistics
to relative2e-10. Source affine consistency retains the old1e-4 gate.
Round U, next weights and response toFP32 for training; retain diagnostics
for resulting label/response roundoff, each relative<=3e-6. No1B mean labels.

Define delta(theta)=edge(theta)-edge(bank) on the frozen current inputs.
Both arms start at the exact original bank parameters, hence zero delta:
(1) output-only: train16 output parameters; (2) nonlinear: train all528
parameters (left/right14x16, edge3x16, bias16, output16). This is the same
model size at deployment, not an additional residual network. Original
input normalizations and cov_factor are fixed in both arms.

Fit each arm for exactly40 epochs of42 cells, in the same deterministic
network/layer permutations from seed810926+epoch. One update per full1024
cell. Objective is mean(delta-U)^2 / TRAIN covariance energy plus mean
next-response(delta-U)^2 / TRAIN response energy, and .001 times the mean
squared parameter change over all528 original coordinates. Norms average
uniformly across six networks and seven layers, from TRAIN only. Fixed Adam
beta(.9,.999), epsilon1e-8, float64 optimizer accumulators, FP32 parameters
after each update, gradient norm cap5, cosine learning rate5e-4 to5e-5 over
1680 updates. Only final parameters are evaluated. No best-epoch selection,
learning-rate/penalty/layer/label/roster search or fit extension.

Use an explicit NumPy forward/reverse implementation, CPU and one thread.
The reverse is the real-arithmetic derivative of this frozen proxy, not a
qualified gradient of the rounded native full estimator. Before arrays from
the real captures, certify forward/symmetry/permutation/null identities,
all528 derivatives against an independent scalar complex-step evaluation
on three fixed small fixtures, plus FP32/FP64 agreement and a directional
descent check. FP64 forward/derivative relative<=2e-10; FP32<=3e-5.
Check a full-width starting delta is byte-zero and source reconstruction
against the recorded native bank at relative<=3e-6 before each target read.
Track finite losses/gradients/parameters and all1680 completed updates.

Seal both final models before opening VALID features for prediction. Produce
and seal both arms' full covariance deltas on VALID before scoring against
their targets. Independently recompute final predictions with an explicit
channel-loop FP64 reference, tolerance3e-5 relative to the model's total
source and1e-4 relative to its delta. All reported losses use FP64 reductions;
cross-check direct residual energy with expanded quadratic identities.

Advance only if EACH VALID network's nonlinear arm has covariance residual
<=.98 of zero-delta baseline AND next-response residual<=.95, AND its
next-response loss<=.98 of the matched output-only arm. Report both arms and
each layer. A failure stops this fixed adaptation, preserving every receipt.
A pass licenses a separately frozen finite native test of the one final
payload with full costs, repeat/null/domain/runtime checks and ordinary1B
development scoring; it is not a candidate qualification or an upload.

One serial nice10 group, six BLAS variables1,8GB RSS ceiling,256MiB emergency
swap cutoff and ZERO added swap required. Units120s; each data-prep120s;
each arm1800s; prediction180s; score180s. No GPU, sampling, downloads or
1B target access. New retained artifacts<=2GB. Preserve all old files and
failures. Never restart a controller solely because observation times out.
