# Phase 2 width-equivariant residual observer

Status: **TARGET-FREE DESIGN ONLY — NO FIT, EXTRACTION, OR PUBLIC TARGET OPEN**

This is the smallest learned successor I would launch around the current
cap-1792 CP3 / Hermite-10 host. It is a sequence model, but it is deliberately
not a learned replacement for the moment recurrence. The analytic host rolls
forward exactly as it does now. A fresh, shared graph-RNN watches that
trajectory and adds one bounded mean residual to each reported neuron.

No Phase 1 checkpoint, feature normalization, residual scale, fitted
coefficient, layer schedule, or optimizer state is reused. The only Phase 1
inheritance is qualitative: train on the model's complete recursive rollout,
and do not confuse teacher-state imitation with autonomous accuracy.

## Decision in one paragraph

Keep CP1792 because its measured total leaves enough room below the `0.1`
score floor for a small observer, and because it is the current Phase 2 raw-MSE
leader. Add a 32-channel, width-equivariant recurrent observer whose messages
move through the actual weight matrix using four normalized elementwise weight
powers. Train the complete 16-layer observer end to end against the public
post-ReLU mean residual. The observer may use teacher moments as low-weight
auxiliary labels, but teacher states are never inputs and never define the
recurrent trajectory. In version 1, the learned correction is readout-only:
it does not alter the host mean, covariance, CP state, K21, gates, or later
features.

## Why this is the safest next learned move

The evidence points away from immediate learned feedback.

- CP1792 finishes the already-opened mini-100 comparison at final raw MSE
  `3.722402415224689e-6`.
- The fully recursive cap-512 K3/K21/covariance candidate is initially useful,
  then diverges: its final raw MSE is `2.0043852911640636e-4`, versus
  `3.722402415224689e-6` for CP1792, and it loses on all 100 networks. That is
  a direct warning that a plausible moment correction can push the whole
  closure onto a much worse state distribution.
- In Phase 1, A164 passed a four-coordinate fixed-state imitation gate with
  normalized held loss `0.5653236097953831`, yet its autonomous L32 raw MSE
  was `9.8091435290705e-6`, worse than the `4.205712685371052e-6` parent.
- The subsequent A166 on-policy action-head run improved its selection score
  from `9.055830797763e-6` to `5.935467254481e-6`, but remained far from
  `1e-7`. Freezing the inherited trunk and Phase 1 scales left too little
  freedom to relearn the new operating point.

The Phase 2 observer therefore starts with a fresh trunk and fresh Phase 2
normalization, receives the actual host trajectory, and is trained through
its own recurrent hidden trajectory. Its output cannot destabilize the host.
Feedback remains a later, separately metered experiment only if attribution
proves that readout-only information is insufficient.

## Is it an RNN?

Yes. More precisely, it is a graph-RNN over the 16 MLP layers.

At every depth it has one 32-channel hidden vector for every one of the 1024
neurons. The same small cell is reused at every depth and every neuron. Between
layers, the actual MLP weights move four groups of eight hidden channels into
the next neuron basis. The shared cell then combines those transported
messages with local analytic/CP diagnostics and emits:

1. the next 32-channel hidden state; and
2. one scalar correction to the host's reported post-ReLU mean.

There are no per-neuron learned embeddings. Relabeling the neurons at any
layer simply relabels the features, messages, hidden state, and output.

## Frozen deployment topology

```text
autonomous CP1792/Hermite host trajectory
          |
          +--> 20 target-blind local features per neuron
          |
previous 32-channel hidden state
          |
          +--> four normalized W^[1..4] message transports
          |
          v
shared 56 -> 64 -> 64 gated recurrent cell
          |
          +--> next bounded 32-channel hidden state
          |
          +--> bounded direct post-mean residual
                       |
host reported mean ----+----> final reported mean
```

Here `W^[p]` means the elementwise power `W ** p`, not a matrix power.

The host runs once. Feature extraction must expose intermediates from that
same pass; it may not rerun Hermite or CP propagation.

## Width-equivariant local features

The exact version-1 local vector has 20 coordinates. Every coordinate is
either per-neuron, a broadcast permutation-invariant network summary, or a
depth coordinate. There are no fixed coordinate probes.

| # | Feature | Source |
|---:|---|---|
| 1 | normalized depth in `[0,1]` | geometry |
| 2 | CP occupancy divided by 1792 | CP state |
| 3 | retained CP energy fraction at the latest cap event | CP state |
| 4 | standardized preactivation mean `alpha` | Hermite host |
| 5 | log preactivation sigma minus its width mean | Hermite host |
| 6 | Gaussian ReLU gate | Hermite host |
| 7 | Gaussian density at `alpha` | Hermite host |
| 8 | Gaussian post-mean divided by pre-sigma | Hermite host |
| 9 | CP-corrected reported mean divided by pre-sigma | CP host |
| 10 | CP mean correction divided by pre-sigma | CP host |
| 11 | CP K3 divided by pre-sigma cubed | CP host |
| 12 | exact Gaussian post-variance divided by pre-variance | Hermite host |
| 13 | Hermite orders 5:10 marginal coefficient energy / total energy | Hermite host |
| 14 | row mean of the current correlation matrix | covariance host |
| 15 | row RMS of the current correlation matrix | covariance host |
| 16 | row absolute mean of the current correlation matrix | covariance host |
| 17 | normalized current-weight column first moment | current `W` |
| 18 | normalized centered column second moment | current `W` |
| 19 | normalized column third moment | current `W` |
| 20 | normalized centered column fourth moment | current `W` |

The centers and scales for these 20 features are freshly computed from Phase 2
fit networks only, separately by depth and feature. They are scalar across the
width dimension. Standardized values are clipped to `[-8, 8]`. A zero or
nonfinite scale fails closed; it is not replaced with a Phase 1 value.

The four pre-normalization message RMS values are log-transformed,
fit-standardized, and broadcast as four additional invariant features. Thus
the recurrent cell sees 24 local/global features plus 32 transported message
channels, for 56 inputs total.

## Stable message transport

The weights satisfy the pinned Phase 2 law `W_ij ~ N(0, 2/n)`, with
`n = 1024`. Split the previous hidden state into four groups
`h^(1),...,h^(4)`, each with eight channels. Define target-blind, analytically
normalized edge matrices

```text
A1 = W / sqrt(2)
A2 = sqrt(n/8)       * (W**2 - 2/n)
A3 = n/sqrt(120)     *  W**3
A4 = n**(3/2)/sqrt(1536) * (W**4 - 12/n**2)
```

and raw messages

```text
u^(p) = transpose(Ap) @ h^(p).
```

The constants are the exact Gaussian second-moment normalizers for the four
elementwise powers. Centering the even powers prevents a width-growing DC
message. For each power group, divide `u^(p)` by its RMS over all destination
neurons and eight channels, using a fixed epsilon, and clip to `[-5,5]`.
Retain the pre-normalization log-RMS as one of the four broadcast features.

This gives three layers of protection against recursive drift:

- analytic width normalization before transport;
- observed group-RMS normalization after transport; and
- a bounded recurrent state after the cell.

All reductions are symmetric across neurons, so this remains equivariant to
independent neuron permutations at every layer.

## Shared recurrent cell and output coordinate

Let `x_lj` be the 24 local/global features and `m_lj` the concatenated 32
messages for neuron `j`. One shared cell computes

```text
q1 = SiLU(Linear56to64([x, m]))
q2 = SiLU(Linear64to64(q1))
c  = tanh(Linear64to32_candidate(q2))
g  = sigmoid(Linear64to32_gate(q2))
h  = tanh(m + g * (c - m))
e  = tanh(Linear64to1_mean(q2))
```

The only deployed learned output is a direct physical mean residual:

```text
delta_mean = indicator(layer > 1) * pre_sigma / sqrt(1024) * e
prediction = max(host_reported_mean + delta_mean, 0)
```

The `pre_sigma/sqrt(width)` scale is geometry-derived and available online.
It is not a Phase 1 residual scale and not selected from a score. The tanh
keeps a single bad feature from producing an unbounded correction. Layer 1 is
hard-gated to the exact Gaussian answer and anchors the recurrence.

The mean head is initialized to exactly zero, so the untrained model exactly
replays CP1792 predictions. All other weights use one fresh Phase 2 seed
derived from the literal string `p2-w1024-d16-residual-observer-v1`; no Phase 1
tensor or random state is loaded.

## Source gates and fail-closed behavior

Four optional input blocks have shared scalar sigmoid gates: Hermite-tail,
CP, covariance summaries, and weight diagnostics. Each of the four message
groups also has one scalar gate. All eight gate logits are fresh parameters;
they are not layerwise tables. Trusted marginal Hermite coordinates and depth
remain always on.

During training, one optional source block is masked with fixed probability
`0.05`, independently by network, to prevent a single fragile diagnostic from
becoming the only route to the answer. This is training regularization, not a
deployment ensemble or variant.

At deployment, any nonfinite host state, feature, message normalizer, hidden
state, or correction makes that layer's learned correction exactly zero and
records a failed observer diagnostic. The host prediction remains available.
No learned value is allowed to enter mean, covariance, CP, K21, Hermite
coefficients, or future host features in version 1.

## End-to-end, on-policy training law

The main examples are Phase 2 networks, not isolated neurons and not teacher
states. Use the already-designed network-level hashed `full` split:

- 700 networks: fit;
- 100: checkpoint selection;
- 100: one-use audit;
- 100: final public lockbox.

For every batch, run the observer through all 16 layers with its current
parameters and its own hidden messages. Backpropagate through the complete
16-step recurrent graph. Do not shuffle layers, cache hidden states, teacher
force messages, or fit a per-layer tape independently. The host feature tape
may be cached because version 1 never changes the host state; the observer's
hidden tape may not be cached across optimizer updates.

The primary label is

```text
public all_layer_mean - frozen CP1792 host reported_mean.
```

The public mean is the scoring target and remains the primary supervision.
Compute fresh per-layer residual RMS scales from the 700 fit networks only.
The fixed training objective is

```text
L = normalized final-layer mean MSE
  + 0.25 * normalized all-layer mean MSE
  + 0.02 * masked auxiliary-coordinate MSE
  + 1e-4 * normalized correction-energy penalty
  + 1e-4 * hidden-state energy penalty.
```

Use one deterministic AdamW run, learning rate `1e-3`, weight decay `1e-5`,
network batch size 2, gradient-norm clip 1, maximum 160 epochs, selection every
four epochs, and patience 24 selection checks. The selected checkpoint is the
lowest network-balanced selection final-layer raw MSE; all-layer MSE is a
reported stability diagnostic, not a second selection knob.

The full trunk, recurrent heads, source gates, and mean head train together
from scratch. This is the crucial difference from A166's frozen Phase 1 trunk.

## Teacher labels are auxiliary attribution, not the policy

The design can consume the separate Phase 2 teacher-attribution program
without becoming dependent on teacher states.

Training-only auxiliary heads may predict residuals in causal pre-mean,
log-variance, standardized K3, and diagonal standardized K4 from `q2`. They
have 260 parameters total and are removed from the deployed checkpoint. The
teacher values:

- are labels only;
- never enter the feature vector, message transport, hidden state, or host;
- are masked on networks without a certified teacher tape;
- receive total loss weight `0.02`, so public post-mean residual remains the
  optimization target; and
- cannot be used to select a hidden state or a feedback coefficient.

The teacher action tests may decide, under their already frozen rules, which
of these auxiliary labels is trustworthy. If that decision is unavailable
when training authority is granted, all auxiliary masks are frozen to zero
for version 1. Adding them later requires a distinct preregistered fit; it is
not a retry of the same run.

## Stability tests before any fit

The implementation must pass all of the following on synthetic and then
target-blind public-weight networks:

1. zero mean-head initialization reproduces CP1792 prediction bytes;
2. independent permutations at every layer commute with the full observer to
   numerical tolerance;
3. duplicating width with analytically rescaled Gaussian weights does not
   change feature/message RMS by more than the declared synthetic tolerance;
4. a 64-step synthetic extension remains finite with hidden absolute value at
   most one and bounded corrections;
5. setting any optional source gate to zero remains finite and preserves
   output shape;
6. no observer output changes a host state byte;
7. CPU and MPS reference rollouts pass the device parity gate; and
8. the converted flopscope implementation replays the frozen Torch CPU
   reference before target-bearing evaluation.

## Static deployment meter

`STATIC_METER_ESTIMATE.json` gives a conservative source-structure estimate.
It starts from the measured CP1792 host total `214,122,341,378` and allocates
`2,047,230,072` operations to the observer, for estimated total
`216,169,571,450`.

That is `9.8302539959%` of the `2^41` budget and leaves
`3,732,754,105` integer operations below the largest total that still receives
the `0.1` score multiplier. The observer itself is only about 0.0931% of the
full contest budget. The deployment model has 12,041 trainable parameters;
the optional four auxiliary heads raise the offline training count to 12,301
but are pruned for submission.

This is a design estimate, not an authoritative flopscope receipt. The
activation row deliberately reserves 64 ordinary-operation equivalents per
nonlinear scalar because the exact v0.12 special-function charge must be
measured in the real source. Promotion is blocked until the final estimator
passes the common subprocess meter with `10 * FLOPs <= 2^41`, residual Python
time below 0.4 seconds, total time below 120 seconds, and peak RSS below 8 GiB.

## MPS and CPU execution plan

MPS is the offline accelerator, not the deployment backend or numerical
authority.

- Generate and hash the frozen CP1792/Hermite feature tapes on CPU from the
  authoritative estimator logic. This is target-blind and done once.
- Stream at most two 64 MiB weight tensors per batch to MPS. Construct the
  four normalized weight powers on device, run the complete 16-step observer,
  and backpropagate in float32. Use layer checkpointing and monitor actual
  unified-memory pressure before every batch.
- Accumulate loss summaries and checkpoint-selection metrics in CPU float64.
- Before training, compare CPU and MPS on two synthetic and two target-blind
  public-weight networks. Require prediction disagreement MSE at most `1e-10`,
  identical finite/output shapes, and two deterministic MPS reruns. If the
  parity or measured `1.5x` end-to-end speedup gate fails, train on CPU.
- Convert only the selected fresh Phase 2 checkpoint to flopscope arrays.
  Submission inference uses the flopscope client and contains no Torch or MPS
  call.

## Promotion and launch blockers

No fit is authorized by this design. The blockers are:

1. The organizers must formally open Phase 2 or publish rules that do not
   invalidate the pinned v0.16/v0.12 contract.
2. Freeze the exact CP1792 host and one-pass feature-extraction source hashes.
3. Materialize the 1000-network hashed split manifest and pass the physical
   target-blind `full` Parquet projection/deny-list audit.
4. Freeze whether the four auxiliary masks are all zero or are licensed by
   the pre-existing teacher-attribution decision before observer target access.
5. Implement the observer once, with the exact 20-feature schema, four message
   normalizers, cell, fresh seed, optimizer, and stopping rule above.
6. Pass zero-head CP1792 parity, permutation equivariance, long-rollout
   stability, source-ablation, and host-noninterference tests.
7. Pass CPU/MPS parity, deterministic replay, measured speedup, and actual
   memory-pressure gates.
8. Obtain an authoritative common-driver flopscope receipt below the `0.1`
   floor with at least 10% of the current static headroom retained as meter
   uncertainty margin.
9. Obtain a separate root release for only the 700 fit and 100 selection
   targets; audit and lockbox targets stay sealed.
10. Seal the selected checkpoint and all audit predictions before the one-use
    audit target opening. Seal lockbox predictions before its one-use opening.
11. Promote only if the audit final raw MSE beats frozen CP1792 with positive
    network-paired direction, the last four layers show no monotone blow-up,
    and every resource/output gate passes. The scientific target remains
    final raw MSE below `1e-7`.

## Explicit exclusions

- no Phase 1 checkpoint, trunk, head, feature center, feature scale, delta
  scale, rank, or feedback coefficient;
- no teacher hidden state, teacher-forced recurrent state, or per-layer
  imitation tape as the primary task;
- no learned mean/variance/K3/K21/covariance feedback in version 1;
- no fixed coordinate probes or per-neuron parameters;
- no architecture, seed, width, channel, output-cap, or optimizer sweep after
  seeing fit/selection performance;
- no mini-100 retuning and no official hidden example access; and
- no graph or HTML mutation from this design package.

## Bound evidence read by this design

This pass read only existing source and receipts; it decoded no new target.

- CP1792 estimator SHA256:
  `035e3bd2fd08d3358f5424fc67f10281e5559e65fa9c6d6dce97791108b176dc`
- CP1792 meter-pointer SHA256:
  `6e30cc3d9849311004a54caae680be3d00c3e629ddc1f4c75c6a3ae3e81bb93b`
- existing joint mini-100 score receipt SHA256:
  `992f8e27d8b44c4ce236051022b4baf43e59b6f610ca57aedb6eb4315b73045c`
- existing recursive-K21 score receipt SHA256:
  `7cc72065af485d96ddcfc9f62eef47f8858e9c6f45599cf52ec3512f77cb681b`
- Phase 2 teacher-attribution design SHA256:
  `bb8ca6d673fef1c660c695e8baad14d9739a53efe9f4e22c7c52a15d4df25dd0`
- Phase 1 A164 imitation gate SHA256:
  `fbc377891d48f945d417b430488485342d14c50b585df5bc6f20162d1c866b91`
- Phase 1 A164 autonomous result SHA256:
  `3da42874f9d3f7419f5e290af89d444785faeeb89405b5b391ec630f6715bc03`
- Phase 1 A166 pause receipt SHA256:
  `cbd97bcf4d133351a23d9fd3e82fb074266cdb9e43b459ecd34c508f38ab8cc3`

