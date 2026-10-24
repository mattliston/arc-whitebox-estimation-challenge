# Source-only proposal: messages as covariance-edge inputs

Status: UNTESTED FUNCTION DRAFT. No numerical qualification, fitted model,
native participant, accuracy result, complete bill, release or upload exists.
The closed528-weight objective trial and every failed gate remain unchanged.
Start any separate study from official331760, not that trial's failed payload.

The current covariance edge uses14 per-node physical features and three
pair inputs (correlation and two directed third-moment slices). Its physical
inputs already depend indirectly on earlier GNN corrections and births.
The proposed addition is DIRECT access to the existing learned messages;
it is not a claim that the current edge ignores all network history.

For layers8 through14, use the previous layer's state and the frozen current
message projection. Compute the same three16-channel transports used by
gnn_step: W.T@src1, (W**2).T@src2, (W**3).T@src3. Concatenate and clip to
[-8,8]. Add messages@U_left and messages@U_right to the corresponding
node-side inputs of the existing16-unit pair tanh. The existing pair inputs,
output weights, symmetrization, pdf/gate/sigma factors and source scale stay
unchanged. U_left and U_right each have shape48x16:1536 newFP32 values total,
shared across these seven layers, initialized to positive zero. All P45,200,
A, cstar, existing528 edge values and normalizers remain frozen.

This changes the consumer and parameterization from the earlier signed-message
branch into the mean encoder. It also differs from old P+Q covariance work:
that trained P and a14x65 congruence head after mean recentering on an older
host. The correlated terminal-energy study is a different output feature.
Those existing experiments retain their own actual outcomes. This bounded
source review does not establish that no related experiment exists elsewhere.

Native integration is NOT implemented by prototype.py. The previous state
lives inside _predict_learned_compact's per-call closure. Add an optional
covariance callback to ONLY the final active _predict_binary definition;
there are five definitions in the bank. Invoke that callback at the existing
pre-mean edge point, with current W[l]. A closure in _predict_learned_compact
can read its current state and P before gnn_step updates state. Preserve the
old edge behavior for callers without the callback. Do not use stale neuron
indices or copy dynamic state into a module global. Static learned parameters
must be initialized within the paid prediction and included in complete bills.

Zero U is an algebraic identity proposal, not a measured byte-identity claim.
Required next evidence before a fit: faithful whole-source inverse for the
constructor and unchanged fallback; independent small-cell arithmetic and
parameter-gradient fixtures; native disabled/zero, cold/warm/repeat outputs,
domains, selectors, full own bills and18-object retirement; and qualification
of the new explicit CPU training translation with all old values frozen.
Ordinary differentiable products remain an approximate proposal engine;
component gradients do not certify the full native derivative.

No new fit schedule, terminal payload, acceptance result or numerical launch
is selected here. Fix the full training protocol and terminal-only selection
before opening targets for any new fit. A later candidate still needs the
complete native40, downstream two-group and actual-client release gates.
All added operations must be paid;1536 is a parameter count, not a complete
cost estimate. Extra messages duplicate work unless a separate exact refactor
is qualified. No accuracy, runtime or memory improvement is assumed.

Use one nice10 CPU worker, six BLAS thread limits1,7.5GB native/training RSS,
zero added swap and the existing470GB disk floor. Reuse existing inputs;
no new datasets, sampling or large cached trajectories are proposed.
