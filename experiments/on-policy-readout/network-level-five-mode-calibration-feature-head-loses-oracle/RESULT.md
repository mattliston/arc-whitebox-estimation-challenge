# Fixed network-level correction does not qualify

Across four network-separated folds on reused FIT128, the whole-network
feature model changes raw MSE by **+0.111002%**, with54/128 wins. Its95%
paired-network interval is[-0.005283%,+0.229964%]. Three of four folds lose.
The five-coefficient intercept control changes raw MSE by**-0.055768%**,
with76/128 wins and interval[-0.107402%,-0.003881%]. Neither approaches
the fixed3% development threshold. There is no survivor or native port.

The diagnostic that fits these same five response modes independently to
each network's true answers removes only**1.548859%** of pooled error.
This is an unattainable, unclipped linear-span fit on these development
cases, not a bound for all global models or a deployable estimator.
It offers little reason to spend further work on the tested calibration.
No ridge, mode, feature, amplitude or sign rescue follows.

The new operation pools the94 existing control features over neurons and
uses a fixed ridge model to predict five coefficients shared by each
network. Each fold trains on96 networks and freezes both models before
reading its32 test feature packets. A separate scorer opens development
labels only after all four fitting processes have retired. CHECK32 and
VALID40 are untouched. These reused development results are not an
independent validation claim. No extra simulation or estimator call ran.

Synthetic explicit-design and block-normal-equation solutions agree to
8.33e-17; zero targets, neuron permutation and mode whitening also pass.
An independent post-score verifier reconstructs the saved models using
physical response coefficients and reconciles all512 scores with scalar
math.fsum. Its exact measurements and resource receipt are in
verification.json and verification_guard.json.

The sources and input bindings were sealed before fitting. Original source
review corrections are recorded in PROTOCOL.md: the full inverse-K21
closure and einsum backend idea both had historical predecessors. Neither
was rerun in this study. The numerical finding here is the new fixed
network-level readout test, not a rediscovered old experiment.

S31#332517 remains2.1736465056908115e-9 adjusted. No new banked gain,
submission, validation run or approval request. Resource and artifact
totals are in closeout.json. The approximately1e-9 goal remains active.
