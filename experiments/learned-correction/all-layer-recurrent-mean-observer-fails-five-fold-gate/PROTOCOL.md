# Fixed-trajectory mean observer with all-late-layer supervision

Phase2 only, prospective2026-09-20 before new captures, target decoding or fit.
Official parent331684 is3.5844144212297072e-9 with100/100 complete. Its exact
packed source is facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
Keep the entire physical trajectory, covariance edge model, original GNN,
readouts and coefficients frozen. Only a separate final mean correction is
proposed. The approximately1e-9 goal remains; this is no success redefinition.

Prior scopes: lazy_frozen_trajectory_observer_v1 and the older v29 observer
trained neural recurrences from final-only1B means. The local mean-innovation
MLP used8M coherent mean/covariance references, failed its TRAIN proxy gate,
and its later full-lower physical endpoint was4.28% worse. The covariance-head
linear-response and direct SPSA arms are closed negatives. None is retrained,
extended or used as initialization. Here a small, explicitly defined linear
observer learns combined mean-error sources from1B mean labels at EVERY late
layer; no true covariance or teacher state appears at inference. This is a
different target, supervision, host and consumer, not a claimed new family or
proof that reference noise explains earlier failures.

TRAIN20=[36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,60].
CHECK=[35,100]. Panel14=[61,62,64,65,68,69,70,72,74,76,77,78,80,81].
Confirmation8=[37,39,42,48,59,63,66,67]. All are reused public research
networks, disjoint across this refit's roles, not pristine holdouts. Use the
existing original1B mc_post files and canonical observer_fit weights. No
protected688/audit/lockbox data, new samples, downloads or dataset copies.

Passive capture: retain existing F30,H64,sigma and parent predictions at all
16 layers via original gnn_step references, without added counted arithmetic.
First unwrapped35/100/35 and independent cold100, then wrapped same lifecycle:
require all16 outputs and complete bills exact, own repeat captures exact and
warm/cold100 fields exact. Require finite arrays, retired original caches and
unchanged source. Acquire20 cold TRAIN captures, each exact to a pinned native
parent output/bill (run an unwrapped parent first if no qualified one exists).
Store no dense covariance/carrier tapes or weight copies. Seal all captures
before label decoding. Capture qualification320s/controller2400s; each cold120s.

Observer definition: r_l=[F_l,H_l],94 existing FP32 coordinates widenedFP64.
For l8..15, X_l=[(r_l-mu)/scale,one_hot(l-8)],102 columns. Fit-only mu is the
coordinate mean over all8 late layers/all neurons/all fit networks; scale is
population standard deviation floored1e-6. No clipping or feature selection.
Let g_l be F_l[:,5] widenedFP64 and sigma_l the captured originalFP64 value.
Start d7=0 and define d_l=g_l*(W_l.T@d_(l-1))+sigma_l*(X_l@beta).
At8 the zero transport is omitted. This is the observer's chosen recurrence,
not an exact native Jacobian or full mean/covariance tangent. The source may
represent accumulated covariance/higher-moment effects; it is not labeled a
pure local physical mean innovation. Parent state never consumes d_l.
Output is max(parent served final+d15,0),FP32; earlier15 rows stay parent.
New arithmetic and model coefficients/normalizers are FP64; no change to any
parent precision. Beta has102 entries. Deployment could use94 standardized
features plus one layer bias instead of constructing a one-hot matrix.

Build raw design B7=0,B_l=g_l[:,None]*(W_l.T@B_(l-1))+
sigma_l[:,None]*[r_l,one_hot(l-8)]. A local control omits all transport.
Per-fold normalization transforms continuous columns as
(B_j-mu_j*sum(B_layer_columns))/scale_j. This equivalence must be verified
against direct normalized recurrence and independent scalar tiny equations.
The deployed sequential FP64 recurrence must match the design action before
targets, with actual FNP tiny units, zero-beta parent identity and fixed
nonzero beta. This tests the observer definition, not a physical derivative.

Five folds by TRAIN ordinal modulo5:16fit/4held. For EACH fold fit both the
recurrent observer and the local control on all8 late post-mean errors,
truth_mc_post[l]-parent_served[l], in ordinary FP64. Weight every layer/neuron
by its network's parent cold multiplier; normalize total weights to1. Each
design column receives fit-only weighted RMS normalization floored1e-12.
Solve the unique ridge system X.T*w*X+1e-3 I; no coefficient exclusion or
unpenalized intercept. Preserve all10 fits and normal-equation certificates.
An independent augmented least-squares fixture must verify the solver.
No nonlinear model, search, optimizer, strength, layer, loss-weight or ridge
rescue; no selecting a different endpoint after any result.

Score held final predictions after FP32 clamp. Proposed adjusted cost is
parent complete cold bill+1GF for either model, a conservative UNQUALIFIED
allowance until native measurement. Require recurrent pooled adjusted/parent
<=.95, recurrent/control<=.99,>=4/5 parent-improving folds,>=12/20 individual
parent wins and worst individual<=1.20. These are exploratory network CV, not
new population or official scores. Failure closes this arm with no full fit,
CHECK, native port, panel or upload. On a pass, fit ONE all20 recurrent model,
freeze/reload parameters, and seal CHECK35/100 predictions before opening
their1B means. CHECK pooled adjusted<=.97 and each<=1.02 with the same1GF
allowance are required. No full-control deployment fit or blending.

Only a complete pass licenses actual native integration: disabled parent
identity, enabled35/100/35+cold100, exact earlier15 parent rows, own repeat
and warm/cold outputs, complete measured increment<=1GF and cached reference
parity (max absolute<=2e-7). Require actual own-cold CHECK gates as above.
Then panel14 needs pooled<1,>=8wins,worst<=1.10; confirmation8 pooled<1,
>=5wins,worst<=1.10. NumPy-free parity, every local residual<=.25s, matched
wall pooled<=1.075/each<=1.10,contract/package/exact source-archive checks
precede qualified upload under standing authorization. Official grade and
100-network completion decide banking. No upload from proxy or CV alone.

One nice10 numerical worker, six BLAS limits1,8GB RSS,zero added swap; existing
256MiB emergency stop. No GPU. Research design unit90s,design acquisition480s,
CV180s (extend only for diagnosed infrastructure,not extra fitting),CHECK60s.
Artifacts<=500MB; at most100 full calls for this finite study. Sources for each
stage must be pinned before execution; all fit/model rules above are fixed
before any targets. Preserve every old/new failed receipt and source version.
