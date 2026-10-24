# Fixed eight-probe masked CP field covariance supplier

Admitted 2026-09-27 before fresh feature capture, fit, or endpoint observation.
One candidate only, applied at post-GNN covariance hooks in layers 8 through 14.
The official bank remains S31, adjusted 2.1736465056908115e-9; the target remains
approximately 1e-9 and the next milestone is 2e-9. No improvement is assumed.

## Field and scope

Use the masked incoming CP component already resident in _E2_LAYER pp and lam.
For two independent sequential 1024-by-8 Rademacher draws from NumPy default_rng
seed 2026092767, save immutable FP32 literals Omega and Psi, with entries +/-1.
Do not scale signs by sqrt(1024). Define invsigma=(1/sigma).astype(float32),
A=pp.T@(Omega*invsigma[:,None]), B=pp.T@(Psi*invsigma[:,None]),
Z=(pp@(lam.astype(float32)[:,None]*A*B))*invsigma[:,None],
and G=(Z@Z.T)/float32(8). The exact operation order and FP32 intermediates
are part of the source definition. Runtime uses paid fnp ordinary matrix
products for the width-eight contractions; no unsupported skinny _mm call.

This observes the numerically reconstructed masked CP component. It contains
repeated faces and distinct-index information; it excludes star, b trace, dead
rows and omitted old-tail rows repaired separately by other consumers.
No subtraction or claim of an exact fourth-cumulant/ReLU identity is made.
Expectation over independent signs is its standardized tensor-slice Gram,
but this one fixed rank-eight proxy is noisy. It is new information absent
from local repeated-face features and the failed RR-transpose feature.

Choose all seven layers before inspecting new values. This reaches multiple
covariance corrections but costs approximately 2.35% in field arithmetic plus
about 2.31% for the supplier, before the thin-star component saving (~0.57%
adjusted alone). A raw improvement around 6% may be needed to bank 2%.
These are admission estimates; actual complete native billing decides.

## Data and one fit

Reuse checkpoint47's TRAIN MINI57,68,45,59,3,12, seven layers each, existing
S31 recurrent-covariance packets and finite-native one-predecessor innovation
labels. Identity and weight pins, rather than numerically matching MINI to
observer-rank identifiers, select networks. No CHECK, generic TRAIN endpoint
targets, VALID targets, extra reference samples or new counterfactual labels
enter fitting.

Fresh captures run exact S31 with passive native NumPy work at the same hook.
Store only Z (32KiB per layer), plus provenance. Require all16 returned bytes
and paid bill equal the existing S31 captures for all six networks; a source
inverse must recover S31. Do not persist factor planes or create large datasets.
Training uses S31 fixed-state features; deployment on thin with recurrent own
state is an acknowledged approximation, not exact physical unrolled training.

Reuse the failed47/64 pair model architecture: raw30+H64 node inputs, local
R/S/S-transpose and new G as four pair inputs, hidden32 tanh, symmetric output
scale .002*s_i*s_j, and exact zero-diagonal/current-mean-annihilating projector.
Fresh 6209 weights; original parent parameters remain frozen.
Use exact old47 TRAIN-only F/R/S normalizers, and one TRAIN-only off-diagonal
RMS for G, floor1e-5. Clip normalized node inputs to [-8,8].
Fresh initialization/shuffle seed2026092767, 40 epochs, 1680 updates, AdamW
learning rate .001 cosine to .00001, weight_decay1e-4, gradient cap5.
Loss is .5 normalized covariance innovation error + .5 normalized next Gaussian
mean-response error + .001 normalized correction energy. The next Gaussian
response is an objective, not a native physical Jacobian claim.
Freeze the final weights; no best-epoch, amplitude/layer/rank/seed/hyperparameter
rescue or second fit. Every finite completed fit proceeds to the actual endpoint.

## Native fidelity and score

Deploy on exact thin-star source SHA256
25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee.
Use ordinary FP32 fnp for the pair supplier and new width-eight products.
Checkpoint64's depth-six repair applied only to its dense RR-transpose Gram;
that operation is absent here, so no depth-six/depth-seven variant is relevant.
Probes and trained values
are embedded literals with no external runtime data or cross-network cache.

Before fitting, a tiny gradient/projector fixture must pass. Verify probe
contractions independently against a tiny explicit tensor fixture and native
fnp implementation. Full-width model-helper comparison uses the saved Z and
frozen training predictions, including positive and signed/asymmetric fixtures.
Relative L2 tolerance 2e-5 for full native supplier versus frozen Torch,
absolute symmetry/diagonal/projector invariants at qualified old64 tolerances.
Disabled all16 output bytes and full bills on generic TRAIN ranks35/100 must
equal thin controls. Preserve every implementation failure; a concrete source
fix may be versioned but scientific feature/model choices remain fixed.

Actual complete billed screen against S31: TRAIN35/100 needs adjusted gain>=1%,
both wins, maximum raw regression<=5%. A survivor proceeds to
TRAIN36,45,52,58,64,70; combinedTRAIN8 needs adjusted gain>=2%, at least6 wins,
max adjusted regression<=5%, paired95% upper bound<0. These are reused
development networks. Failure closes this candidate without further tuning.
Only a survivor receives matchedVALID40 with >=2% adjusted gain, >=30 wins,
paired95% upper bound<0, then actual20worker, setup, boundary, runtime and
package checks. User has authorized submission after qualifying checks;
no additional submission permission is required.

Root alone owns numerical jobs, serial under nice10, six thread limits1,
7.5GB process-group RSS guard and zero added swap acceptance. No job restarts
on observation timeout. Immutable exclusive output files and hash seals preserve
sources, features, fit, failures and results. No deletion or .git mutation.
