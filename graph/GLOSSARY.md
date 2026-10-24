# Glossary

Definitions are tied to the archived implementation and research records. Unstated acronym expansions remain explicit; inferred interpretations are labeled.

## _gop2_gnn_step (gop2)

The per-layer function that computes the learned mean correction and updated learned state. The designated file contains multiple successive definitions; the final live definition begins at line 25662 and incorporates the 961-column on-policy readout. The name's letters are not expanded in the source.

Expansion: gop2 not expanded in the record.

Evidence status: inferred.

- [estimator.py#L17745](../release/337641/extracted/estimator.py#L17745)
- [estimator.py#L25531](../release/337641/extracted/estimator.py#L25531)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).
- [README.md#L27](../release/337641/README.md#L27)
- [estimator.py#L25662-25694](../release/337641/extracted/estimator.py#L25662-25694)

## arc-cosine kernel (Cho-Saul)

For zero-mean jointly Gaussian pre-activations, the ReLU second moment is E[ReLU(z_i) ReLU(z_j)] = sigma_i sigma_j [sqrt(1-rho^2) + (pi-arccos rho) rho] / (2 pi). This is a reference Gaussian pair formula used in the research. The designated implementation's first-layer branch evaluates Gaussian Hermite terms through correlation order 6; later pair maps use terms through order 3 plus cumulant corrections.

Evidence status: defined.

- Frozen-source reference: `explainers/33_one_move_many_names.html` (outside the copied Phase 2 evidence set).
- [ANGULAR_LABEL_TANGENT_AND_INNOVATION_SCOPE_V1.md#L154](../records/research/2026-09-07-exact-gaussian-angular-energy-initialization-leaves-nonlinear-transport-mean/ANGULAR_LABEL_TANGENT_AND_INNOVATION_SCOPE_V1.md#L154)
- [FIRST_LAYER_HOMOGENEOUS_QUADRATIC_CONTROL_PREFLIGHT_V1.md#L39](../records/research/2026-09-07-first-layer-homogeneous-quadratic-controls-have-exact-means-but/FIRST_LAYER_HOMOGENEOUS_QUADRATIC_CONTROL_PREFLIGHT_V1.md#L39)
- [estimator.py#L16959-16978](../release/337641/extracted/estimator.py#L16959-16978)
- [estimator.py#L829-833](../release/337641/extracted/estimator.py#L829-833)
- [PROTOCOL.md#L42-47](../records/research/2026-09-10-first-layer-quadratic-sampling-control-stops-after-48-64/PROTOCOL.md#L42-47)

## b carrier (collective gain mode, Sym(C x b))

The collective gain mode: for a given input the whole previous layer is a little larger or smaller than usual, scaling every unit's fluctuation together; it removes all of the mean excess kurtosis and ~87% of the per-unit skew variance. The card carries its third-order part as a vector b (the trace vector t_a = sum_c K3[a,c,c] = Cov(x_a, squared norm)) in the symmetrised form Sym(C x b); the card's b holds 20-30% of the gain part, the bank and star the rest. The card's harmful error is uncorrelated with the gain part.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L31-32` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L50-55` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L101-111` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L486` (outside the copied Phase 2 evidence set).
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)

## beta (birth beta) / gated beta / gbeta2 / K15

beta = 2 (gaussian_mean - host*g) is the birth coefficient of the pair's cross term (born = bp^2 (birth*births) + (beta*births) bp^T); gbeta2 is a physics change adding -2 host (gq - g); the gated beta (K15) multiplied that change by a per-network gate of the kink fraction f = mean(|alpha| < 1.5); every form was null at n = 24 and the beta site was closed 2026-10-13.

Evidence status: defined.

- [estimator.py#L524-536](../release/337641/extracted/estimator.py#L524-536)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1935-1943` (outside the copied Phase 2 evidence set).

## BFIX gamma (b fix)

A zero-cost repartition of the gain term between the b carrier and the plane: after the live-loop b update, b += _BFIX_GAMMA * (radial_new - radial)/6 * m. Built 2026-09-22 (porter), shipped at gamma 0.1 since S2 (line 3946), moved to 0.119 (the quadratic vertex, -0.05%) by the opus-solo zb4 stack (line 19837), which is the designated files' value; 0.14 flat on the final stack.

Expansion: b fix = the gain-mixture fix of the b carrier.

Evidence status: defined.

- [estimator.py#L3419](../release/337641/extracted/estimator.py#L3419)
- [estimator.py#L3945](../release/337641/extracted/estimator.py#L3945)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L1` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4255` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L74` (outside the copied Phase 2 evidence set).
- [estimator.py#L3946](../release/337641/extracted/estimator.py#L3946)
- [estimator.py#L19837](../release/337641/extracted/estimator.py#L19837)

## bill / utilisation / adjusted / raw / final-layer MSE / all-layer MSE

bill = the metered FLOPs of one prediction; utilisation = bill / 2^41; the score floors this factor at 0.1; raw = the final-layer MSE of the 1024 unit means against the organizers' 1e9-sample truth; adjusted = the per-network product MSE x max(0.1, utilisation); the official score is the MEAN over networks of the per-network adjusted values (not mean MSE x mean utilisation); all-layer MSE = the error averaged over layers 0-14, published but not scored.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L8-10` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L48` (outside the copied Phase 2 evidence set).
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)

## birth / hub / newborn

Birth: at each layer every unit h near its switching point (selected by min(g, 1-g) >= the gate floor) mints a pair of new bank columns p+- = s_h + t+- e_h, with s_h = g o Cpre[:, h] (zero own entry), e_h the unit vector of h, t+ and t- the pair's two weights and gap = t+ - t-; the unit is the pair's hub, this layer's pairs are the newborn cohort, and six numbers per pair (the hub coefficients) are kept from birth. Fitted birth coefficients scale the newborn weights (worth about a quarter of the physics-only error).

Evidence status: defined.

- [K4L_SPEC.md#L14](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L14)
- [NCAR6_SPEC.md#L11-15](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L11-15)
- [estimator.py#L508-536](../release/337641/extracted/estimator.py#L508-536)
- [estimator.py#L1601-1605](../release/337641/extracted/estimator.py#L1601-1605)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L188-189` (outside the copied Phase 2 evidence set).

## bsk (_BSK, skew-corrected birth)

A correction of the birth coefficient: the kink density (the birth weight d = pdf/sigma) taken to first order in the unit's skewness (opus-engineer bsk, birth = _bsk(birth, aux)); the one refit-base term that helped with the learned parts on (bsk -0.57% torch, -0.34% native on S42); _BSK = 0.434 in the designated files (shallow bowl, optimum ~0.45).

Expansion: bsk = birth skew correction (by its uses: 'the skew-corrected birth').

Evidence status: defined.

- [estimator.py#L3417](../release/337641/extracted/estimator.py#L3417)
- [estimator.py#L24597](../release/337641/extracted/estimator.py#L24597)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L6017` (outside the copied Phase 2 evidence set).

## c0 / c1 (K31 row factor)

The two constants of the per-row K31 scale rowf = c0 + c1 * mu*k3/v^2 applied at layers 1-14 (physicist, 2026-09-22: 0.85/0.5 provisional, 0.8/0.5 best on TRAIN); c1 was frozen at 0.5 in the joint refits while c0 moved (0.709 -> 0.694 in loopR3; 0.6222 in S41's joint dial solve); the designated files carry _K31R_TABLE = (0.5426, 0.5027) at every layer.

Evidence status: defined.

- [estimator.py#L3929-3935](../release/337641/extracted/estimator.py#L3929-3935)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L840` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L596-597` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L23` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L27` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L31` (outside the copied Phase 2 evidence set).
- [estimator.py#L24594](../release/337641/extracted/estimator.py#L24594)

## c2 card

Astra's card of mid September 2026: with the wall repair (serial-top D6 push kernel replaced by a D4 packed kernel plus a square depth cap of 4, after c2 timed out on the platform at 54-58 s local boundary wall) it was graded #330931 = 4.188e-9 on 2026-09-13 (raw 2.130e-8, util 0.1967, no timeouts), the new bank. Mapped 2026-09-16: bill 87% third-order bank (young cube-column push 31%, bank plane reads 28%, old-tier QR/projection 13%, cubic star push 10%), covariance push 8%; the final error is born in layers 12-15; every rank/age/gate dial loses; true K21+K22 planes at L12-15 would give -25% but every partial supply is poison.

Expansion: Recorded label for the mid-September estimator; literal expansion is not stated.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1899-1920` (outside the copied Phase 2 evidence set).

## c2sub

A warm-started subspace iteration (16-32 columns) producing the (C^2)_ii regressor so that the exact multiple-of-C^2 form can be removed in the pair law's diagonal regression; ~1.3% of bill for ~1.2 points, closed 2026-10-02 (e60_c2sub).

Expansion: C-squared subspace.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L584-587` (outside the copied Phase 2 evidence set).

## card

The team's name for the deterministic moment-closure estimator file. Layer by layer it carries the per-unit mean m, the full 1024x1024 covariance C, the third-order cumulant tensor as a bank of rank-one columns plus a b carrier and a plane-only remainder (the star), and one fourth-order scalar per layer (radial); a pair map turns (mu, C_pre, K21, k4, K22) into the post-ReLU mean and covariance, and a learned per-layer correction adjusts the mean. 'Torch card' is the PyTorch research copy, 'native' the flopscope file that is scored.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L29-37` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L51` (outside the copied Phase 2 evidence set).
- [README.md#L20](../release/ARCHIVE_README.md#L20)
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)

## carrier

A finite representation of cumulant content transported between layers, such as signed bank columns, the collective-gain b vector, a star plane, or norm-coupling partner vectors. Early Phase 2 cards capped rank-one cumulant terms; later files combine explicit young cohorts with a compressed shared frame for older cohorts.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L31-32` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L54` (outside the copied Phase 2 evidence set).
- [NCAR_SPEC.md#L1](../experiments/fourth-order-terms/norm-coupling-carrier-16-partner-vectors-lowers-mse-2/NCAR_SPEC.md#L1)
- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L130-134` (outside the copied Phase 2 evidence set).

## census (bill census, key census)

An attribution of every metered charge (or every tree key or request) of one prediction to its step, site or layer: the S50 census (bank push 44.4%, bank read 35.3%, dead-cut star 9.1%, L15 row-cut k3 3.5%, head 1.5%, pair map 1.1%, pair-law left 1.1%); the key census over selection networks builds the prebind list.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L36` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L431` (outside the copied Phase 2 evidence set).
- [estimator.py#L6515](../release/337641/extracted/estimator.py#L6515)

## centered22

The designated pair map constructs the centered-square derivative rows bb - (2 m / sigma) aa and uses them in both factors of channel 22. This computes the fourth-order centered pair contribution within the series evaluation. It is separate from the K22 input cumulant plane.

Expansion: Centering of the pair map's (2,2) channel.

Evidence status: defined.

- [estimator.py#L889-899](../release/337641/extracted/estimator.py#L889-899)
- [estimator.py#L958-968](../release/337641/extracted/estimator.py#L958-968)

## closure (moment closure)

An estimator that never runs the network on samples: it carries a finite set of distribution summaries (cumulants: mean, covariance, third and fourth cumulants) and pushes them through each layer with deterministic maps, the linear part exactly and the ReLU by Gaussian closed forms plus correction terms. 'Closing' means truncating the infinite cumulant hierarchy by inventing the dropped higher cumulants. Every closure is a representation plus a propagation rule plus a readout.

Evidence status: defined.

- Frozen-source reference: `explainers/24_what_a_closure_is.html` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L32-50` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).
- [README.md#L20](../release/ARCHIVE_README.md#L20)

## coefH

The pair-law setup constructs coefH from the newborn plus/minus diagonal weights and the two signed cross weights. Its expressions combine the birth, row-form, and diagonal coefficients (_NP_BIRTHS, _NP_ROWS, _NP_DIAG) with hub statistics, pair roots, and layer scales. Later calibration experiments regenerate this table when those coefficients change.

Expansion: Polynomial coefficient table for pair-carried newborn weights.

Evidence status: defined.

- [estimator.py#L11493-11530](../release/337641/extracted/estimator.py#L11493-11530)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L5913` (outside the copied Phase 2 evidence set).

## cohort / age / young / old tier

A cohort is the set of bank columns born at one layer; its age is the number of layers since birth. Young columns are the cohorts not yet joined (ages <= 4), pushed and read exactly; the old tier is the cohorts aged >= 5, compressed into the shared rank-384 frame at the joins (layer 7 takes cohorts 0-2, layer 10 cohorts 3-5, layer 13 cohorts 6-8). The old third-order content is worth 18-29x in raw error; the old tier plus its reads are ~35% of the bill.

Evidence status: defined.

- [K4L_SPEC.md#L34](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L34)
- [estimator.py#L2637-2643](../release/337641/extracted/estimator.py#L2637-2643)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L31` (outside the copied Phase 2 evidence set).

## CP3 / CP4 carriers and carrier caps (cap256, cap512, cap2048)

In the early Phase 2 cards (September 1-16) the third- and fourth-order cumulant tensors were carried as sums of rank-one legs with a cap on the number of legs kept by energy (CP3_CAP 8192-16384 legs; CP4_CAP 128-2048, the cap256 card = D11 + honest closure + CP4 cap 256). The cap sweep varies the number of retained legs: on the deployed lineage CP3_CAP 16384 gave raw +38.6% at util 0.554 and 8192 +351% (caps are co-adapted and cannot be cut for bill). The CP4 top-energy cap dropped unit-vector births (the 'kurtosis cap defect', 2026-09-02). The October bank of signed column pairs replaced the capped CP carriers.

Expansion: Third- and fourth-order CP tensor carriers, represented as sums of rank-one terms.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1925-1942` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_cp3_energy.py#L31-35` (outside the copied Phase 2 evidence set).
- [MECHANISM_TREE.md#L39](../experiments/error-attribution/exact-variance-correction-error-1-466e-8-versus-deployed/MECHANISM_TREE.md#L39)
- [LANE_g3_tracker.md#L341-343](../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L341-343)

## D11 (card lineage)

The deployed estimator lineage of early September 2026 (the 'old D11 recursion', directory public_d11_util_halving_v1): CAP256_v2 = D11 + honest closure + CP4 cap 256 with the variance head off was the board best #329754 (adjusted 1.6116e-8, raw 2.965e-8, util 0.5435, rank ~22) on 2026-09-07; its cap ladder and T-69/T-69b plane-feed records are cited by later lanes.

Expansion: not expanded in the record (a design/lineage label of the early September estimator).

Evidence status: inferred.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1942` (outside the copied Phase 2 evidence set).
- [LANE_g3_tracker.md#L343](../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L343)
- [CENTERED_FEEDBACK_ROUND_V1.md#L13](../experiments/learned-correction/centered-feedback-convention-closed-after-matched-closed-loop-test/CENTERED_FEEDBACK_ROUND_V1.md#L13)
- [PROJECTED_PLANE_FEED_ROUND_V1.md#L3](../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/PROJECTED_PLANE_FEED_ROUND_V1.md#L3)

## D7 / D7SQ

D7 executes eligible wide products as 2048-wide column packets sharing a depth-7 A transform; a final packet is zero-padded to a multiple of 128 when needed. The depth names refer to matrix-multiplication recursion, not network layers. D7SQ is the separately controlled square-product depth-7 route.

Expansion: Depth-7 recursive matrix-product routes.

Evidence status: defined.

- [estimator.py#L7896-7913](../release/337641/extracted/estimator.py#L7896-7913)
- [estimator.py#L17816-17827](../release/337641/extracted/estimator.py#L17816-17827)
- [estimator.py#L7980-7999](../release/337641/extracted/estimator.py#L7980-7999)

## dead cut (dead-side row cut, _DEADCUT_Z) / DCV (dead-cut covariance) / dead-cut star

Two related cuts of dead units (units far below their threshold). (1) The dead-side row cut of the push (porter PT-15, 2026-09-24): the rows kept at layers 1-14 are select(a + z, 0), i.e. units with alpha > -z; z = 3.5 originally, 3.0902 (gate 1e-3) in the designated files; DC35 = z 3.5 at layers 8-14 was dead on store ranks. (2) DCV (a44, 2026-10-10): the covariance sandwich W_a' (C_aa W_a) and the star product run on the live roster only (gate >= _DCV_FLOOR = 0.001, rounded to 64, 'pad' mode, later at its true inner size), dropping 0.6-2.0e-3 of C_pre's Frobenius norm, nearly all cross covariances of the cut units; the 'dead-cut star' is that reduced covariance + star product (~9% of the bill).

Expansion: DCV = the dead-cut covariance sandwich (by its code comments).

Evidence status: defined.

- [estimator.py#L5360-5362](../release/337641/extracted/estimator.py#L5360-5362)
- [estimator.py#L13439](../release/337641/extracted/estimator.py#L13439)
- [estimator.py#L12428-12436](../release/337641/extracted/estimator.py#L12428-12436)
- [estimator.py#L13303-13309](../release/337641/extracted/estimator.py#L13303-13309)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L672` (outside the copied Phase 2 evidence set).
- [estimator.py#L13305](../release/337641/extracted/estimator.py#L13305)
- [estimator.py#L3279](../release/337641/extracted/estimator.py#L3279)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L51` (outside the copied Phase 2 evidence set).

## detached star (star basis)

The rank-32 basis in which the star plane is carried ('star-Nystrom basis'). 'Detached' describes refits that back-propagated with that basis held constant, which misses 60-85% of the derivative for dials acting through the carried b and can have the wrong sign; run M used the differentiated basis.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L393-396` (outside the copied Phase 2 evidence set).

## dial

A scalar coefficient of the card set by finite-difference line search (or exact dial gradients) on held-out networks rather than derived: about ten physics dials (bscale, b-source, X0/X1/X2, s_carry, s_new, c0, k21n, the k4 bands, F4 theta, K4L c, GSK, BFIX gamma, SS beta, NP strength, row-cut z ...). A 'zero-bill dial' changes no FLOPs. Per-layer dial tables did not transfer across networks (E13); dial wins on the base file vanish on the readout stack.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L35` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L259-263` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L448-455` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L86` (outside the copied Phase 2 evidence set).

## E1 / E2 / E3 and M1 / M2 / M3

The message transform uses elementwise powers W, W^2, and W^3 on three learned source blocks. The energy transform uses W^2, W^4, and W^6 on their squared source values, then applies nonnegative clipping and log1p before the encoder. Thus M2 is the W^2 message, and E3 is the W^6 energy block. GOP2 applies rank-2 contraction to all energy blocks and to M2. Names such as E3 can also denote unrelated experiment identifiers; this definition applies to the learned-feature context.

Expansion: Three energy blocks and three weight-power message blocks of the learned correction.

Evidence status: defined.

- [estimator.py#L142-163](../release/337641/extracted/estimator.py#L142-163)
- [estimator.py#L17720-17761](../release/337641/extracted/estimator.py#L17720-17761)

## edge module (learned covariance edge, ecut, _GN_EDGE)

A small learned covariance correction at layers 8-14 (the 'message edge'): a low-rank update of the post covariance from per-unit messages through a tanh nonlinearity, restricted (the edge cut) to the response rows x live columns the covariance edge keeps; worth ~1% MSE for ~1.2% of bill; the exact all-rows version carries nothing more. The tanh was replaced by a rational stand-in (_O20_ECUT_RADICAL_A/B), a measured approximation.

Expansion: ecut = edge cut (the rosters the covariance edge keeps).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L34-35` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L404-405` (outside the copied Phase 2 evidence set).
- [estimator.py#L3305](../release/337641/extracted/estimator.py#L3305)
- [estimator.py#L25328-25330](../release/337641/extracted/estimator.py#L25328-25330)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).

## exact switch / L14 exactness (OTT, YNT, RTT, RC)

An exact switch turns one of the card's cost-saving approximations off (ECUTX, READX, STARX, LCRX, J3R): mid-layer gains that do not reach layer 15 (10-16 ladder). L14 exactness restores approximations only at the last layers, where the pushed columns feed the final k3 directly: YNT (crop z 2.5 -> 3.0 at the L13 and L14 pushes) and OTT (old-tail positive side 3.0 at the L14 push) paid (#337126 -0.10%, #337142 -0.03%); RTT (response-row floor z 1.5, a sharp optimum) and RC (read column cut, _RC_COLZ 2.0) restores did not; the stack is raw-driven and transferred at ~3/4.

Expansion: OTT = old positive-tail diagonal approximation; YNT = young negative trace crop; RTT = the read's response-row floor; RC = read column cut.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L5953-5960` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L704` (outside the copied Phase 2 evidence set).
- [estimator.py#L24556-24587](../release/337641/extracted/estimator.py#L24556-24587)

## F2J

A learned correction fitted to an early Phase 2 closure's residual features, with a per-neuron state transported by the weight matrix and gates and a learned source map from native features. Its value and co-adaptation were measured separately from the deterministic host. The label should not be substituted for the final October readout architecture.

Expansion: Recorded label for an early learned residual operator; literal expansion is not stated.

Evidence status: defined.

- [MECHANISM_TREE.md#L35-36](../experiments/error-attribution/exact-variance-correction-error-1-466e-8-versus-deployed/MECHANISM_TREE.md#L35-36)

## F32_COV

The covariance push W^T C W computed in float32 instead of float64 (flopscope bills float64 at twice the rate): identical error on ranks 35/100 at -7% util (T-138, 2026-09-07); carried into every later card.

Expansion: float32 covariance push.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1935` (outside the copied Phase 2 evidence set).

## F4 / F4C / cp4

Every bank column stores one scalar from its birth, A = -alpha / (sigma (1 + radial/3)) of its hub, so the columns carry the Gaussian-born fourth-order star Q = sum_r theta lam_r A_r p_r^4 for free; its feed into the next births is an in-place shift of the young carried columns after the gate, g o pp -> g o pp + theta A (d/2) o pp o pp (columns at most two layers old), with a cheap anchor removing the added tensor's plane from the star. F4 (theta 0.6) shipped in S37; F4C = the exact-coefficient mixture: _F4C_THETA = the strength of F4's rule per age ((0.704, 0.665)) and _F4C_C = the strengths of the exact S1 and S2 terms per age ((0, 0.459), (0, 0.258)), applied at the end of steps 1..14 to cohorts born 1 or 2 layers earlier, 21.5 flops per element of live rows x shifted columns. cp4 was the torch screen option name (not the CP4 carrier of early September).

Expansion: F4 = the physicist's fourth-cumulant feed term (candidate F4); cp4 = its screen option.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L265-278` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L343` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L349-350` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L46` (outside the copied Phase 2 evidence set).
- [estimator.py#L10638-10648](../release/337641/extracted/estimator.py#L10638-10648)

## FHM

At layers 1-14 the effective learned mean correction is d_eff = max(d, -m), ensuring m + d_eff >= 0. The covariance recentering uses the same effective correction. ZMS tests an additional covariance projection for the units clamped to zero.

Expansion: Recorded name for the fixed nonnegative-mean correction; the letters are not expanded.

Evidence status: defined.

- [estimator.py#L17316-17333](../release/337641/extracted/estimator.py#L17316-17333)

## fire / bank (submission) / S-number

To fire is to submit a file to the hosted grader; a bank (banked, bank step/level) is a graded submission that became the team's best public score; S<n> names the candidate generations of the September/October line (S35 = #333111 on 2026-10-01, S36 #333429, S37 #333435, S38 #333481, S39 #333528, S40 #333559, S41 #333587, ... S63 the last of the automated refit line).

Expansion: S<n> = the n-th candidate generation of the September/October line.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L16` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L288-291` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L363-366` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L371-372` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L466` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L528` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L577` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L596` (outside the copied Phase 2 evidence set).

## first network / boundary run / P(loss)

The first network is a worker's cold first prediction; a boundary run is the release lane's run of a candidate under the platform's setup clock plus a slow-worker stress to project wall, residual and requests; P(loss) is the modelled probability that a 100-network run loses at least one network to the 120 s wall or 0.4 s residual limits.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L289-290` (outside the copied Phase 2 evidence set).

## flopscope / fnp / BudgetContext / whestbench / whest

flopscope is the competition's metered array client (client-server; every array operation is a request billed by a weight table: matmul 2MNK, pointwise 1 per element, take 4, exp 16, transpose 0; float16 and float32 rate 1, float64 rate 2); fnp = flopscope.numpy; BudgetContext(flop_budget=...) is the metering context the grader opens per network (the file opens its own only in setup, namespaces 'prebind' and 'g5const', which is off-budget by the contract); whestbench is the harness (BaseEstimator contract: setup(context) once per worker, predict(mlp, budget) -> (depth, width) array; whest validate / package / run CLI; 0.14.0 packaged the release, 0.16.1 runs predict in a per-MLP BudgetContext).

Expansion: fnp = flopscope.numpy.

Evidence status: defined.

- [DEPENDENCY_MANIFEST.md#L7-8](../release/DEPENDENCY_MANIFEST.md#L7-8)
- [DEPENDENCY_MANIFEST.md#L17-21](../release/DEPENDENCY_MANIFEST.md#L17-21)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L25` (outside the copied Phase 2 evidence set).

## fragment

A self-contained appended code block (fragment_<name>_v<k>.py) that wraps or overrides functions of the parent file to add one lever (fragment_k4l34_v1, fragment_lazybind_v2, fragment_governor_v4g, fragment_smprune_r002 ...); round stacks chain fragments onto a fired file with chain_round_stack_v3/v5.sh, and a fragment must pass a zero test (bit for bit equal to the parent with its strength at 0).

Evidence status: defined.


## gate / gate floor / birth floor / response floor

The gate g = Phi(a) is the probability a unit's ReLU is open (the ReLU's linearisation), used to transport carried content through the ReLU. The card selects its rosters by thresholds on min(g, 1-g): the birth floor (which units become hubs), the response floor (which rows the read computes) and the alive floor, all 0.01 and measured at their optimum at every depth; late birth-floor raises are a cliff.

Evidence status: defined.

- [estimator.py#L510](../release/337641/extracted/estimator.py#L510)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L377-381` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).

## gate33

The final transport gate augments its linear skewness and kurtosis corrections by -t3^2 He5(alpha) phi(alpha)/72, where t3 = k3/sigma^3 and He5(alpha) = alpha^5 - 10 alpha^3 + 15 alpha. The source fuses these three polynomial terms into the existing gate evaluation. The label 33 identifies the product of two third-cumulant factors.

Expansion: Quadratic-skewness correction to the transport gate.

Evidence status: defined.

- [estimator.py#L23997-24025](../release/337641/extracted/estimator.py#L23997-24025)

## Gaussian closure

The naive closure that carries only (mean, covariance) and assumes jointly Gaussian pre-activations at every layer; its post-ReLU mean is sigma*phi(a) + mu*Phi(a) with a = mu/sigma. The card adds third- and fourth-order corrections on top of it.

Evidence status: defined.

- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L34-50` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/24_what_a_closure_is.html` (outside the copied Phase 2 evidence set).

## GC-quiet predict

gc.freeze of the setup object graph and a raised gen-2 collector threshold during predict (restored afterwards), to keep collector pauses (gen0/1 ~24 ms per network plus a growing end scan) out of the residual; interpreter plumbing, no computation; fired as v26g #336754 on 2026-10-17.

Expansion: GC = garbage collector.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L39` (outside the copied Phase 2 evidence set).
- [DEPENDENCY_MANIFEST.md#L11](../release/DEPENDENCY_MANIFEST.md#L11)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4879` (outside the copied Phase 2 evidence set).

## GMM4

A proposed endogenous propagation state with up to four positive masses, per-component means, and full covariance factors. A selected component is split before a nonfinal ReLU using a fixed three-node Gaussian rule; each child uses a Gaussian scalar moment map and order-12 Hermite covariance approximation. This is a separate proposed estimator family, not a four-component version of the signed cumulant bank.

Expansion: Positive Gaussian mixture with at most four components.

Evidence status: defined.

- [POSITIVE_GMM4_SPLIT_MERGE_DESIGN_V1.md#L36-92](../experiments/wall-and-residual/persistent-positive-four-component-gaussian-mixture-fixed-no-fit/POSITIVE_GMM4_SPLIT_MERGE_DESIGN_V1.md#L36-92)

## GNN payloads (gnn_hcap256, GNN_HCP3_8K)

The learned per-layer GNN closure corrections of early September, trained as co-adapted payloads for one card each: payload_gnn_hcap256.npz for the cap256 card (CP4 cap 256, F32_COV; cand_gnn_hcap256_f32cov.py reproduced the instrument) and payload_gnn_hcp3_8k.npz for the CP3-8192 card (chain_cp3_8k.sh, 2026-09-07); un-co-adapted caps break the closure.

Expansion: h = head/hook payload trained on the named closure (inferred); cap256 = CP4 cap 256 card; hcp3_8k = CP3 cap 8192 card.

Evidence status: inferred.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_cp3_energy.py#L35` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1925` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/HANDOFF_before_checkpoint15.md#L1935` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/README.md#L18` (outside the copied Phase 2 evidence set).

## GOP2

The implementation preserves the full mean component and projects centered features through a fixed two-dimensional output range. It applies this contraction to the three squared-weight energy blocks and the W^2 message; the W and W^3 messages initially retain their original products. A later response-weighted rank-4 compression changes W^3 independently. The on-policy readout overrides _gop2_gnn_step but retains this message structure.

Expansion: The code does not expand GOP2; it labels fixed rank-2 energy/message contractions.

Evidence status: defined.

- [estimator.py#L17691-17764](../release/337641/extracted/estimator.py#L17691-17764)
- [estimator.py#L19791-19800](../release/337641/extracted/estimator.py#L19791-19800)

## governor / fast rescue / rescue tiers / _FAST_RESCUE_FRAC

A residual-time and wall-triggered routing switch (2026-10-16, candidate_GOV_v3/v4e/v4f/v4g): it read the meter's live residual_wall_time_s and a wall projection and above thresholds (0.39/0.41/0.46 s; wall 112/118/123 s) moved the file to cheaper-in-time tiers: FULL (each covariance sandwich as one tree product), EXACT (flop-diet flags off: QL, compute, SM40, D7 ...), LEAN (learned hook, edge, K4L local term off: +36% MSE on that network), SHED (the young read returns 0: x40 MSE). The 'fast rescue' is the per-layer variant: an 'exact-algebra emergency executor' (_FAST_RESCUE_FRAC, _FAST_RESCUE_LAYER set by a runtime controller) that made _shared_old384_mm fall back to its parent route. Inert in native judges (no _active_context) and in the designated files (the map is empty at the file's last line), removed from #337641; retired by Matt's rule of 10-19.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L23` (outside the copied Phase 2 evidence set).
- [estimator.py#L22134-22137](../release/337641/extracted/estimator.py#L22134-22137)
- [README.md#L21](../release/337641/README.md#L21)
- [estimator.py#L22133-22140](../release/337641/extracted/estimator.py#L22133-22140)
- [estimator.py#L25714](../release/337641/extracted/estimator.py#L25714)

## Gram (symmetric Gram, Gram core)

A product of the form A^T A or J W J^T: the join's Nystrom sketch is built from the Gram of the joined columns, the old tier keeps a 'weighted Gram core', and _O20_GRAM_ON computes only the upper half of symmetric Gram products (exact).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).
- [estimator.py#L2620](../release/337641/extracted/estimator.py#L2620)

## Gram-Charlier / Edgeworth corrections

Cumulant corrections to Gaussian ReLU moment formulas. The scalar mean includes k3/6 and k4/24 times the corresponding Gaussian derivative expectations; the pair map includes mixed third- and fourth-cumulant planes. Experiments also considered terms quadratic in the third cumulant. The final transport gate includes the gate33 quadratic-skewness correction, so the earlier rejection of a scalar-mean correction does not imply every quadratic-skewness term is absent.

Evidence status: defined.

- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L63-65` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/24_what_a_closure_is.html` (outside the copied Phase 2 evidence set).
- [estimator.py#L988-995](../release/337641/extracted/estimator.py#L988-995)
- [estimator.py#L23997-24025](../release/337641/extracted/estimator.py#L23997-24025)

## GSK / GK4 / GBK (gk, transport gate)

The transport gate used to carry the inherited plane (gq_i^2 gq_j cp_ij and its diagonal), the gated young columns, the old tier's gated basis and the b carrier through the ReLU is corrected for the unit's skew and kurtosis: gq = Phi(a) + (t3/6) He2(a) phi - (t4/24) He3(a) phi, exactly d(host)/d(mu) of the card's own mean formula; one polynomial evaluation on nine per-unit rows, 5 requests a layer. _GSK and _GK4 scale the two corrections, _GBK (0) the cross-form feed. Values in the shipped file: the gk fragment sets GSK 1.0 / GK4 0.6, the opus-solo zero-bill stack zb4 sets GK4 1.0 / GSK 0.9, the 2026-10-19 release stack GK4 1.04 (platform read -0.082%); _GSK 0.8 was noise-level on the final stack.

Expansion: GSK = gate skew correction strength x_s; GK4 = gate kurtosis correction strength x_k; GBK = x_b, the feed of the cross form Sym(b (x) kappa3) into the carried third cumulant; gk = the arm with gsk = gk4 = 1.

Evidence status: defined.

- [estimator.py#L13942-13949](../release/337641/extracted/estimator.py#L13942-13949)
- [estimator.py#L13967](../release/337641/extracted/estimator.py#L13967)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1600` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1604` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L74` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L6128` (outside the copied Phase 2 evidence set).
- [estimator.py#L13936-13946](../release/337641/extracted/estimator.py#L13936-13946)
- [estimator.py#L13968-13970](../release/337641/extracted/estimator.py#L13968-13970)
- [estimator.py#L19833-19840](../release/337641/extracted/estimator.py#L19833-19840)
- [estimator.py#L24603](../release/337641/extracted/estimator.py#L24603)

## HCF / final-layer head columns

The optional final-layer head-input fragment adds standardized physical features to the first hidden-layer bias through 128 weights per feature. The defined columns are f15_diag (the pair-carried law diagonal residualized against v, 1, and mu^2, then divided by v), f15_k31l_diag (the young pairs' fourth-chaos plane diagonal), and f15_k4_local (local fourth cumulant divided by v^2). The designated file selects f15_diag.

Expansion: Final-layer learned-head input columns; HCF is the code prefix.

Evidence status: defined.

- [estimator.py#L12233-12248](../release/337641/extracted/estimator.py#L12233-12248)

## hcfa / _HCF_AGES

The hcfa finite-difference arms multiply _HCF_AGES, the strengths of the age-1 and age-2 cohorts used to form head column 18 (f15_k31l_diag). They change this feature's cohort weights, not the number of head units. The negative arm sets the tuple to (0.9, 0.9).

Expansion: Age strengths for the final-layer head's fourth-chaos diagonal feature.

Evidence status: defined.

- [estimator.py#L12233-12245](../release/337641/extracted/estimator.py#L12233-12245)
- [candidate_hcfa_m_sci_v1.py#L24456-24457](../experiments/dials/15-fitted-zero-bill-dials-re-centred-finite-differences/code/candidate_hcfa_m_sci_v1.py#L24456-24457)

## head-input law

The finding (2026-10-02 18:41Z) that the unit-specific diagonal of the pair law's matrix, (B_ii - fit)/(tr C v_i) on the read's response rows, fed as a new (31st) input column of the learned mean correction, is worth -1.8..-2.1% on the judging ranks (105 of its 128 zero weights beyond 3 se); 6 x_i is the unit-specific kurtosis the second-chaos closure implies. Direct physics uses of the same number lose; the head shrinks and calibrates it. Shipped as S42 (-1.55% native).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L623-630` (outside the copied Phase 2 evidence set).

## Hermite rows

Derivatives of the Gaussian ReLU moments with respect to the pre-activation mean, organised as Hermite-polynomial rows of orders 0..6 that the pair map computes once and publishes for reuse; the S2tr card's pair series is written in this 'Hermite-grouped' form.

Evidence status: defined.

- [estimator.py#L848](../release/337641/extracted/estimator.py#L848)
- [estimator.py#L888](../release/337641/extracted/estimator.py#L888)
- [estimator.py#L7497](../release/337641/extracted/estimator.py#L7497)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4131` (outside the copied Phase 2 evidence set).

## hub coefficients S1..S4

The exactly known fourth cumulant of a hub unit's post-activation with m legs on the hub and 4-m on the covariance column: S1 = -a pdf/v (= E f'''), S2 = 2 (g - g^2 - m d), S3 = 3 (1-g)(r2 - 2 m^2), S4 = kappa4(x) = r4 - 4 m r3 + 6 m^2 r2 - 3 m^4 - 3 (r2 - m^2)^2 (with v = sigma^2, d = pdf/sigma, a = alpha, g = gate, m = the host mean); kept per pair from birth and used by F4b, K4L, K31L4 and the pair-carried law.

Evidence status: defined.

- [K4L_SPEC.md#L17-22](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L17-22)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L308-310` (outside the copied Phase 2 evidence set).

## inherited vs local (born) error

Splitting each layer's mean error into the part inherited from the previous layer (gate x W^T e_{l-1}, first order) and a local remainder, propagated linearly to layer 15 (the linearised total reproduces the final MSE exactly): the final MSE is 79% inherited from the layer-14 means and 22% L15-local; by layer of origin L13-15 52.5%, L8-12 38%, L0-7 7%; about 83% of each layer's error is passed on.

Evidence status: defined.

- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4127-4129` (outside the copied Phase 2 evidence set).

## join (old-tier join / join truncation)

At layers 7, 10 and 13 the oldest young cohorts are absorbed into the old tier: the joined columns J are projected onto a fresh rank-384 frame built by a randomised range finder (Nystrom sketch Y = G[:, rows] of the Gram G = J W J^T + Qp core Qp^T, QR, factors Fj = Qn^T J, weighted core updated), re-truncating all old content each time. The 'join truncation' is the error of that rank cut, the only remaining >= 1% oracle item; the 'every3' schedule (L7/10/13) beat one join per layer by -2% raw at neutral bill.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L31` (outside the copied Phase 2 evidence set).
- [estimator.py#L5212](../release/337641/extracted/estimator.py#L5212)
- [estimator.py#L6956](../release/337641/extracted/estimator.py#L6956)

## JORD

Taking the Nystrom join rows in energy order (as the UVP header intended, so the leading r_u = 256 frame coordinates that keep the pair means carry the most energy) instead of index order: -0.24% MSE on 24 seeds at zero bill.

Expansion: join rows in energy order.

Evidence status: defined.


## JSK (join sketch) / JSK-lite / kc / _JSK_KC

The Nystrom low-rank sketch of the joined plane columns at each join. JSK-lite (2026-10-18) computes the sketch product Y = J1 W1 J1[rows]^T on only the kc heaviest joined columns (energy lam_c^2 |J_c|^2) while the factors Fj = Qn^T J and the read stay exact over all columns and the core is rescaled to the full joined trace: bill -1.4% at raw ~0. _JSK_KC maps join layer -> kc: 2048/1536/1536 (line 6958; fired as #336854), overridden to 1536/1024/1024 by the 2026-10-19 squeeze (line 24552; fired as #337065, -0.39% adjusted on 32 store ranks), the setting of the designated files.

Expansion: JSK = join sketch; kc = number of sketched columns.

Evidence status: defined.

- [estimator.py#L6958](../release/337641/extracted/estimator.py#L6958)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L52` (outside the copied Phase 2 evidence set).
- [estimator.py#L24550-24552](../release/337641/extracted/estimator.py#L24550-24552)

## K13 / K14 / K15 (scout-card screens)

K13 = K4L ages 3-4 at the final layer only on top of gk (k4lf34 / k4lf34h; passed, -0.199% on the board as #335709); K14 = the banded beta scan (gbs -0.5 at L5-9, +1 at L10-11, -0.5 at L12-14), null; K15 = the gated beta, null. Not cumulant orders.

Expansion: numbered idea-sprint arms of the scout-card lane.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1787-1791` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1864` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1928-1943` (outside the copied Phase 2 evidence set).

## K21

The mixed third-cumulant plane K21(i,j) = kappa(z_i,z_i,z_j), whose diagonal is the per-unit third cumulant. In the signed-column implementation the bank contribution is sum_r lambda_r p_ir^2 p_jr, supplemented by the star and collective-gain terms. Pair-carried fourth-order corrections can be folded into this read's left factor; the resulting effective plane is distinguished from the bare third-cumulant bank.

Expansion: the (2,1) third-order plane.

Evidence status: defined.

- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L59-60` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).
- [estimator.py#L103-120](../release/337641/extracted/estimator.py#L103-120)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L33` (outside the copied Phase 2 evidence set).
- [estimator.py#L5121-5130](../release/337641/extracted/estimator.py#L5121-5130)
- [NCAR6_SPEC.md#L42-52](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L42-52)

## K22

The plane kappa(z_i, z_i, z_j, z_j). In the card the isotropic form K22 = radial * (v v^T + 2 C o C)/3, present in the covariance step as cov += d d^T (Co^2 + k22/2)/2; measured 95-99% isotropic, correlating 0.967-0.982 with the truth.

Expansion: the (2,2) fourth-order plane.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L33` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L105` (outside the copied Phase 2 evidence set).

## k3

Each pre-activation's third central moment, read as k3 = diag(K21). The pair map separately forms t3 = k3 / sigma^3, the dimensionless standardized skewness. At the final layer only a row-cut third-cumulant push is computed.

Expansion: Per-unit third cumulant; skewness is its standardized form k3 / sigma^3.

Evidence status: defined.

- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L58` (outside the copied Phase 2 evidence set).
- [estimator.py#L110](../release/337641/extracted/estimator.py#L110)
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- Frozen-source reference: `explainers/24_what_a_closure_is.html` (outside the copied Phase 2 evidence set).
- [estimator.py#L851-854](../release/337641/extracted/estimator.py#L851-854)
- [estimator.py#L900](../release/337641/extracted/estimator.py#L900)

## K31

The plane kappa(z_i, z_i, z_i, z_j). In the card it is a per-row form, T31 = t4 * rowfactor * rho with the per-row K31 scale rowf = c0 + c1 * mu*k3/v^2 (table _K31R_TABLE, (0.85, 0.5) at port time); the pair-carried law (NCAR6) and K31L4 add network-specific content to its remainder slot. The true (3,1) plane is only 19-23% explained by those terms.

Expansion: the (3,1) fourth-order plane.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L33` (outside the copied Phase 2 evidence set).
- [K4L_SPEC.md#L9](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L9)
- [estimator.py#L3929-3935](../release/337641/extracted/estimator.py#L3929-3935)
- [NCAR_SPEC.md#L11](../experiments/fourth-order-terms/norm-coupling-carrier-16-partner-vectors-lowers-mse-2/NCAR_SPEC.md#L11)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L602-605` (outside the copied Phase 2 evidence set).

## K31L4 / NP-K3L / _NP_K3L

The fourth-chaos part of the young pairs' local quartic (the part no second-chaos carrier holds) as a (3,1) plane. K4L uses the pair's local quartic Q = e1 p^3 D + e2 p^2 D^2 + e3 p D^3 + e4 D^4 on its pushed columns p (+) and q (-), D = p - q, on the diagonal only; the plane is K31L_ij = (1/4) sum_h [dQ/dp_i p_j + dQ/dD_i D_j] minus its second-chaos projection (two of the three i-slots replaced by the pair's means of p^2, pD, D^2), which the pair law already carries; used as K21_eff = K21 - (mu_i/(3 v_i)) K31L on the read's rows and columns, folded into the same read's left factor as a cubic-plus-linear polynomial per column (switch _NP_K3L; per-layer strengths per age _NP_K3L_BY_LAYER; ages 1-2 shipped in S41, a third age by RAD, age 4 viable only as a layer band).

Expansion: K31 local, fourth chaos.

Evidence status: defined.

- [NCAR6_SPEC.md#L59-83](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L59-83)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L588-591` (outside the copied Phase 2 evidence set).
- [estimator.py#L11340](../release/337641/extracted/estimator.py#L11340)
- [estimator.py#L11338-11346](../release/337641/extracted/estimator.py#L11338-11346)
- [estimator.py#L24592](../release/337641/extracted/estimator.py#L24592)

## k4

Each pre-activation's fourth central moment minus three times its variance squared. The base card sets k4_i = radial * v_i^2, with one shared dimensionless radial coefficient per layer and band multipliers; K4L adds a unit-specific correction. The pair map uses t4 = k4 / v^2. The cumulant feeds the mean formula, fourth-order pair-map rows, and final-layer mean.

Expansion: Per-unit fourth cumulant; excess kurtosis is its standardized form k4 / v^2.

Evidence status: defined.

- [K4L_SPEC.md#L7-9](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L7-9)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L329-330` (outside the copied Phase 2 evidence set).
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- [estimator.py#L883](../release/337641/extracted/estimator.py#L883)
- [estimator.py#L901](../release/337641/extracted/estimator.py#L901)

## k4 bands (k4a / k4b / k4c)

Three per-layer-band dial multipliers on the card's isotropic per-unit k4 = radial v^2 (the k4 band dial scales the base radial k4 before K4L adds its dk); in the file the triple _S19_K4BAND, (0.9826, 0.9812, 0.9613) in the designated files' micro-stack; +-1% / +-2% 'k4band' arms were flat on the final stack.

Evidence status: defined.

- [K4L_SPEC.md#L7-8](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L7-8)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L330` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L840` (outside the copied Phase 2 evidence set).
- [estimator.py#L24590](../release/337641/extracted/estimator.py#L24590)

## K4L / _K4L_C / _K4L_AGE

Each young pair born at hub h contributes to the per-unit fourth cumulant of unit i the local quartic loc_i = 4 S1 Y^3 Wt + 6 S2 Y^2 Wt^2 + 4 S3 Y Wt^3 + S4 Wt^4 (Wt and Y the pair's transported legs, S1..S4 the hub's exact birth coefficients); the isotropic level is kept by a weighted regression and k4_i += c dk_i. Native form: the quartic is written on the pushed columns p and D = p - q as D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))) with four coefficients per pair fixed at birth, 12 flops per element of rows x pairs, float32; rows the row cuts do not push get no correction. _K4L_C is the strength per age ((0.873, 0.725) in the fragment, (0.814, 0.676) in the designated files' micro-stack), _K4L_AGE the age window (2; 4 where K4L34 is on). -2.8% pooled torch at birth, -1.4% on fresh ranks on top of F4.

Expansion: K4 local (non-isotropic per-unit fourth cumulant from the hub coefficients).

Evidence status: defined.

- [K4L_SPEC.md#L13-42](../experiments/fourth-order-terms/young-pair-hub-coefficients-give-non-isotropic-kurtosis-correction/K4L_SPEC.md#L13-42)
- [estimator.py#L3323](../release/337641/extracted/estimator.py#L3323)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L325-338` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L341-348` (outside the copied Phase 2 evidence set).
- [estimator.py#L10866-10875](../release/337641/extracted/estimator.py#L10866-10875)
- [estimator.py#L24596](../release/337641/extracted/estimator.py#L24596)

## K4L34 / k4lf34h

The K4L hub terms taken from pairs aged 3-4: k4lf34h = at the final layer only at strength 0.5 (fired in #335709, board -0.199%); K4L34 = also at the inner layers 8-15 at strength 0.5 (arm C, -0.51% on 24 networks), carried in the file as _K4L34_BY_LAYER = {8..15: (0.5, 0.5)} (a wrapper of _k4l_local that appends the two strengths and sets the age window to 4 at the listed layers); strengths 0.3/0.7/1.0 and ages 5-6 null.

Expansion: K4L from pairs aged 3-4.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1787` (outside the copied Phase 2 evidence set).
- [estimator.py#L13872-13905](../release/337641/extracted/estimator.py#L13872-13905)
- [estimator.py#L13930](../release/337641/extracted/estimator.py#L13930)

## kappa_C (recentering, KC lever)

The learned-correction covariance-recentering coefficient. For mean correction d, S = host d^T + d host^T + d d^T; the native implementation subtracts (1 + rc_l - kc_l) S using effective dial rc_l - kc_l. Different records use kappa for the net scale and kc for the add-back, so their numerical values are not interchangeable without the rc convention. Lowercase kc in JSK denotes a join-sketch column count instead.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L141-144` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L565-570` (outside the copied Phase 2 evidence set).
- [estimator.py#L12522-12527](../release/337641/extracted/estimator.py#L12522-12527)

## KC / RC (mean-recentering coefficients)

After a learned mean correction d, define S = host d^T + d host^T + d d^T. The covariance update subtracts (1 + rc_l - kc_l) S. The training implementation adds kc_l S after its other state hooks; the native implementation folds the coefficients into rc_eff = rc_l - kc_l. This RC is distinct from the read-column cut named RC elsewhere.

Expansion: KC = kappa_C add-back; RC is the existing covariance-recentering dial.

Evidence status: defined.

- [estimator.py#L12522-12527](../release/337641/extracted/estimator.py#L12522-12527)

## kink (kink unit)

A unit whose pre-activation mean is near zero relative to its spread (|a| = |mu/sigma| < 1.5 in the file's kink mask; |a| < 1 in the oracles): at the kink the post-ReLU mean is maximally sensitive to the variance and the k3 sensitivity vanishes. Kink units are the birth hubs of the bank and carry most of the late-born error (the top 5% of L14 units carry 54% of it).

Evidence status: defined.

- [estimator.py#L3966](../release/337641/extracted/estimator.py#L3966)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L30` (outside the copied Phase 2 evidence set).

## kurtosis cap defect

Finding of 2026-09-02 (phase-2 cross review): the deployed card's per-unit kurtosis was under-scaled 13% (L2) to 63% (L15) against the 32M truth; first attributed to the CP4 rank-2048 top-energy cap dropping unit-vector births, then (T-49) to the fourth-order planes K22/K31 losing about half of the (3,1)/(2,2) content born at each layer because the plane-only push does not carry the distinct fourth-order classes; frozen per-layer k3/k4 rescale tables were a zero-cost improvement at the time.

Evidence status: defined.

- [MECHANISM_TREE.md#L32](../experiments/error-attribution/exact-variance-correction-error-1-466e-8-versus-deployed/MECHANISM_TREE.md#L32)

## L3o / _l3o_last

The L3o output routine reconstructs the terminal quadrants of a recursive matrix product, combining the upper output half as one pair addition. Later _l3o_last overrides reuse owned output views and batch terminal additions. It is an output-transform helper, not a mathematical layer-3 correction.

Expansion: Reduced-request output unwind of the recursive matrix kernel.

Evidence status: defined.

- [estimator.py#L11942-11953](../release/337641/extracted/estimator.py#L11942-11953)
- [estimator.py#L20727-20759](../release/337641/extracted/estimator.py#L20727-20759)

## LCR / _LCR_RU

At layers 10-14, reconstruct the old tier's common pair component using only _LCR_RU coordinates while retaining every difference direction and all signed coefficients. The carried frame, core, and Q remain unchanged. LCRX sets the common rank to 384; the retained setting is 256. This is a local reconstruction approximation, not a reduction in the transported frame's rank.

Expansion: Late local common-factor rank restriction; the record does not state a literal acronym expansion.

Evidence status: defined.

- [estimator.py#L8873-8906](../release/337641/extracted/estimator.py#L8873-8906)

## learned correction (learned mean correction, GNN, heads)

A per-unit message-passing correction that augments the propagated physical mean and carries a 64-channel learned state. The final encoder has 78 explicit inputs per unit: 30 normalized physical features, 24 message features, and 24 energy features; it has two 128-wide hidden layers. Additional physical statistics enter through per-layer hidden-bias terms. Embedded float32 parameter arrays supply the encoder, messages, per-layer heads, and on-policy readout. Older records describe different message widths and parameter counts.

Expansion: GNN = graph neural network style per-unit message-passing correction.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L53` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L34-35` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L118-125` (outside the copied Phase 2 evidence set).
- [estimator.py#L700-754](../release/337641/extracted/estimator.py#L700-754)
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- [estimator.py#L25662-25696](../release/337641/extracted/estimator.py#L25662-25696)
- [estimator.py#L11192](../release/337641/extracted/estimator.py#L11192)
- [estimator.py#L12406](../release/337641/extracted/estimator.py#L12406)

## low-rank star (_lr_factor randomized SVD, _LR_K / _LR_Q / _LR_R / _LR_SEED, S1LR, lrnys)

The star plane A is carried as Q B with a randomized range finder: Y = A Omega (Omega a fixed n x 48 Gaussian drawn once from the seeded generator _LR_SEED = 20260924), Q = qr(Y), one power step with re-orthonormalisation (_LR_Q = 1, needed because the star's spectrum is flat: 98% of its energy needs rank 131-720), B = Q^T A and an exact truncation to rank _LR_R = 32 (a randomized SVD); the push and the s2 product then run as thin products. S1LR (2026-09-24) computes the star's dense product s1 = pw w from the rank-32 factors too (native bill ~-2.1%, raw +0.43%). The porter's 'lrnys' variant replaces A Omega by a gather of the 48 largest-norm columns (test only).

Evidence status: defined.

- [estimator.py#L6655-6680](../release/337641/extracted/estimator.py#L6655-6680)
- [estimator.py#L6884-6896](../release/337641/extracted/estimator.py#L6884-6896)
- [estimator.py#L7159-7170](../release/337641/extracted/estimator.py#L7159-7170)

## LVFS / lvfuse_sized32

Fuse every left-operand tree, and fuse right-operand/output trees only when their size is at most _LVFS_LIMIT. The original limit is 32_000_000 floats. Larger right/output trees use per-component operations. This threshold is a tree element-count limit, not a request-count limit; the source states identical operation order, predictions, and bill.

Expansion: Size-limited level-operation fusion; LVFS is the implementation prefix.

Evidence status: defined.

- [estimator.py#L12594-12614](../release/337641/extracted/estimator.py#L12594-12614)

## m2 / qb / eold / qold (learned-head physical features)

The added head features are: m2 = t3^2 He4(alpha) phi(alpha)/72, the second-order mean term divided by sigma; qb = the off-diagonal read plane contracted against c = (gate/sigma)^2 mu, then normalized by (c dot mu - c_i mu_i) v_i; eold = sum_r |lambda_r| pp_ir^2/v_i over cohorts aged at least three layers; qold = their uncentered local quartic divided by v_i^2. The record uses these as features supplied to learned heads. Lowercase m2 here is distinct from the W^2 message M2.

Expansion: Second-order mean term, plane contraction, old-cohort energy, and old-cohort local quartic.

Evidence status: defined.

- [estimator.py#L11233-11244](../release/337641/extracted/estimator.py#L11233-11244)

## mean formula

The card's per-unit post-ReLU mean: m = mg + k3*e/6 + k4*f/24, where mg = sigma*(a*Phi(a) + phi(a)) is the Gaussian term (a = mu/sigma), e = -a*phi(a)/v is the third-cumulant sensitivity and f the fourth-cumulant term ((a^2-1)*phi(a)-type); i.e. the Gaussian ReLU mean plus Gram-Charlier/Edgeworth tilts from the unit's own skew and kurtosis. Its inputs are the per-unit mean, variance, skew and kurtosis; fed true inputs it is accurate to the Monte Carlo noise floor.

Evidence status: defined.

- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L360` (outside the copied Phase 2 evidence set).

## Mehler series (Price series)

The Gaussian pair-moment expansion in powers of correlation, weighted by expectations of derivatives of the two transformed variables. In the designated file, later-layer Gaussian pair terms use correlation orders 1-3, while the separate first-layer Gaussian branch uses orders 1-6. The derivative-array width also accommodates cumulant corrections and must not be read as the Gaussian truncation order.

Evidence status: defined.

- Frozen-source reference: `explainers/46_the_pure_closure_propagator.html` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L41-43` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L999` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4131` (outside the copied Phase 2 evidence set).
- [estimator.py#L829-833](../release/337641/extracted/estimator.py#L829-833)
- [estimator.py#L16959-16978](../release/337641/extracted/estimator.py#L16959-16978)

## micro-stack (flat-dial parabola optima, #337212)

The 2026-10-19 release stack of zero-bill dial moves read at predict time, each set at the vertex of its parabola on the store ranks: _S19_K4BAND (0.9826, 0.9812, 0.9613), _NP_K3L_BY_LAYER (per-layer three K31L4 strengths), _K31R_TABLE (0.5426, 0.5027) at every layer, _K4L_C (0.814, 0.676), the _NP_STRENGTH table and _BSK 0.434; graded as #337212 (1.4296e-9, -0.225%).

Evidence status: defined.

- [estimator.py#L24588-24599](../release/337641/extracted/estimator.py#L24588-24599)

## MX1 / MX2 / MX3

These terms contain a hub's second derivative, two carried third-cumulant legs, and one Gaussian covariance edge, exemplified by d_a kappa3(z_a,z_b,z_c) g_b g_c g_d C_ad. MX1 requires theta_ri = 6 sum_a d_a pp_ra E_ai Y_ai, a contraction over hubs and bank columns. The research distinguishes this family from the pair-local norm-coupling law and local fourth-chaos term.

Expansion: Three placements of mixed fourth-cumulant terms linear in the third cumulant.

Evidence status: defined.

- [k31m.py](../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31m.py)

## NCAR

For iid Gaussian weights the fourth cumulant of a layer's pre-activations has a second-chaos part carried by one matrix, Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k) (how the covariance of a pair moves with the layer's squared norm), whose network-specific part is mostly top covariance eigenvectors x per-unit partner vectors. NCAR carried the partner vectors q_m from layer to layer with matrix-vector products and added (3/n) Ntilde as an extra K31 plane (-4% on held-out 651). Superseded 2026-10-02 12:54Z by the pair-carried law NCAR6.

Expansion: The record does not state a literal expansion of NCAR; its specification calls it a vector carrier of network-specific norm coupling.

Evidence status: defined.

- [NCAR_SPEC.md#L1-12](../experiments/fourth-order-terms/norm-coupling-carrier-16-partner-vectors-lowers-mse-2/NCAR_SPEC.md#L1-12)
- [NCAR_SPEC.md#L64-65](../experiments/fourth-order-terms/norm-coupling-carrier-16-partner-vectors-lowers-mse-2/NCAR_SPEC.md#L64-65)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L481-488` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L512-515` (outside the copied Phase 2 evidence set).

## NCAR6 (pair-carried law, pair law, NP)

The whole lean law of the norm-coupling matrix carried on the bank's own column pairs: each hub's fresh source is a symmetric 2x2 form [[S2, rv], [rv, D]] on its pair (s, e) (births, row forms, diagonal), carried as one weight w_r per bank column plus one cross weight x_h per pair and transported by the bank's own push, gates and joins. It is folded into the K21 bank read: diagB_i = sum_r (w_r pp_ir^2 + x_r pp_ir pp_partner), a regression of diagB on (tr C v_i, mu_i^2, 2 mu_i (C mu)_i) gives the multiple of C to remove (aB), the read's left factor becomes pp^2 lam + f (w pp + x pp_partner) with f_i = -strength mu_i/(n v_i), and plane_ij -= f_i aB C_ij off the diagonal. Native lever NP (a38, v9: every per-unit and per-hub formula is one polynomial evaluation from constant tables built at import; switches _NP_FORMS = which forms the fold removes, _NP_DFIT = the regression with a constant in place of the table, _NP_BETA = the carried weights from the card's beta, _NP_K3L = the K31L4 term). -14.4% on held-out 651 (torch), -5.6..-7.5% pooled; fired as S40 (#333559, third place).

Expansion: NCAR version 6; NP = the native lever name of the pair-carried law (expansion not stated).

Evidence status: defined.

- [NCAR6_SPEC.md#L1-52](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L1-52)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L546-573` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L577-583` (outside the copied Phase 2 evidence set).
- [estimator.py#L11324](../release/337641/extracted/estimator.py#L11324)
- [estimator.py#L11379-11418](../release/337641/extracted/estimator.py#L11379-11418)
- [estimator.py#L11320-11336](../release/337641/extracted/estimator.py#L11320-11336)

## network sets (TRAIN640, VALID40, TRAIN90, A41, judging ranks, line-search ranks, panel 35/100/37/39)

TRAIN640 = the 641-network training split (TRAIN160 holds the panel networks 35/100/200); VALID40 = 40 validation networks, a biased judge for learned parameters; A38/A41 = generated fresh sets (seeds 7100392+, 7100584-591 ...); TRAIN90 = the 90 on-policy capture networks; judging ranks = the 48-50 held-out ranks never in a gradient batch; line-search ranks = the 16 used to size refit steps; the panel/screen four = networks 35, 100 (training) and 37, 39 (held out) with 32M/16M truths.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L249-258` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L44-49` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L674` (outside the copied Phase 2 evidence set).
- [README.md#L26](../release/337641/README.md#L26)

## NP strength (_NP_STRENGTH)

The per-layer strength of the pair-carried law's fold in the bank read (the NP lever); the first joint dial solve set it to 1.83/1.40/1.18 by layer band (S41), the table in the file runs 1.82 at layer 1 down to ~1.01 late; x0.97 was -0.1% on the base file but did not transfer to the readout stack.

Evidence status: defined.

- [estimator.py#L11379-11418](../release/337641/extracted/estimator.py#L11379-11418)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L596-597` (outside the copied Phase 2 evidence set).

## NT join

Static Nystrom oversampled by p extra rows (p32/p64/p128) followed by an eigen-truncation back to rank 384; it captures the whole join prize (-1.30% MSE natively = the no-join ceiling -1.55%) but trades about 1:1 with bill (rates 0.64-1.14). Closed 2026-10-16.

Expansion: Nystrom with eigen-truncation (inferred).

Evidence status: inferred.


## Nystrom join

The sketch used at the joins since S23: the rows with the largest Gram diagonal are the pivots, Y = G[:, rows], then QR; the basis is the column space of a column subset of the Gram. Under the Winograd kernels' rounding (4-8e-6 relative) this one-product sketch is noise-optimal: power steps, whitening, oversampling with eigen-truncation (NT) and random or value-weighted pivots were all measured worse or 1:1 with bill.

Evidence status: defined.

- [estimator.py#L5212](../release/337641/extracted/estimator.py#L5212)

## O18 / O20 switches (_O18PAD_RULES, _O18_REQUEST_PENALTY, _O20_GRAM_ON, _O20_MU32/_O20_B32, _O20_ECUT_RADICAL)

_O18PAD_RULES pads some products up to kernel-eligible shapes (nominally more work, fewer metered multiplies); _O18_REQUEST_PENALTY = 2,000,000 is the metered FLOPs a route must save per added request to be selected among exact routes; _O20_GRAM_ON computes only the upper half of symmetric Gram products; _O20_MU32 / _O20_B32 use float32 instead of float64 for the mean and b-carrier transports; _O20_ECUT_RADICAL_* replaces the edge cut's tanh by a rational nonlinearity (a measured approximation).

Expansion: Prefixes of dated October release-code variants (inferred from their definitions and record).

Evidence status: inferred.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).
- [estimator.py#L24038-24040](../release/337641/extracted/estimator.py#L24038-24040)
- [estimator.py#L23650-23656](../release/337641/extracted/estimator.py#L23650-23656)
- [estimator.py#L24615-24618](../release/337641/extracted/estimator.py#L24615-24618)
- [estimator.py#L24629](../release/337641/extracted/estimator.py#L24629)
- [estimator.py#L25328-25330](../release/337641/extracted/estimator.py#L25328-25330)

## oracle (open loop / closed loop)

An experiment that replaces one carried quantity by its Monte Carlo truth inside the card to size the final-error share attributable to it: open loop = at one layer, measuring the next layer's fresh error; closed loop = substituted at every layer (or a range), free-running after. Examples: exact pre-mean into the terminal layer -59%; exact pre-covariance at L10-14 -26/-34%; truth planes at L10-14 -18%.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L131-140` (outside the copied Phase 2 evidence set).

## OTT (old positive tail)

An approximation of the old tier's positive-tail diagonal by the Gaussian first-chaos trace: only the local reconstruction of old rows beyond the tail threshold _OTT_Z = 2.5 is cut, the complete carried tensor is unchanged. The L14 exactness stack raises the threshold to _OTT14_Z = 3.0 at the L14 push only ('old-tail positive side 3.0': raw -0.15% for ~+0.09% bill); the exact version (OTT off, all old rows exact) is worth -0.32% for +2.8% bill.

Expansion: OTT = old-tier positive-tail diagonal (by its code comments).

Evidence status: defined.

- [estimator.py#L8564-8570](../release/337641/extracted/estimator.py#L8564-8570)
- [estimator.py#L24573-24587](../release/337641/extracted/estimator.py#L24573-24587)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4076` (outside the copied Phase 2 evidence set).

## pad18

A whitelist maps eighteen observed matrix-product shapes to padded shapes under the recorded flopscope price model. It applies to ordinary products and one shared-old384 product. The source introduces local temporary arrays and retains the parent dispatcher's cache policy.

Expansion: Fixed table of eighteen product-shape padding rules.

Evidence status: defined.

- [estimator.py#L24036-24044](../release/337641/extracted/estimator.py#L24036-24044)

## pair map

The card's bivariate step (the function nonlinear(mu, C, K21, k4, ...) in the file): from a layer's pre-activation mean, covariance and third/fourth-order inputs it produces the post-ReLU mean, the post covariance (a Mehler/Price series in the pair correlation truncated at rho^3, plus Gram-Charlier terms from K21, K31 and K22) and the radial coefficient. About 1-2% of the bill.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L33-34` (outside the copied Phase 2 evidence set).
- [estimator.py#L103-136](../release/337641/extracted/estimator.py#L103-136)
- [estimator.py#L847](../release/337641/extracted/estimator.py#L847)

## pe3 / pe4

Per-layer dial families added to the refit line at S54: multipliers (1 + pe3[l]) and (1 + pe4[l]) on the pair map's local skew and kurtosis terms e3 and e4 (via t3 / t4), which appear only in the pair map; natively _PE34_BY_LAYER.

Expansion: pair-map e3 / e4 multipliers.

Evidence status: defined.

- [estimator.py#L900-902](../release/337641/extracted/estimator.py#L900-902)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1808` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1822` (outside the copied Phase 2 evidence set).

## peel / shared / fusion budgets (level fusion)

Route-selection budgets of the kernel planner. Peel = exact depth peeling: a product is split so that an exact Smirnov L6 body (families such as peel6_p20_4_4_17) handles the main block and the fringe goes to the incumbent kernel ('fixed public-tariff strict750 mixed fringe policy'; plans _PEEL_PUSH_PLANS / _PEEL_READ_PLANS selected only by the saved public weighted arithmetic tariff, -0.73% bill on the matched census); shared = one operand's transform reused across products (the shared _SMX_ENV arena); level fusion = tree levels / chunk keys fused so fewer requests are issued ('restricted level fusion' = left-operand trees only: bill 0, bytes identical, kernel time -13%). Each route is chosen by metered FLOPs saved per added request (_O18_REQUEST_PENALTY).

Expansion: peel = exact depth peeling.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).
- [estimator.py#L19574-19582](../release/337641/extracted/estimator.py#L19574-19582)
- [estimator.py#L7442](../release/337641/extracted/estimator.py#L7442)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L225` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L288` (outside the copied Phase 2 evidence set).
- [estimator.py#L19578-19584](../release/337641/extracted/estimator.py#L19578-19584)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4582` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4586-4592` (outside the copied Phase 2 evidence set).

## plane

A matrix-valued cumulant slice of a layer indexed by unit pairs: the (2,1) plane K21, the (3,1) plane K31, the (2,2) plane K22. 'The plane' alone usually means the K21 plane the pair map reads; 'plane content' or 'plane error' is the part of the final error attributable to the planes.

Evidence status: defined.

- Frozen-source reference: `docs/PURE_CLOSURE_RESEARCH_WRITEUP.md#L59` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L33` (outside the copied Phase 2 evidence set).
- [NCAR6_SPEC.md#L44](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L44)

## plane bank (bank of columns)

The card's representation of the third-order cumulant tensor: a set of symmetric rank-one columns p_r with weights lam_r, two columns born per kink (hub) unit at each layer and pushed through W at every later layer, with the old cohorts compressed into rank-384 frames at layers 7, 10 and 13. About 80% of the bill (push ~43%, read ~35%). Not to be confused with 'bank' = a graded submission that became the best public score.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L29-31` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L431-433` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L51` (outside the copied Phase 2 evidence set).
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)

## power-step sketch

A refinement of the join sketch: one power iteration Y = G1 Y1 on the Gram of a column subsample, closing 60-80% of the join-basis gap offline. It failed natively (+28% MSE) because the step squares the spectrum and float32 plus kernel rounding destroy the trailing directions; whitened variants stayed 2.5-4x worse than Nystrom. Closed 2026-10-17.

Evidence status: defined.

- [estimator.py#L6661](../release/337641/extracted/estimator.py#L6661)
- [estimator.py#L7162](../release/337641/extracted/estimator.py#L7162)

## prebind / prebinding / _PB_WALL_S / lazy bind

Binding the fixed-shape frames and arena views of the Winograd trees in setup (Estimator.setup runs once per worker outside every predict's budget; the platform books it as setup_executions) so that predict pays no binding requests: a cold network otherwise binds ~219 alt-basis tree cache entries (~22 view requests each). Tree keys are network-specific but recur: a key census over 8 selection networks found 609 distinct keys, the 358 used by at least two cover 62-76% of a held-out network's keys; they are bound most frequent first until _PB_WALL_S (2.8-3.0 s) runs out, then each cache is re-ordered so the most frequent keys are evicted last (predictions unchanged; a cold network's bill falls ~1.7%). A lazy bind is a key setup did not bind, sliced for free from a maximal setup 'box' (fragment_lazybind) or bound at first use inside predict, costing requests and residual on cold networks.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L25-27` (outside the copied Phase 2 evidence set).
- [estimator.py#L6512-6526](../release/337641/extracted/estimator.py#L6512-6526)
- [estimator.py#L6592](../release/337641/extracted/estimator.py#L6592)

## proxy-to-hosted transfer

The ratio of the hosted public-score change to the local (proxy) change: ~0.5 for readout fits (hosted tracks the held-25 pooled number); product-size bill cuts transfer in full, lazy-bind/prebind cuts about half; raw-driven physics pieces transferred at 0 to 3/4.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L69` (outside the copied Phase 2 evidence set).

## public / private split / designation

Each run grades 100 networks; the public board shows the mean over evaluation indices 0-49 (so every worker's cold first network is public), the other 50 are hidden; the private re-evaluation scores the designated submissions on a fresh private suite (two designations: #337641 and #337619).

Evidence status: defined.

- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- [README.md#L3-8](../release/ARCHIVE_README.md#L3-8)

## push

Transporting the bank's columns (and the covariance, star and b carrier) through the next weight matrix, pp = W^T p, which is exact; the bank push is the largest bill item (43-44%: young roster pushes ~33%, old-tier reconstruct ~10%).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L30-31` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L431` (outside the copied Phase 2 evidence set).
- [estimator.py#L1601](../release/337641/extracted/estimator.py#L1601)
- [estimator.py#L231](../release/337641/extracted/estimator.py#L231)

## push share / read share

The fractions of the bill spent on pushing the bank columns (43-44%) and reading the plane out of them (34-35%); the two together, ~80%, are the third-order plane bank.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L36` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L431` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L51` (outside the copied Phase 2 evidence set).

## pushSM

A use of the grouped Smirnov kernel family for the matrix products that transport bank columns. The record treats push-Smirnov separately from the adopted young-read plans and records it as held after wall-cost testing.

Expansion: Smirnov matrix-product plans applied to plane-bank pushes.

Evidence status: defined.


## QL / quarterload

The parent loader copies four natural quadrants, then overwrites one quadrant with a sum or difference. QL copies the other three quadrants and computes that operation from the untouched input, under explicit float32 and shape checks. It changes loading, not rank or arithmetic. _ASTRA_QL_ON is false in the designated file after the later override.

Expansion: Quarter-load rewrite of the A/B input loaders.

Evidence status: defined.

- [estimator.py#L16482-16490](../release/337641/extracted/estimator.py#L16482-16490)
- [estimator.py#L20768](../release/337641/extracted/estimator.py#L20768)

## RAD (refit line S51-S63)

The refit-after-drop pipeline: after a bill cut ('drop') the learned arrays and per-layer dial families (ds, pe3, pe4, k21k) are re-fitted on the current card with exact gradients, a point (.pt) is frozen before any check, and the port loop (chain_rad_port) builds the native file; a round's CHECK compares the frozen point against the unrefitted card on fresh networks. Its first round (MS, 2026-10-11) was a null ablation of the carrier strengths; the automated line S51-S63 ran unattended from 2026-10-12 at about one round per 8-10 h (-0.2..-0.4% per round), stopped to 2 workers after two dry rounds (S60).

Expansion: RAD = refit-after-drop (the trainer's name for the pipeline, 2026-10-11).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1223-1227` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1808-1829` (outside the copied Phase 2 evidence set).
- [estimator.py#L13795](../release/337641/extracted/estimator.py#L13795)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L1123` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L3079` (outside the copied Phase 2 evidence set).

## radial

The card's single fourth-order scalar per layer: k4_i = radial * v_i^2 and K22 = radial (v v^T + 2 C o C)/3; computed as radial = 3 sum(post22) / (n (n+2)) (or divided by tr(cov)^2 + 2 sum cov^2).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L32-33` (outside the copied Phase 2 evidence set).
- [estimator.py#L130-134](../release/337641/extracted/estimator.py#L130-134)

## RC (read column cut, _RC_COLZ)

The young read is computed only on the live columns whose gate is at least Phi(-_RC_COLZ) (z = 2.0, columns rounded up to 32 as select() rounds) and zero on the rest; rows, births, pushes, transport, the diagonal and the omitted rows are unchanged and the pair law's forms go back on the read's block only. Native, 16 fresh networks: MSE +0.38% for bill -0.88% (S47e). The restore at L13/L14 (z 2.5) was dead (raw +0.32%).

Expansion: RC = read column cut.

Evidence status: defined.

- [estimator.py#L13560-13572](../release/337641/extracted/estimator.py#L13560-13572)

## read (bank read / plane read / young read)

Contracting the pushed columns into the plane the pair map needs on the layer's response rows x live columns (the K21 read: sum_r lam_r pp_ir^2 pp_jr); the 'young read' is the read of the young cohorts' columns (done by the Smirnov grouped kernels), the old tier's read is an inner product in the frame. Second-largest bill item (~35%).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L431-433` (outside the copied Phase 2 evidence set).
- [NCAR6_SPEC.md#L43-50](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L43-50)

## readout (on-policy second-step readout)

A per-layer linear correction of the means fitted by ridge regression on captures of the deployed estimator's own trajectory: the parent file (#337506 source) was run on 90 fresh training networks (TRAIN90), per-layer features captured, and a 961-column ridge fit per layer (ridge 0.03, damping 0.5, five TRAIN72/held-18 folds with seals before the held sets) exported as float32 head rows 1..15 plus one _gop2_gnn_step override. 'On-policy' = fitted on the deployed file's own trajectory rather than off-policy captures. Hosted: -0.40% (#337565, 705 columns), then -0.12% (hg) and -0.11% (fh); the learning curve saturates at ~-1.15% proxy.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).
- [README.md#L26-27](../release/337641/README.md#L26-27)

## readout feature blocks (first128, gate design 705, hg, fh, hgfh, 961 columns)

The linear design has 321 columns: state64 + intercept1 + second hidden128 + first hidden128. Adding elementwise squares of the two hidden layers gives 577 columns; first-hidden x gate gives the 705-column gate design. The hg block adds second-hidden x gate (128), and fh adds the elementwise product of the first and second hidden activations (128), giving 961 columns. These hidden vectors belong to the per-unit learned correction; first128 does not select the first 128 units of the 1024-wide network. The final implementation saves first90 = h before the second encoder layer and computes first90 * h afterward.

Expansion: first128 = first 128-wide encoder hidden layer; hg = second hidden layer x gate; fh = first hidden layer x second hidden layer.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L72` (outside the copied Phase 2 evidence set).
- [README.md#L26](../release/337641/README.md#L26)
- [estimator.py#L25698](../release/337641/extracted/estimator.py#L25698)
- [estimator.py#L25682-25690](../release/337641/extracted/estimator.py#L25682-25690)
- [lc_blocks2_fold0_v1.py#L15-17](../release/common/readout/lc_blocks2_fold0_v1.py#L15-17)

## refit / joint refit (jmB..jmI) / refit base / co-adaptation

A refit re-fits the learned parameters (arrays, heads, dials) on the current card; a joint refit solves several live arms at once (learned arrays, head columns 6/17/20, m2/eold/qb, qold, kappa_C) with sizes by least squares on 16 line-search ranks and a forward judge on 48 held-out ranks (jmC -0.99%, jmI -0.82%; S44/S45). The refit base is the physics base a refit runs on. Co-adaptation: the learned parts and dials are fitted to the card's own biases, so single truth-ward physics changes do little or hurt, a refit does not buy back a rank cut, and base-file dial wins vanish on the readout stack.

Expansion: jm<letter> = the lettered joint refits of 2026-10-10 (expansion of jm not stated).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L561-570` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L636-674` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L633-635` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L74` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L86` (outside the copied Phase 2 evidence set).

## release gates (stage A / C / D, prefire, setup5)

The release lane's qualification gates before a fire: stage A = einsum-signature audit and compile check (no new operation signatures), stage C = boundary legs (projected wall, residual, request counts), stage D = memory and handles (RSS, footprint, handle count), prefire = the import audit, setup5 = the 5-worker setup timing; run by release/lead_chain.sh and later release_min_v1.sh.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L364-366` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L469-471` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L598` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L629-630` (outside the copied Phase 2 evidence set).
- [README.md#L13-14](../release/337641/README.md#L13-14)

## request / put / request diet / byte-identical

A request is one client-server round trip of the flopscope client (each operation, each handle fetch); a put is a data upload. The request diet is the set of rewrites that issue fewer requests for the same arithmetic: 'byte-identical' (bit-identical) rewrites keep the predictions bit for bit (e.g. fnp.split returns all column blocks in one request), while 'rounding-level' rewrites change the float order and need ten networks to judge. Caps of the release lane after S40: first network <= 43,800 requests, warm <= 42,700.

Evidence status: defined.

- [estimator.py#L847](../release/337641/extracted/estimator.py#L847)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L25` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L409-410` (outside the copied Phase 2 evidence set).

## residual (residual time)

The per-network client-side time of the participant's Python not covered by metered kernels, capped at 0.4 s by the rules; it is driven by the number of flopscope requests (~16 us local, ~1.4x on the platform per request; ~45K instructions per request) and by sync points and garbage-collector pauses; the designated files run at ~0.35-0.37 s max.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L48` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L63` (outside the copied Phase 2 evidence set).

## response rows / live columns / alive mask

The rosters of a layer's read: the response rows are the units for which the plane is computed (min(g, 1-g) above the response floor; 1,024 at layers 1-2, ~600-700 at layers 12-14, 544 at L14 on rank 35), the live (alive) columns are the units whose bank columns are kept (gate above the alive floor; 768 at L14), and the alive mask gates the carried columns; the read is a product of shape response rows x live columns x bank columns.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L377-381` (outside the copied Phase 2 evidence set).
- [estimator.py#L3305](../release/337641/extracted/estimator.py#L3305)

## ridge / damping (readout fit)

Ridge is the L2 penalty of the readout regression (0.1 for the 705-column fit, 0.03 for the 961-column fit; 0.01 transfers worse natively, 0.3 worse in CV); damping (0.5) is the factor by which the fitted step is shrunk (1.0 overfits, CV -0.34%). The exact object damping scales is inferred from its usage alongside ridge.

Evidence status: inferred.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L72` (outside the copied Phase 2 evidence set).

## RLF

Sum the old-tier trailing read and the young-tier read in the existing depth-6 leaf basis, then perform one common output unwind. The code states exact real arithmetic but changed float32 arithmetic; it does not reduce rank or concatenate operands.

Expansion: Read-leaf fusion.

Evidence status: defined.

- [estimator.py#L14712-14717](../release/337641/extracted/estimator.py#L14712-14717)

## roster

The index set of units a step keeps (response rows, live columns, births, dead-cut rows), rounded up to multiples of 32 or 64 so the products hit the fixed kernel shapes. Roster sizes are the only client-side scalars the file pulls from the meter. Exact rosters were measured dead: the ~15 padding units per roster carry ~2.3% of the accuracy.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L29` (outside the copied Phase 2 evidence set).
- [estimator.py#L2646](../release/337641/extracted/estimator.py#L2646)
- [estimator.py#L3305](../release/337641/extracted/estimator.py#L3305)

## row cut (L15 row cut, _L15_ROWCUT_Z, row-cut z)

At the terminal layer the k3 push and the star diagonal run only on rows with |mu/sigma| <= z (the rest get k3 = 0), because the k3 sensitivity of the output row, -z phi(z)/(6v), vanishes beyond |z| ~ 3 (fable-physicist LG-4, build_l15_rowcut.py). z = 3.5 in the fragment (line 3993), 3.0 in the designated files (line 24554; fired as #337090, -0.30% adjusted); z 2.5 is a cliff (+4.7% MSE). The scored row is the final-layer post mean m = mg + k3 e/6 + k4 f/24.

Evidence status: defined.

- [estimator.py#L3264](../release/337641/extracted/estimator.py#L3264)
- [estimator.py#L3992-3999](../release/337641/extracted/estimator.py#L3992-3999)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L51` (outside the copied Phase 2 evidence set).
- [estimator.py#L24553-24554](../release/337641/extracted/estimator.py#L24553-24554)

## RQMC / randomized QMC

The Phase 2 sampling comparators include randomly shifted lattice blocks and digitally scrambled Sobol rules. Analytic response controls estimate E[q] + Q[f-q] using a random surrogate q with a known mean. Subtracting a deterministic closure mean alone cancels algebraically and does not reduce sampling variance. The executable's finite-grid, clipping, and float32 conventions are distinguished from ideal continuous-law unbiasedness in the source.

Expansion: Randomized quasi-Monte Carlo sampling.

Evidence status: defined.

- [DERIVATION_AND_BOUNDARY.md#L10-44](../experiments/sampling-estimators/matched-cost-random-control-variate-lattice-sampling-comparators-metered/DERIVATION_AND_BOUNDARY.md#L10-44)
- [DERIVATION_AND_BOUNDARY.md#L93-116](../experiments/sampling-estimators/matched-cost-random-control-variate-lattice-sampling-comparators-metered/DERIVATION_AND_BOUNDARY.md#L93-116)
- [estimator.py#L1-34](../records/research/2026-09-10-first-layer-quadratic-sampling-control-stops-after-48-64/estimator.py#L1-34)

## RRPI

The research parent identified as hill_climb_pair_rr_inverse_v1/candidate.py with source hash prefix a3efb3a1. October 1 comparisons use this parent and its learned parameters. The folder name identifies a pair representation and inverse variant, but does not establish a unique literal expansion of RRPI.

Expansion: Recorded candidate-family label; literal acronym expansion is not stated.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/_stale/RESULT_20261001T2140Z.md#L9` (outside the copied Phase 2 evidence set).

## RTT (response-row floor of the read)

The floor z on the rows the young read computes (response rows |mu/sigma| <= z, the rest completed by a rank-2 band); z 1.5 is a sharp optimum: 1.25 costs raw +2.06% for bill -0.31%, 1.75 / 2.0 give raw -0.8% for +1.6..+2.5% bill; the exact prefix variants (RTT2 off: READR -0.17% for +2.2%) do not pay.

Expansion: RTT not expanded in the record (the read's response-row floor with a rank-2 completion, by its uses).

Evidence status: inferred.

- [estimator.py#L17997](../release/337641/extracted/estimator.py#L17997)

## S2tr

The 2026-10-15/16 base file of the lineage (the 'S2' release with the trimmed Smirnov plans) used as the shared control (ctl_s2tr) of the October 16 hunts on 8 noise-free seeds 7103248-55 and the 32 store ranks. Inferred from its use as the control name.

Evidence status: inferred.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L3726` (outside the copied Phase 2 evidence set).

## sandwich / full sandwich / two-strip / _GSAFE5_LIMIT

A sandwich is a covariance product W^T (C W); a 'full sandwich' is done as one tree product (-315 requests a layer for +1.95% FLOPs); two-strip splits it in two strips (-6.5% of requests for +0.67% FLOPs); _GSAFE5_LIMIT is the number of the five full sandwiches kept (0 after the round-2 diet funded it: -0.47% bill).

Evidence status: defined.


## scale mode (scale mixture)

The gain mode seen as a rigid scale mixture: conditional second moments respond to the squared norm of an earlier layer with elasticity ~0.5 for means and 0.8-1.0 for variances and covariances, so Sym(C x b) has the right structure; 'the scale mode's column' is the direction the b6 read compares the plane against.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L101-111` (outside the copied Phase 2 evidence set).
- [estimator.py#L3345](../release/337641/extracted/estimator.py#L3345)

## seal

A frozen candidate file with fitted coefficients exported into it (e.g. the on-policy readout seal at strength x1.35, seal_s1.35/candidate_first128_onpolicy_blocks_release_v1.py, with EXPORTED_v1.json recording parents, coefficients and the step override); 'seals before the held sets' means the coefficients are frozen before the held-out folds are scored. Stacks are built on a seal.

Evidence status: defined.

- [README.md#L19](../release/337641/README.md#L19)
- [README.md#L27-28](../release/337641/README.md#L27-28)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).

## setup / setup5 / SETUP_TIMEOUT

setup(context) runs once per worker before any predict, outside any BudgetContext, and is limited to 5 s including the import (SETUP_TIMEOUT fails the whole submission); the file spends it prebinding fixed-shape frames, view aliasing, Smirnov plan preparation and constant tables (~14.8k requests, 4.10-4.26 s on the hosted graders, 3.3-3.5 s locally); #337614 lost one worker to SETUP_TIMEOUT. setup5 is the local release gate that times setup on 5 workers.

Expansion: setup5 = the 5-worker setup-timing gate.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L25` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L64` (outside the copied Phase 2 evidence set).
- [README.md#L14](../release/337641/README.md#L14)

## shared-384 / old384 / _shared_old384_mm

'Shared old384 state' is the old tier's shared rank-384 frame (frame Q, the weighted Gram core, the signed column roster) that all joined cohorts live in; _shared_old384_push / _shared_old384_mm are the push and the fast-kernel matrix product routines of that tier (the Winograd/Strassen tree route; each kernel product costs ~70 requests), which the retired fast-rescue governor would have made fall back to its parent route. Inferred from the function names and their call sites.

Expansion: the shared rank-384 old-tier frame; its fast matrix product.

Evidence status: inferred.

- [estimator.py#L1825-1833](../release/337641/extracted/estimator.py#L1825-1833)
- [estimator.py#L2620](../release/337641/extracted/estimator.py#L2620)
- [estimator.py#L2646](../release/337641/extracted/estimator.py#L2646)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L23` (outside the copied Phase 2 evidence set).

## SM40 / Smirnov plan / _ASTRA_SM40_PLANS / _SM40_CAP / size-8e9 drop

The Smirnov young-read dispatch ('exact rank40 read circuit, over owner-qualified incumbent scratch views'): _ASTRA_SM40_PLANS maps a product shape to a prepared grouped-read plan from view families such as L6_other (6, 10, 5, 29, 5), L5_wide20 (5, 20, 11, 54, 10) and L5_narrow20, prebound at setup (the 'table167' = 167 keys); _SM40_CAP = 4 caps the Smirnov reads per predict (a dict wrapper whose get returns None once _ASTRA_SM40_STATS['calls'] >= the cap, so the dispatcher falls back to the parent read); the size-8e9 drop removes the 9 plans with m*k*n >= 8.0e9. Together they are the wall-margin tail of #337619/#337641 (lines 25700-25710): bill +0.18%, raw unchanged, hosted wall -5 to -6%. SM40 off at import fails setup.

Expansion: SM40 = the 'exact rank40 read circuit' (its code header; Astra's Smirnov young-read dispatch).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L55` (outside the copied Phase 2 evidence set).
- [README.md#L18](../release/337641/README.md#L18)
- [estimator.py#L18500](../release/337641/extracted/estimator.py#L18500)
- [estimator.py#L18498-18504](../release/337641/extracted/estimator.py#L18498-18504)
- [estimator.py#L25700-25710](../release/337641/extracted/estimator.py#L25700-25710)

## Smirnov (grouped reads)

The grouped strided-read kernel family: reads of the third-order plane bank executed as grouped/batched leaf products over strided views ('3x6x3 L5', '3x3x6 L5' executables at depth L4/L5/L6, packets P10-P40), which save FLOPs (40-560 MF per read) but cost 2-4x the incumbent's server instructions. Variants: SM40 (the young-read plan table), push-Smirnov (held), join/old-bank Smirnov (wall-infeasible), SMX (the shared dispatch).

Expansion: Smirnov is the recorded kernel-family name; the cited implementation does not supply a further name expansion.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L55` (outside the copied Phase 2 evidence set).
- [estimator.py#L19053](../release/337641/extracted/estimator.py#L19053)
- [estimator.py#L19578](../release/337641/extracted/estimator.py#L19578)

## SMX

The shared Smirnov dispatch API banked 2026-10-16: one _SMX_ON flag (part of the governor's EXACT set), _smx_register / _smx_fits / _smx_call, a lazy binder with view families from arena offset 0, flat outer/padding buffers, and signed executables (_SMX_RECORDS); itself a wall optimisation (off: wall +11.6%, adjusted +0.5%).

Expansion: Smirnov-X, the shared Smirnov dispatch.

Evidence status: defined.

- [estimator.py#L19401-19403](../release/337641/extracted/estimator.py#L19401-19403)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L75` (outside the copied Phase 2 evidence set).

## SS beta (_SS_BETA)

The per-layer strength of the star-direction fragment (fragment_star_dir_v1; 'the trainer's beta at the star's layer l is _SS_BETA[l-1]'): the live star is one-direction (_SS_RANK = -1 at every layer, line 13531) with strength _SS_BETA, 0.9105 in the fragment and 0.85 in the designated files (opus-solo zb4 stack, line 19839); 0.80 / 0.90 closed as worse or flat.

Expansion: SS not expanded in the record (star-direction fragment).

Evidence status: inferred.

- [estimator.py#L13542-13549](../release/337641/extracted/estimator.py#L13542-13549)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L3933` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L6017` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L7571` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L74` (outside the copied Phase 2 evidence set).
- [estimator.py#L13531](../release/337641/extracted/estimator.py#L13531)
- [estimator.py#L19839](../release/337641/extracted/estimator.py#L19839)

## stack / tail

A stack is a candidate built by appending modifications (tails) to a parent seal or fired file; a tail is an appended block of Python that overrides constants or wraps functions (e.g. tail_sm40_cap4_size8e9_xsg_off.py: Smirnov-plan size drop, read cap 4, scatter-for-gather off), recorded verbatim in the build record. Python name resolution makes the last definition live, which is why the released file has shadowed definitions.

Evidence status: defined.

- [README.md#L16-18](../release/337641/README.md#L16-18)
- [README.md#L28](../release/337641/README.md#L28)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L13` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L31` (outside the copied Phase 2 evidence set).

## star

The plane-only remainder of the third-order tensor: the part of K21 represented separately from the signed rank-one column bank. The record contains dense and low-rank star pushes, including a rank-32 detached basis; later files use the direction-based star with _SS_RANK = -1 and per-layer strength _SS_BETA. In the F4 derivation, the fourth-order star means the hub-with-three-smooth-legs diagram.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L32` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L269` (outside the copied Phase 2 evidence set).
- [estimator.py#L231](../release/337641/extracted/estimator.py#L231)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L393-394` (outside the copied Phase 2 evidence set).
- [estimator.py#L13531-13549](../release/337641/extracted/estimator.py#L13531-13549)

## store (VM32 / Mac32) / held-out 25 / fresh / cross-half truth / paired

The store is a fixed set of 32 ranked networks with cached 2 x 32M-sample Monte Carlo truths used for paired scoring (VM32 on the Linux VM, Mac32 on the Mac); the held-out 25 are the store ranks never used for fitting; fresh = newly generated networks (fresh12, fresh24, seeds 7100000+); cross-half truth = scoring against the two independent truth halves so truth noise cancels (final U); paired = variant and control scored on the same networks (for wall, launched within a minute of each other). Raw effects below ~0.5% cannot be read on 2 or 7 ranks.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L67-69` (outside the copied Phase 2 evidence set).

## strength (readout strength x1.5 / x1.35)

A scalar multiplying the exported readout coefficients (shrinkage): x1.5 on #337619, x1.35 on #337641, the peak of the five-point ladder x1.05/1.2/1.35/1.5/1.65 on the 961-column seal (VM32 adjusted -0.572/-0.654 vs #337506); the optimum moves below 1.5 as columns grow.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L54` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L69` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L73` (outside the copied Phase 2 evidence set).
- [README.md#L7-8](../release/ARCHIVE_README.md#L7-8)

## sync point

A client-side materialisation of a flopscope array value (tolist(), bool(), float(), int(...)) that forces a server round trip and is charged to the residual; the only ones kept are integer roster counts; #337259 retired 32 of 41 of them.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L6128` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L29` (outside the copied Phase 2 evidence set).

## T22 / GTR

The pair map stores T22_ij = K22_ij / (v_i v_j). In the GTR optimization, the Gaussian series and the T22-weighted derivative series share already-rounded F2*G2 and F3*G3 products. Reusing those products preserves the original reductions and divisions; the T22 plane is not itself a matrix-multiplication kernel.

Expansion: T22 = standardized (2,2) cumulant plane; GTR = Gaussian/T22 common-product reuse.

Evidence status: defined.

- [estimator.py#L920-924](../release/337641/extracted/estimator.py#L920-924)
- [estimator.py#L16664-16715](../release/337641/extracted/estimator.py#L16664-16715)

## torch card / native / parity

The torch card is the PyTorch research re-implementation used for training, oracles and screens; it omits several native cost-saving approximations and was 0.7-3.4% more accurate than the native file on 2026-10-01 and 16-19% by 2026-10-20 (decide natively). Native = the flopscope file as scored. Parity = the check that a native port reproduces the torch arm network by network (correlation and size D of the per-unit change).

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L114-116` (outside the copied Phase 2 evidence set).
- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)

## TPS / terminal scale batching

Batch the terminal output-only scale nodes of Smirnov plans. Intermediate add/subtract nodes retain their operation, order, and destination. The preparation records the output coefficients and applies them as one terminal multiplication.

Expansion: Recorded prefix for terminal Smirnov scale batching; literal acronym expansion is not stated.

Evidence status: defined.

- [estimator.py#L20907-20930](../release/337641/extracted/estimator.py#L20907-20930)
- [estimator.py#L20972](../release/337641/extracted/estimator.py#L20972)

## truth (MC truth, organizers' truth, truth688)

Monte Carlo reference values: the organizers' 1e9-sample per-network final means (the scored truth; noise floor 0.07e-9 at layer 15), truth688 = the organizers' all-layer 1e9-sample means for 688 public networks, and the team's own 2 x 32M-sample per-unit truths of pre- and post-activation mean, variance, k3, k4 and covariance (two independent halves) on the store ranks.

Evidence status: defined.

- [47_where_the_last_error_lives.html](../estimator/research_notes/where_the_last_error_lives.html)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/TEAM_LOG.md#L253` (outside the copied Phase 2 evidence set).

## TT3

The Phase 2 migration documents identify a centered TT3 residual alongside the CP3 carrier and explicitly require rank, sketches, rounding thresholds, and cap schedules to be reselected for width 1024. Tensor-train is the inferred reading of TT in this CP/TT tensor-transport context; the cited migration text does not spell the acronym out.

Expansion: Third-order tensor-train residual representation (inferred expansion).

Evidence status: inferred.

- [README.md#L21-35](../records/research/2026-08-23-width-1024-scaling-screen-fixed-rank-carriers-fit-dense/source_README.md#L21-35)
- [MIGRATION_MATRIX.md#L5-8](../records/research/2026-08-23-width-1024-scaling-screen-fixed-rank-carriers-fit-dense/MIGRATION_MATRIX.md#L5-8)

## twin (safe twin, clean twin, science twin)

A second file that differs from a candidate in one setting: the safe twin keeps a safety setting (governor/rescue, depth-7 off) for the designated run; the clean twin has scatter-for-gather off (#337617 = clean twin of #337614); the science twin is the local-evaluation copy of a released file (same estimator under a science-mode wrapper used for paired local scoring). Matt's rule 10-18: no safe-twin fires on the lead's own judgment.

Evidence status: defined.

- [README.md#L17](../release/337641/README.md#L17)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L65` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L67` (outside the copied Phase 2 evidence set).

## u64 law (client call-form measurement)

The referenced call-form benchmark found roughly 42-45K client instructions per request across call forms and concluded that reducing the request count was the useful residual-time lever. The label identifies the measurement; it does not denote a uint64 numerical law.

Expansion: Recorded client-residual measurement, named for its experiment lane.

Evidence status: defined.


## ubown / UBcluster / cluster

The ubown route builds a B transform directly in retained independent storage, removing an arena-to-cache copy while charging the new owned bindings. The cluster extension applies related reuse to UBC, UBX, and layer-7 cache routes under their existing caps and owner/prediction lifetimes. These are storage and scheduling changes, not cumulant approximations.

Expansion: Owned B-transform storage and a related exact-copy reuse family; UB is not expanded in the record.

Evidence status: defined.

- [estimator.py#L15498-15511](../release/337641/extracted/estimator.py#L15498-15511)
- [estimator.py#L15598-15608](../release/337641/extracted/estimator.py#L15598-15608)

## UBX / UBC-X / UBX_RC

UBC-X caches counted copies of the prefix frame's right-operand B transforms for old-tier inner-read products, reusing unchanged transformed values across layers 10-14. UBX_RC adds a prediction-local reconstruction cache for the layer-7 frame and clears it when that frame is no longer used. Owner, generation, shape, and prediction identities delimit reuse.

Expansion: Recorded labels for reuse of old-tier right-operand transform trees; literal acronym expansions are not stated.

Evidence status: defined.

- [estimator.py#L14027-14039](../release/337641/extracted/estimator.py#L14027-14039)
- [estimator.py#L14253-14308](../release/337641/extracted/estimator.py#L14253-14308)

## UVP split (u-leg / v-leg)

In the old-tier frame a birth pair p+- = U (f_u +- f_v/2) is stored as a common part f_u (the u-leg, kept on the frame's leading r_u = 256 coordinates) and a difference part f_v (the v-leg, on all 384 coordinates); the split read and reconstruct in u/v form (_uvp_split, _uvp_recon / _astra_bcc_uvrecon) is the native form, worth -1.3..-1.6% bill.

Expansion: u / v pair coordinates of the old-tier frame.

Evidence status: defined.


## W3

The learned correction has W, W^2, and W^3 message products. The later W3R fragment compresses the W^3 message with a fixed response-weighted rank-4 basis while leaving W and the GOP2 W^2 contraction unchanged. Here W^3 denotes elementwise powers of the network weights, not a three-matrix composition.

Expansion: Elementwise-cubed weight message.

Evidence status: defined.

- [estimator.py#L142-163](../release/337641/extracted/estimator.py#L142-163)
- [estimator.py#L17720-17728](../release/337641/extracted/estimator.py#L17720-17728)
- [estimator.py#L19791-19828](../release/337641/extracted/estimator.py#L19791-19828)

## wall (wall clock)

The per-network wall clock of predict on the hosted grader, capped at 120 s (a TIME_EXHAUSTED network is killed at ~120 s roundtrip and the run is wiped from the board); hosted wall tracks requests (0.71) and puts (0.74), not FLOPs (0.12); the final lineage ran 99-113 s, and #337534 failed on a +10-12% grader slowdown.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L48` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L55` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L62` (outside the copied Phase 2 evidence set).

## warm / cold

Cold = a worker's first prediction (caches, keys and lazy binds not yet made: higher requests and residual); warm = later predictions. The public evaluation is ~95% warm (5 workers x ~20 networks), whereas fresh-process local science bills are cold and overstate compute gains (setup-time/first-use savings are worth ~5% of face value publicly).

Evidence status: defined.


## Winograd / Strassen kernels (packed kernel, cp3, hz, strips)

Fast matrix-product kernels that perform fewer metered multiplies: recursive Strassen/Winograd-type products over a fixed arena ('audited four/five-level fixed-arena Winograd product', depths 4-7, _CP3_WINOGRAD_LEVELS = 4) with packed push/read view sets (_PACKED_PUSH / _PACKED_READ) bound at setup; the tree leaves are batched matmuls (_hz_leaf and _ipt_leaf_unwind_ab = 67% of the bill), the Winograd adds/subtracts ~8-15%; 'hzs' = half-zero-strip padding kernels (bill -1.2%); kernel rounding is 10-25x BLAS (Frobenius 4-8e-6). Depth trades bill against wall.

Expansion: cp3 not expanded in the record; hz / hzs = half-zero strip kernels.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L33` (outside the copied Phase 2 evidence set).
- [estimator.py#L10-17](../release/337641/extracted/estimator.py#L10-17)
- [estimator.py#L264](../release/337641/extracted/estimator.py#L264)
- [estimator.py#L453](../release/337641/extracted/estimator.py#L453)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L37` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/SESSION_NOTES.md#L221-222` (outside the copied Phase 2 evidence set).

## XSG (scatter-for-gather)

Gathering rows by a scatter primitive (_xsg_rows / _xsg_cols at nine sites, gated by _XSG_ON) because the meter charged it less than a plain index gather A[idx]; predictions byte-identical, bill 0.077% lower. Judged the one fair-accounting exposure (rules s5.5: a transformation whose only purpose is a cheaper-charged path) and retired (_XSG_ON = set()) in every designated file.

Expansion: XSG = scatter-for-gather.

Evidence status: defined.

- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L35` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/WRITEUP_DISCLOSURE_DRAFT_v1.md#L76` (outside the copied Phase 2 evidence set).
- [estimator.py#L15452-15487](../release/337641/extracted/estimator.py#L15452-15487)

## YNT (young negative trace)

The young push at layers 7-14 crops the rows with gate a < -2.5 (dead units) and repairs their contribution by a trace; at layer 14 the pushed young columns feed the final layer's k3 directly (hyper-sensitive), so the crop threshold becomes _YNT_Z = 3.0 for the L13 and L14 pushes only (a per-layer wrapper of _shared_old384_push; the literal 2.5 at the three active sites, including the U/young fusion prepare, reads _YNT_Z). Part of the L14 exactness stack (#337126, #337142): raw -0.23% for +0.11% bill at L14, the L13 rung doubles it, L12 a wash.

Expansion: YNT = young negative trace.

Evidence status: defined.

- [estimator.py#L24556-24572](../release/337641/extracted/estimator.py#L24556-24572)

## zb4 (zero-bill dial stack)

The four zero-bill dial arms with negative store means stacked and appended last by the opus-solo lane: _GK4 1.0, _BFIX_GAMMA 0.119, _GSK 0.9, _SS_BETA 0.85 (store 32 ranks -0.30%, 21/32 lower); these are the dial values the designated files carry (except GK4, later 1.04).

Evidence status: defined.

- [estimator.py#L19833-19840](../release/337641/extracted/estimator.py#L19833-19840)
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4557` (outside the copied Phase 2 evidence set).
- Frozen-source reference: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/TEAM_LOG.md#L4568` (outside the copied Phase 2 evidence set).

## ZMS

The experiment identifies units whose effective learned correction differs from the raw correction and whose resulting mean is exactly zero. It zeros the corresponding covariance rows, columns, and variance entries at layers 1-14. The source calls this a covariance-only hypothesis; it leaves that layer's means and higher-moment targets unchanged.

Expansion: Zero-mean support covariance projection.

Evidence status: defined.

- [fragment_zero_mean_support_HELD_v2.py#L1-22](../experiments/moment-propagation/zero-mean-support-correction-two-network-adjusted-0-32/code/fragment_zero_mean_support_HELD_v2.py#L1-22)
