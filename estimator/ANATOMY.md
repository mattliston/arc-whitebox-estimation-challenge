# Estimator anatomy

This document describes the two source files designated for Phase 2 private re-evaluation: submissions #337619 and #337641. It is based on static source inspection and the archived release records.

The designated files estimate the mean activations of a 16-layer, width-1024 ReLU network under the input distribution. Their core is deterministic moment propagation, augmented with low-rank cumulant carriers and a learned correction to the propagated means. They use the competition's metered array client for numerical work.

## Source convention

Runtime source links point to `submissions/337641/estimator.py` unless another submission is named. The inspected frozen origins are `phase-two-release/337641/extracted/estimator.py` and `phase-two-release/337619/extracted/estimator.py`; the release copies must preserve their bytes and line numbers. Related records are under `release/`, copied from the frozen `phase-two-release/` tree. Layer indices are zero-based.

The files were assembled through appended overrides. Their first definitions of a function or constant are often superseded, and some wrappers retain explicitly captured parent functions. The last assignments and the active call chain determine the submitted algorithm.

## Target and execution path

With the source's weight orientation, the network is

$$
h_{-1}=x,\qquad x\sim\mathcal N(0,I_{1024}),\qquad
h_l=\max(W_l^T h_{l-1},0),\quad l=0,\ldots,15.
$$

The estimator returns a nonnegative `float32` array of shape `(16, 1024)` containing the estimated means of every layer. The submitted class accepts only width 1024, depth 16 and a prediction budget of at least `2**41`; it returns zeros otherwise. The active algorithm propagates distribution statistics and draws no input samples. The arrays, decompositions and learned models are evaluated through `flopscope.numpy`. ([Interface](../submissions/337641/estimator.py#L1078), [initial state](../submissions/337641/estimator.py#L24632), [output](../submissions/337641/estimator.py#L24938).)

The original class method calls `_predict_learned_compact`, which supplies a mean-correction hook, a birth-coefficient hook, a terminal scalar hook and a covariance-correction hook. The intervening wrappers manage arithmetic routes, reusable storage and prediction-local state. The active numerical loop is the last `_astra_fhm_core`, explicitly installed in `_ASTRA_FHM_ACTIVE_CORE` at line 24949. The final assignment to `Estimator.predict` is `_lb3_predict` at line 24547; that wrapper delegates to its captured parent. Reading that last wrapper alone does not reveal the estimator. ([Hook construction](../submissions/337641/estimator.py#L757), [core dispatch](../submissions/337641/estimator.py#L17661), [active core](../submissions/337641/estimator.py#L24632).)

## Carried state and linear transport

The implementation maintains the following state between nonlinearities. These objects approximate a distribution; they do not specify its full joint law.

| Object | Mathematical role | Source names |
|---|---|---|
| Mean vector and covariance matrix | First and second central moments | `m`, `C` |
| Signed rank-one third-order factors | A sum of cubic tensors, read through the slice $K_{21,ij}=\kappa(z_i,z_i,z_j)$ | `p`, `lam`, `newborn` |
| Compressed older factors | A shared basis and factor records for previously generated columns | `shared_old`, `Q`, `F`, `core` |
| Third-order residual matrix | The remaining repeated-index third-moment information after factor and trace contributions | `star` |
| Trace-mode vector | Third-order terms coupled to covariance | `b` |
| Fourth-order scale and local coefficients | A covariance-shaped fourth cumulant plus recent pair-specific corrections | `radial`, `_K4L_B`, `_F4C_COEF`, `_NP_STATE` |
| Learned recurrent state | Information passed through the network's weights to the next correction | `state`, with 64 columns per neuron |

The initial state is `m = 0`, `C = I`, with zero higher-cumulant carriers and zero learned state. Each layer forms $\mu=W^Tm$, approximately transports covariance as $C_{\rm pre}=W^TCW$, transports the third-order factors, and constructs the cumulant slices needed by the ReLU closure. The active implementation computes mean and `b` transport in `float32` and widens their outputs to `float64`. Large covariance and cumulant planes also use `float32`; selected scalar statistics and diagonals are maintained in `float64`. Thus these equations describe the intended real arithmetic, not bitwise identities between alternative implementations. ([Initialization and transport](../submissions/337641/estimator.py#L24657), [cumulant assembly](../submissions/337641/estimator.py#L24755), [precision switches](../submissions/337641/estimator.py#L24626).)

For a factor matrix $P$ and signed weights $\lambda$, the cubic contribution is

$$
\kappa_{ijk}^{\rm factor}=\sum_r\lambda_rP_{ir}P_{jr}P_{kr},\qquad
K_{21,ij}^{\rm factor}=\sum_r\lambda_rP_{ir}^{2}P_{jr}.
$$

Transport multiplies factor columns by $W^T$; the unmodified cubic read contracts their squares against weighted columns. The physical slice is $K_{21,ij}=\kappa(z_i,z_i,z_j)$. The runtime array of this name is an effective read: the implementation also folds selected fourth-order responses into its off-diagonal factor contribution. Thus the displayed cubic, residual and trace terms describe the third-order part of the approximation, not the entire effective read. The active loop adds the transported residual and the trace-mode contribution

$$
\widehat K^{(3)}_{21,ij}=K_{21,ij}^{\rm factor}+K_{21,ij}^{\rm residual}
          +v_i b_j+2C_{{\rm pre},ij}b_i,
$$

and separately restores the more accurately accumulated diagonal. The diagonal is the per-neuron third cumulant $k_{3,i}$. ([Factor read](../submissions/337641/estimator.py#L239), [active assembly](../submissions/337641/estimator.py#L24762).)

## ReLU moment closure

For variances above the implementation’s `1e-10` scale floor, write $v_i=C_{{\rm pre},ii}$, $\sigma_i=\sqrt{v_i}$, $\alpha_i=\mu_i/\sigma_i$, and let $\phi,\Phi$ be the standard normal density and distribution function. The Gaussian mean is

$$
m_i^{G}=\sigma_i\phi(\alpha_i)+\mu_i\Phi(\alpha_i).
$$

The scalar closure adds first-order third- and fourth-cumulant terms:

$$
\hat m_i=m_i^{G}
-\frac{k_{3,i}}{6\sigma_i^2}\alpha_i\phi(\alpha_i)
+\frac{k_{4,i}}{24\sigma_i^3}(\alpha_i^2-1)\phi(\alpha_i).
$$

The code similarly corrects raw moments of orders two through four, then centers them to obtain variance and the next third and fourth cumulants. This is a truncated cumulant expansion, not an exact formula for an arbitrary non-Gaussian input. The first-layer mean has no incoming non-Gaussian correction because the input distribution is Gaussian. ([Scalar map](../submissions/337641/estimator.py#L988), [pair-map marginal moments](../submissions/337641/estimator.py#L844).)

The pair map computes covariance, the repeated-index third-cumulant slice and a fourth-order slice together. Its Gaussian part is a Hermite expansion in the correlation $\rho_{ij}=C_{{\rm pre},ij}/(\sigma_i\sigma_j)$. The general pair map retains Gaussian powers through order three, plus its mixed and marginal cumulant corrections. The first-layer specialization retains Gaussian powers through order six. Allocated buffers and older function names still mention order six, so their names are not evidence that the active general map uses six terms. Diagonal entries come from the scalar moments rather than the off-diagonal series. ([General series views](../submissions/337641/estimator.py#L816), [series and centering](../submissions/337641/estimator.py#L902), [first-layer specialization](../submissions/337641/estimator.py#L16939).)

The fourth-cumulant scale is projected onto a covariance-shaped form. In the covariance metric, its update is

$$
r_{\mathrm{new}}=\frac{3\sum_{ij}K_{22,ij}^{\mathrm{post}}}
{(\mathrm{tr}\,C^{\mathrm{post}})^2+2\sum_{ij}(C^{\mathrm{post}}_{ij})^2}.
$$

At a later preactivation the corresponding approximation is
$K_{22,ij}=r(v_iv_j+2C_{{\rm pre},ij}^2)/3$, with diagonal $rv_i^2$. Empirical layer-band multipliers and local fourth-cumulant terms modify the per-unit fourth cumulant used by the scalar map. Additional calibrated covariance terms act at layers 2–14. ([Projection](../submissions/337641/estimator.py#L972), [fourth-order assembly](../submissions/337641/estimator.py#L24772), [covariance corrections](../submissions/337641/estimator.py#L7624).)

## Third- and fourth-order carriers

After each ReLU, the code separates inherited third-order content from newly generated content. It constructs two signed newborn columns per selected unit, with weights of opposite sign and shifts chosen from the local third-moment innovation. A learned update with three retained analytical directions modifies the newborn base. The original four-direction fit was reduced by removing its gate-derivative leg; the stored four-column coefficient head remains, but the active update uses only its first three outputs. ([Removal experiment](../experiments/birth-terms/fourth-birth-direction-leg-removal-saves-0-091-work/README.md).) The remaining third-moment target is retained in `star`, with its diagonal explicitly repaired. These updates are performed after the mean correction and its covariance compensation, so changes to the learned correction affect subsequent deterministic propagation. ([Birth construction and residual](../submissions/337641/estimator.py#L24859), [birth-direction update](../submissions/337641/estimator.py#L22797), [current-state birth coefficients](../submissions/337641/estimator.py#L2375).)

The third-order transport gate differs from the Gaussian gate $\Phi(\alpha)$. Its active polynomial includes skewness, kurtosis and a second-order skewness term. The birth weight starts from $d(1+r/3)$, where $d=\phi(\alpha)/\sigma$, and is multiplied by a skewness correction to the density at the ReLU threshold. The trace-mode vector also receives a correction proportional to $(r_{\rm new}-r)m$. ([Transport gate](../submissions/337641/estimator.py#L24003), [birth-density correction](../submissions/337641/estimator.py#L12116), [trace update](../submissions/337641/estimator.py#L24845).)

Older columns are compressed into a shared rank-384 basis at layers 7, 10 and 13. The join schedule adds birth cohorts 0–2, 3–5 and 6–8 respectively. The basis is obtained from a Nyström sketch of a weighted Gram matrix, using squared signed-cube coefficients as nonnegative weights. The sketch selects 1536, 1024 and 1024 high-weight columns at those joins; factor projection still uses the complete joined column block. Where needed, a small Gram core is solved, symmetrized and rescaled to the joined trace. This is an approximation of the inherited tensor. Recent cohorts retain their explicit signed columns. ([Join schedule](../submissions/337641/estimator.py#L5077), [rank rule](../submissions/337641/estimator.py#L7480), [active join](../submissions/337641/estimator.py#L24278), [final sketch sizes](../submissions/337641/estimator.py#L24552).)

Two further families retain information that a single fourth-order scale would lose. The local fourth-cumulant correction (`K4L` in the source) evaluates a quartic polynomial of each recent signed column pair, subtracts its fitted covariance-shaped component, and feeds the remainder into the scalar fourth cumulant. Its main strengths depend on pair age; additional ages three and four are enabled at layers 8–15. The fourth-order carrier shift (`F4c`) changes recent third-order columns and supplies matching trace and diagonal corrections to the residual. The norm-coupling pair terms (`_np_*`) alter the squared-column read and carry additional pair coefficients between layers. These families are distinct and should not be summarized as one fourth-order tensor. ([Local fourth cumulant](../submissions/337641/estimator.py#L10967), [older local terms](../submissions/337641/estimator.py#L13886), [column shift](../submissions/337641/estimator.py#L10771), [pair correction](../submissions/337641/estimator.py#L11619), [pair update](../submissions/337641/estimator.py#L15048).)

The final layer uses only the transported variance diagonal and third/fourth scalar cumulants, followed by its learned scalar correction. It avoids building a full final covariance. Its third-order factor and residual reads are restricted to rows near the ReLU threshold, $|\alpha|\leq3.0$, with the trace-mode term handled separately. Row selection rounds the retained count for the kernels, so the threshold is not an exact Boolean cut at every retained row. ([Terminal branch](../submissions/337641/estimator.py#L24684), [row-cut calculation](../submissions/337641/estimator.py#L3996), [selector](../submissions/337641/estimator.py#L440), [final threshold](../submissions/337641/estimator.py#L24554).)

### Physical cumulants and the effective read

For preactivations $z$ with mean $\mu$, covariance $C$, and centered values $\widetilde z=z-\mu$, the norm-coupling target is

$$
N_{ij}=\sum_k\kappa(z_i,z_j,z_k,z_k)
=\mathrm{Cov}(\widetilde z_i\widetilde z_j,\|\widetilde z\|^2)-2(C^2)_{ij}.
$$

Here $C^2$ is a matrix product. The identity is exact; the random-weight approximation relating a component of $K_{31,ij}=\kappa(z_i,z_i,z_i,z_j)$ to $3N_{ij}/n$, and the retained recurrence for that component, are modeling choices. The implementation stores a structured approximation on the existing factor pairs, subtracts fitted covariance/mean components, and folds a retained fourth-order response into the read. The derivation record separates the statistical target, proposed law and historical measurements.

The local quartic has two uses: its diagonal corrects scalar $k_4$, and its mixed $(3,1)$ slice contributes to the effective read. The [local-slice derivation](../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/NCAR6_SPEC.md#L59) removes second-chaos contractions already represented by norm coupling and retains the fourth-chaos remainder $L^{(4)}$. Its first-order reference fold is $\Delta\widetilde K_{21,ij}=-\mu_iL^{(4)}_{ij}/(3v_i)$. This is an approximate response substitution, not an identity between physical cumulants. The release adds strengths and masks and separately restores the cubic diagonal. The active calculation retains pair cross sums when forming the local subtraction. ([Local polynomial and fold](../submissions/337641/estimator.py#L14502), [form removal](../submissions/337641/estimator.py#L11722), [diagonal restoration](../submissions/337641/estimator.py#L24749).)

Scheduled factor compression occurs inside the push, before the read. The join implementation projects stored factors in the preceding postactivation coordinates, then transports the shared basis. The end-of-layer commit gates that basis and records the new cohort; it does not perform the scheduled basis construction. ([Projection and transport](../submissions/337641/estimator.py#L24278), [push before read](../submissions/337641/estimator.py#L24731), [end-of-layer commit](../submissions/337641/estimator.py#L2663).)

## Learned mean and covariance corrections

The mean-correction model combines 30 per-neuron closure features with weight-transported information from the preceding layer's 64-dimensional state. The base features include normalized moments, gate probability, covariance and third-cumulant summaries, layer coordinates and weight statistics. The final payload uses eight channels for each of three message families based on $W$, $W^2$ and $W^3$, and eight channels for each of three corresponding energy families. This gives 78 explicit encoder inputs: 30 base features, 24 messages and 24 energies. Further current-layer cumulant statistics enter through a learned adjustment to the first encoder bias. ([Features](../submissions/337641/estimator.py#L201), [message layout](../submissions/337641/estimator.py#L686), [extra bias inputs](../submissions/337641/estimator.py#L11192), [final payload shapes](../submissions/337641/estimator.py#L25696).)

The encoder has two 128-wide SiLU layers and a 64-wide linear state output. Messages are not all full dense products: the active $W^{\circ2}$ message and all three energy routes use a rank-two approximation with an exact constant component; the $W^{\circ3}$ message uses a rank-four approximation in an encoder-derived metric. The weight powers are elementwise; the energy contractions use $W^{\circ2}$, $W^{\circ4}$ and $W^{\circ6}$. The mean readout is multiplied by the preactivation standard deviation, a learned per-layer RMS scale and the bounded gain $1+0.25\tanh(\text{gain}_l)$. Layer 0 receives no learned mean correction. At layer 15, unavailable pair features are masked and terminal scalar features are supplied. ([Rank-two contraction](../submissions/337641/estimator.py#L17702), [rank-four message](../submissions/337641/estimator.py#L19807), [active message routing](../submissions/337641/estimator.py#L25452), [final step](../submissions/337641/estimator.py#L25662), [gain](../submissions/337641/estimator.py#L23851).)

At layers 1–14 the applied correction is $\delta_{\rm eff}=\max(\delta,-m)$, ensuring a nonnegative updated mean. The covariance receives the corresponding centering correction

$$
C\leftarrow C-m\delta_{\rm eff}^T-\delta_{\rm eff}m^T
                   -\delta_{\rm eff}\delta_{\rm eff}^T,
$$

followed by the calibrated layer-dependent compensation in `_GN_RC` and `_GN_KC`. The same effective correction feeds both operations. The corrected mean is then propagated into the next layer. A separate learned pair model adds a symmetric covariance correction at layers 8–14; the final file uses eight hidden channels and a rational approximation to its former `tanh` nonlinearity. ([Nonnegative correction](../submissions/337641/estimator.py#L17322), [covariance compensation](../submissions/337641/estimator.py#L24809), [pair-model implementation](../submissions/337641/estimator.py#L25460), [rational nonlinearity](../submissions/337641/estimator.py#L25328), [final pair width](../submissions/337641/estimator.py#L25519).)

### On-policy linear readout

The final revision of the readouts at layers 1–15 was fitted from features captured along the parent estimator's own corrected trajectory on 90 training networks. Updating intermediate readouts changes later inputs, so the final estimator can follow different trajectories; this was a one-step fit on frozen parent captures. The archived coefficient record specifies 15 fitted layers, 961 columns, ridge `0.03` and damping `0.5`. The 961 columns comprise the following blocks, where $s$ is the 64-wide state, $f$ the first 128-wide hidden activation, $h$ the second 128-wide hidden activation and $g$ the gate probability:

| Block | Width | Exported tensor |
|---|---:|---|
| $s$ | 64 | `head` |
| Constant | 1 | `head_b` |
| $h$ | 128 | `hidden_head` |
| $f$ | 128 | `first_hidden_head` |
| $h^2$ | 128 | `sq_hidden_head` |
| $f^2$ | 128 | `sq_first_head` |
| $fg$ | 128 | `gate_first_head` |
| $hg$ | 128 | `gate_hidden_head` |
| $fh$ | 128 | `first_hidden_prod_head` |

Products and squares in this table are elementwise. The internal term `first128` refers to the first hidden activation of this encoder, not a subset of the 1024 network neurons. The fit contains constraints and residualization against existing readout directions, so it should not be reconstructed as an unconstrained ridge regression over the concatenated matrix. The fitting sources define that procedure. ([Fit record](../release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json#L1068), [fitting loop](../release/common/readout/fit_blocks_full90_v1.py#L17), [constrained block solve](../release/common/readout/lc_blocks_fold0_v1.py#L32), [feature products](../release/common/readout/lc_blocks2_fold0_v1.py#L15).)

Export multiplies each fitted layer's coefficient update by the variant's strength. It adds the first 321 coefficients to existing readout tensors and creates five new 128-column readout tensors for the remaining 640 coefficients. Row 0 is preserved for existing tensors and is zero for new tensors. Strength is baked into the `float32` coefficients; it is not a scalar multiplier on the whole estimator, the whole mean correction or the recurrent encoder. The encoder weights and existing non-readout tensors remain unchanged by this export. ([Exporter](../release/common/readout/export_blocks_v1.py#L89), [runtime readout](../submissions/337641/estimator.py#L25682).)

## Training records and provenance limits

The embedded arrays are the inference specification. The following records document individual stages of their development; this release has not reconstructed a complete, byte-verified ledger from every fit to every final parameter. Historical training and validation sets include reused development networks. They do not form one untouched train/test partition for #337641.

**Recurrent mean encoder.** A documented September refit trained 45,200 mean-model parameters, including the recurrent encoder, readout head and gain, through the whole estimator while freezing the other modules. Its objective was final-layer mean MSE plus 0.05 times the mean MSE over earlier layers, using existing billion-sample Monte Carlo mean references. The protocol specifies 160 training and 40 validation networks and evaluation of the terminal fit rather than selection of the best validation checkpoint. This is evidence for that refit, not the full history of the final encoder. ([Trainable arrays](../records/research/2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/PROTOCOL.md#L30), [objective and split](../records/research/2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/PROTOCOL.md#L81), [completed record](../experiments/learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/).)

**Birth-direction coefficients.** The original global fit trained 60 layer/direction coefficients against downstream means on 160 training and 40 validation networks, with the response heads frozen. Its loss combined final-layer MSE with 0.05 times mean earlier-layer MSE; the first layer's candidate directions vanish at Gaussian initialization. A later conditional pilot fitted a shared 64-by-4 map from learned state to bounded direction coefficients on 20 training and four validation networks, with the encoder and global coefficients frozen. That pilot missed its threshold; a later frozen composition was separately tested. Later joint fits included both coefficient families, and the final release retains three of the four original direction actions. ([Global fit](../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/PROTOCOL.md#L85), [conditional action and fit](../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/PROTOCOL.md#L48), [composition](../experiments/learned-correction/frozen-birth-direction-mean-head-neural-composition-passes-native40/).)

**Covariance pair model.** The initial oracle-state pilot used coherent eight-million-sample covariance targets on six training networks and two held out. Its objective combined normalized covariance error and a fixed next-Gaussian-mean-response error. That pilot failed its prescribed proxy gate; later frozen deployment was a separate experiment. The final width-eight reduction instead distilled a frozen teacher's covariance corrections at layers 8–14: training ranks were 35, 150, 200 and 250; validation ranks were 259, 277 and 287. It minimized relative physical covariance error with an anchor penalty and selected the checkpoint by training loss only. Validation was not used for selection, and this compression fit used no network-truth targets. These records do not establish every earlier change to the teacher. ([Original design](../experiments/wall-and-residual/nonlinear-joint-plane-pair-pilot-information-gate-failed/JOINT_EDGE_COVARIANCE_DESIGN_V1.md#L11), [separate deployment](../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/), [compression target and split](../experiments/learned-correction/learned-edge-hidden-width-eight-150665691-mac-2338808-fresh/code/fit_edge_hidden_v1.py#L11), [selection rule](../experiments/learned-correction/learned-edge-hidden-width-eight-150665691-mac-2338808-fresh/code/fit_edge_hidden_v1.py#L171), [installed final width](../submissions/337641/estimator.py#L25511).)

**Later joint fitting.** The archived joint-fit loader includes the mean model, both birth-coefficient families, covariance pair model and its message/context arrays. It names a 641-network source pool and two 20-network validation groups, one for selection and one for reporting. The continuation scripts use smaller filtered gradient batches and separate sizing/judging subsets. A source-pool size therefore does not establish the number of networks that trained the final coefficients. ([Parameter families and source pool](../experiments/learned-correction/joint-refit-accepts-three-independent-check-steps-fourth-step/code/opus_reexam_20261001_v1/trainer/otr_common.py#L29), [gradient/selection subsets](../experiments/learned-correction/joint-refit-accepts-three-independent-check-steps-fourth-step/code/opus_reexam_20261001_v1/trainer/cert/refit/jmix13.sh#L2).)

**Final per-layer readout fit.** Fifteen constrained linear solves use all 90 training networks and the average of two Monte Carlo residual target streams, with the ridge/damping settings documented above. An earlier feature-selection fit used 72 training and 18 validation networks. The exporter changes readout rows 1–15 and leaves the recurrent encoder, birth coefficients and covariance model fixed. Complete-stack local comparisons in the paper are separate from the fit's in-sample measurements. ([Fit](../release/common/readout/fit_blocks_full90_v1.py#L20), [feature selection](../release/common/readout/lc_blocks2_fold0_v1.py#L1), [export](../release/common/readout/export_blocks_v1.py#L95).)

## Arithmetic routes and worker lifecycle

The main computational work is matrix multiplication for covariance transport and third-order factor pushes and reads. The source contains ordinary products, recursive Strassen/Winograd-style products, symmetry-aware products, shape-specific padding, shared products and grouped strided-read kernels. Many changes reduce the number of metered arithmetic operations or remote requests while retaining the intended algebra; projection, row omission, reduced precision and learned nonlinear approximations change the estimator numerically. These are different types of change and require different validation. ([Recursive product](../submissions/337641/estimator.py#L16), [padding dispatcher](../submissions/337641/estimator.py#L24048), [grouped read](../submissions/337641/estimator.py#L19004), [symmetric first-layer product](../submissions/337641/estimator.py#L24615).)

The covariance product omits sufficiently inactive units using a gate floor of `0.001`, with the retained count rounded up to a multiple of 64. Factor pushes and reads have their own threshold and roster rules. The source's `_FULL_LOWER_MOMENTS = True` flag therefore does not imply that every dense covariance entry is transported without approximation. Recent negative-tail and older-tail corrections have layer-specific settings, including more retained information near the final layers. ([Covariance roster](../submissions/337641/estimator.py#L12432), [active covariance product](../submissions/337641/estimator.py#L14612), [factor-push row rule](../submissions/337641/estimator.py#L5365), [late-layer thresholds](../submissions/337641/estimator.py#L24559).)

`setup(context)` prebinds fixed-shape arenas, views and constant structures before a network is assigned. A later first prediction checks whether the setup-created arena survived on the array server; if it did not, the code resets the affected caches and binds lazily. Prediction wrappers initialize and retire network-dependent state. The source explicitly rejects several nested or concurrent prediction contexts. It is written for the challenge worker lifecycle, not as a thread-safe shared estimator instance. ([Setup and first-prediction check](../submissions/337641/estimator.py#L6611), [prediction-local cache example](../submissions/337641/estimator.py#L9986), [core context ownership](../submissions/337641/estimator.py#L24644).)

The final inner prebinding wall allowance is `_PB_WALL_S = 2.8`; this is only one part of setup, not its total runtime. Setup and prediction wrappers also control Python's cyclic garbage collector, restore settings in `finally` blocks, and perform an explicit full collection before prediction returns. These details affect the wall and residual-time behavior needed to reproduce a hosted run. ([Final prebinding allowance](../submissions/337641/estimator.py#L20554), [collection and freeze](../submissions/337641/estimator.py#L22037), [prediction collection](../submissions/337641/estimator.py#L22054), [outer setup wrapper](../submissions/337641/estimator.py#L23719).)

Both designated files remove grouped-read plans whose first three dimensions satisfy $mkn\geq8.0\times10^9$, then permit at most four grouped reads per prediction. Subsequent reads use the existing fallback. Both also set `_XSG_ON = set()`, selecting ordinary indexing at the former scatter-for-gather sites, and `_FAST_RESCUE_FRAC = {}`, disabling residual-triggered route changes. These final assignments are part of the released algorithm. ([Shared final settings](../submissions/337641/estimator.py#L25701), [read fallback and counter](../submissions/337641/estimator.py#L19004), [residual dispatch guard](../submissions/337641/estimator.py#L22175).)

## Final constants and appended overrides

The file is an accumulated source artifact. Definitions, literals and method assignments are repeatedly appended. A later assignment wins for dynamic global-name lookup; a previously captured parent function still points to the earlier object. Some tables are converted into arrays on first use and then cached. Editing an earlier definition, or changing a constant after a cache has been populated, does not necessarily change the active computation. The final `_astra_fhm_core` therefore also rebinds `_ASTRA_FHM_ACTIVE_CORE`; the on-policy export asserts that `_GNN_PARAMS` and its dependent cache have not been initialized. ([Core rebinding](../submissions/337641/estimator.py#L24949), [readout-cache assertions](../submissions/337641/estimator.py#L25629), [literal decoding](../submissions/337641/estimator.py#L3859).)

Representative final settings are listed below. The full source, including its embedded tensor literals, remains the complete specification.

| Quantity | Final setting in both variants | Source line |
|---|---|---:|
| Shared older-factor ranks at layers 7, 10, 13 | `384, 384, 384` | [7480](../submissions/337641/estimator.py#L7480) |
| Nyström sketch column counts | `1536, 1024, 1024` | [24552](../submissions/337641/estimator.py#L24552) |
| Covariance gate floor / roster rounding | `0.001` / `64` | [12432](../submissions/337641/estimator.py#L12432) |
| Negative-side factor-push threshold | `_DEADCUT_Z = 3.0902` | [13439](../submissions/337641/estimator.py#L13439) |
| Terminal third-order row threshold | `_L15_ROWCUT_Z = 3.0` | [24554](../submissions/337641/estimator.py#L24554) |
| Young negative-tail threshold | `2.5`; temporarily `3.0` at layers **13 and 14** | [24559](../submissions/337641/estimator.py#L24559) |
| Older positive-tail threshold | `2.5`; temporarily `3.0` at layer 14 | [8566](../submissions/337641/estimator.py#L8566), [24576](../submissions/337641/estimator.py#L24576) |
| Third-order row-scale pair | `(0.5425898117822543, 0.5026934507950427)` | [24595](../submissions/337641/estimator.py#L24595) |
| Local fourth-cumulant age strengths | `(0.8143237066192276, 0.6764917767774942)` | [24596](../submissions/337641/estimator.py#L24596) |
| Fourth-cumulant layer-band strengths | `(0.9825906877831871, 0.9812092841443462, 0.9613375186706858)` | [24593](../submissions/337641/estimator.py#L24593) |
| Skewness correction to birth density | `_BSK = 0.434` | [24598](../submissions/337641/estimator.py#L24598) |
| Transport-gate skewness / kurtosis strengths | `_GSK = 0.9`, `_GK4 = 1.04` | [19838](../submissions/337641/estimator.py#L19838), [24603](../submissions/337641/estimator.py#L24603) |
| Trace-mode correction strength | `_BFIX_GAMMA = 0.119` | [19837](../submissions/337641/estimator.py#L19837) |
| Message channels per weight-power family | `8` | [25431](../submissions/337641/estimator.py#L25431) |
| Learned covariance hidden channels | `8` | [25519](../submissions/337641/estimator.py#L25519) |
| Grouped-read cap | `4` per prediction | [25709](../submissions/337641/estimator.py#L25709) |

The young-tail wrapper's executable condition includes both layers 13 and 14, despite nearby older comments describing layer 14 only. Similarly, later assignments disable the direct grouped-load and exact-copy experiments near the end of the file; their earlier definitions remain present. Comments and historical labels must be checked against the last assignments and captured aliases. ([Layer condition](../submissions/337641/estimator.py#L24566), [retired load routes](../submissions/337641/estimator.py#L25502), [later route switches](../submissions/337641/estimator.py#L25611).)

## Designated source identities and exact differences

The source files were checked against the estimator members of their archived submission packages and against the hashes in the package manifests.

| Submission | Source SHA-256 |
|---|---|
| #337619 | `f66cb28baf0c018f25bb749d14ce9e9aedbb47b8f2acd408680cad169485d557` |
| #337641 | `5a4c15405c4462703aeead1bb6ccbc9b98ccf6ba2b5974ce23713303c87d594d` |

The top-level Python syntax trees differ in only four statements: the three definitions of `_fast_rescue_live_residual`, and the final `_GNN_LITERAL` assignment. In #337641 all three helper bodies are replaced by `return None`; the surrounding wrappers and empty final dispatch table remain. The final readout literal encodes strength `1.35` instead of `1.5`. Both variants use the same coefficient record and feature design. All nine exported readout tensors were reconstructed from the preceding literal, the archived coefficients and the stated strength, and their float32 bytes matched exactly. The other 13 tensors in the final readout literal were unchanged. This is a static verification, not a numerical execution of either estimator. ([#337619 final readout](../submissions/337619/estimator.py#L25723), [#337641 final readout](../submissions/337641/estimator.py#L25696), [#337619 last helper](../submissions/337619/estimator.py#L22227), [#337641 last helper](../submissions/337641/estimator.py#L22209).)

The release build record for #337641 describes the source **before** its helper-body edit and records hash `1a29eba718f433cb10dbad0d2ea34999c596627d41bab58541a0b374960283fe`. The submitted file is identified by its package manifest and the `5a4c1540…` hash above. A build-record hash and the final archive-member hash are therefore not interchangeable. ([Build record](../release/337641/build/BUILD_v1.json#L14), [package manifest](../release/337641/extracted/manifest.json#L14), [release explanation](../release/337641/README.md#L21).)

| Recorded hosted quantity | #337619 | #337641 |
|---|---:|---:|
| Official public score | `1.4005566073702658e-09` | `1.4006490071107744e-09` |
| Networks graded / failed | `100 / 0` | `100 / 0` |
| Mean compute utilization | `0.10865223953959685` | `0.10865887533043861` |
| Final-layer MSE, public aggregate | `1.289e-08` | `1.289e-08` |
| Prediction wall, mean / maximum, seconds | `98.587 / 110.304` | `98.678 / 109.745` |
| Residual wall, mean / maximum, seconds | `0.334 / 0.362` | `0.336 / 0.363` |

These are archived hosted measurements, not results obtained while writing this document. The official score is the mean of per-network adjusted scores; it is not the product of the two aggregate means in the table. The archived grading round reports 50 public and 50 private networks, and its public leaderboard score should not be presented as the result of the separately designated private re-evaluation. ([#337619 grade](../release/337619/hosted_grade/SUMMARY.md#L5), [#337641 grade](../release/337641/hosted_grade/SUMMARY.md#L5).)

## Reproduction boundaries

**Inference artifact.** Each submitted `estimator.py` embeds the tables and learned tensors it needs. Its imports are `flopscope`, `flopscope.numpy`, `BaseEstimator` from `whestbench`, and the standard-library modules `base64`, `zlib`, `time`, `math` and `gc`. Embedded float32 blobs are decoded through the metered array interface. There is no dependency on the training captures for inference. The packager records WhestBench `0.14.0`, flopscope `0.10.0` and Python `3.11.5`, with manifest minimum Python `3.11`. These describe the local packaging environment; the hosted judge and the local VM gates used their own environments. ([Imports](../submissions/337641/estimator.py#L1), [decoder](../submissions/337641/estimator.py#L3859), [packaging versions](../release/337641/extracted/manifest.json#L18), [environment/data index](../data/DATA_INDEX.md).)

**Retraining and export.** The release includes the final coefficient record, the export procedure, parent source files, and capture/fitting records. Reconstructing the final readout tensors from those records is distinct from rerunning training. The copied research scripts retain imports, directory assumptions and absolute VM paths from the working environment; the small `release/common/readout/` subset alone is not a runnable training package. For example, the fit imports a chain that expects `fit_extended_gate_v1.py` and `SECOND_FIT_RECIPE_v1.json`, and the capture pipeline assumes `/home/arcworker` paths and its original runner modules. Large captures and truth tensors are separately inventoried and are not all in git. Repeating a fit requires the corresponding dependency mapping, seeds, captures, targets and numerical environment. ([Fit dependencies](../release/common/readout/fit_blocks_full90_v1.py#L10), [local import chain](../release/common/readout/lc_blocks_fold0_v1.py#L10), [capture paths](../release/common/readout/run_capture_pipeline_v1.py#L8), [data inventory](../data/DATA_INDEX.md).)

**Numerical and timing claims.** This document describes source behavior and archived measurements. Source hashing, syntax-tree comparison and literal decoding verify identity and coefficient construction; they do not establish numerical parity across array-client versions, platforms or linear-algebra implementations. Neither the estimator nor its frozen build/training scripts were imported or executed for this inspection. Reproducing the hosted score also requires the evaluation networks, targets, metering rules and worker lifecycle; the source alone cannot reconstruct a private evaluation set.
