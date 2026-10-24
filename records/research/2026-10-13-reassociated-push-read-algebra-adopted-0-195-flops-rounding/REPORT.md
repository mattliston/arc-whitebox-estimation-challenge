# Astra hunt, 2026-10-13

Goal: measured method gains beyond s51k2tw_x (b1d647e9), toward the public top three. No submissions. All experiment artifacts remain in this directory; only measured gains may be announced to TEAM_LOG.

## Constraints

One numerical process at a time, nice 10, CPU only, one BLAS thread, guard_footprint_v3, fresh guard records, Native RSS backstop 7.5 GB and guard v3 default footprint limit 7.5 GB (Matt / LEAD_NOTE.md override); torch RSS and footprint 4 GB; memory_pressure free at least 50%. Teammates perform independent source analysis only.

## Initial evidence

The source candidate sha256 is b1d647e9a340a1f8da7925403636fa2c3a91f58f7e7c21db13ed59db435a8e48. The top two HANDOFF sections and PATH_1p5 ledger were read. Existing live team work owns beta kink/eigenvalue gates and late age-4 bands. No measured Astra gain yet.

## Execution notes

The sandbox rejected setpriority on the first lightweight read (`nice: setpriority: Operation not permitted`). No numerical job was launched. Guard execution will require the normal approved unsandboxed process launch so nice/ps/sysctl work.

## Baseline and pairing audit

Native exact certificate: `../opus_reexam_20261001_v1/scout_native/cert_fs_s51k2twx_136.json`, seed 7100136, 257,830,154,394 billed flops, 39,285 charged requests, prediction SHA `5eb243477612152e`, final MSE 1.5292630539702287e-8. Prior guard: peak RSS 3.928 GB, footprint 3.079 GB.

The scout K result files on seeds 7100584-591 use S50 parameters. They must not serve as S51 controls despite informal labels. All eight exact S51+gk11 learned-ON controls exist instead under `../opus_reexam_20261001_v1/physicist/t134_refit_screen/results/t134_a_ctl_on_<seed>.json`. `harness_audit/sq_base_manifest_v1.json` pins them. A private simulator parity control must reproduce all 16 layer MSEs before using them.

## Active experiments and source leads

- `micro_v1/candidate.py` (28df9de9): scale young read lambda on the smaller roster; move star plane division before rank-one expansion. Native certificate running under guard v3, PB_ALL=1. No measured claim yet.
- `winograd_review/fragment_qz7_v1.py`: exploit the second unused bit in HZ7 remainder-32 padding. Same transforms and depth. Small structural tests next; rounding and request cost require measurement.
- `bank_algebra/fragment_read_merge_v1.py`: combine old trailing and young bank read, saving a shared output reconstruction; fused accumulation version being developed to avoid operand concatenations.
- `physics_alternative/cutsim_sq_v1`: parameter-free Sq=1 correction of newborn mixed fourth-cumulant coefficients S1-S3. S4 unchanged to avoid duplicate inherited kurtosis. Control + paired-eight requests prepared.
- `beta_rule/patch_bankenergy_v1.py`: choose beta direction by newborn pair energy sent into Nystrom compression, with no fitted network labels. Private screen in preparation.

All teammate work so far is source analysis/preparation only. Root owns the single numerical lane. `guarded_entry.py` enforces a shared flock, nice>=10, one BLAS thread, CPU visibility, and repeated memory-pressure gates. Torch worker has equivalent lock, nice check, and a per-layer pressure gate.

### QZ7 structural check and memory adjustment

`qz7_toy_v1.json`: seven forced HZ7 shapes, including quarter-zero row, inner, both, mixed half/quarter, unchanged control, and two packets. All patched outputs are byte-identical to parent. Warm bill savings per affected small product: 3,226,944 to 23,664,256 FLOPs; extra requests +6 to +26. Control unchanged. `qz7_toy_v1_guard.json`: completed, peak RSS 0.458 GB, no swap growth. This is structural evidence, not the required native network certificate.

`micro_v1_guard.json`: no completed certificate; RSS backstop triggered at 4,084,924,416 bytes after 219.85 s, physical footprint peak 3,405,222,712 bytes. No gain claimed. The guard retired only its own experimental process. For subsequent native runs the runner now invokes gc and macOS allocator pressure relief every 5 s, outside the estimator and charged arithmetic; the existing torch engine uses the same API. The 4 GB guard remains unchanged.

`qz7_v1/candidate.py` (d68a4401) now under native certificate. `qz7_rlf_micro_v1/candidate.py` combines QZ7 with reviewed read-leaf fusion and small star/lambda reassociations; prepared but not run.

## Updated native memory authorization

Matt directed reading `LEAD_NOTE.md` and using `--rss-backstop 7500000000` for native certificates. The note also specifies guard v3 default 7.5 GB footprint limit; all other constraints continue, and torch stays at 4 GB. Updated `launch_native_v1.sh` accordingly. No running process was killed or changed by us. The already-running QZ7 attempt ended at its old 4 GB RSS backstop (4.157 GB RSS, 2.982 GB footprint, no swap growth); its terminal guard record is `qz7_v1_guard.json`. A new certificate with the authorized limits is `qz7_native_v2`, launched only after the old process was authoritatively terminal.

### QZ7 native result: no gain on required seed

`qz7_native_v2_cert.json` and `_comparison.json`: seed 7100136, PB_ALL=1. Trial bill 257,830,154,394 and requests 39,285, both exactly equal to baseline; prediction SHA 5eb243477612152e unchanged. `_QZ7_STATS` reports zero affected products and packets. Thus the structurally valid patch does not help this required network and is NOT a landed gain. Guard completed in 346.01 s; peak RSS 3.733 GB, footprint 3.120 GB, no swap growth. No TEAM_LOG gain line was written.

Next: D96 structural check, which targets an active padded inner dimension; then reviewed bank-read leaf fusion and small reassociations.

### D96 structural check / native stack

`d96_toy_v1.json`: four affected shapes plus an unaffected control, checked cold and warm, with encoded-zero support assertions. Warm savings 3,486,240–14,502,016 FLOPs; +17 to +28 requests; relative output difference versus parent 5.85e-5–6.84e-5 on random matrices. Comparison against float64 multiplication remained within the test accuracy bound. Control was byte-identical and bill-identical. Guard completed: RSS 0.485 GB, footprint 0.320 GB.

Unlike QZ7, D96 has existing depth-7 padded calls in the old 7100136 census at layers 5, 11, 12; current applicability is still being measured. `native_stack_v1/candidate.py` (6562cc7e) combines D96, reviewed RLF, read-lambda, star division movement, and star beta fusion. QZ7 is excluded. Native run `native_stack_v2` is active. The preliminary `native_stack_v1` run ended immediately with a missing candidate path after a preparation path error; the path was corrected and input-existence checks added before launch.

Allocator relief is now opt-in (`ASTRA_MEMORY_RELIEF=1`); subsequent 7.5 GB native runs use the normal allocator by default. Layer progress records added to the wrapper have no effect on charged arithmetic. Future launches snapshot their wrapper source.

## Measured gain 1: native algebra stack

Timestamp 2026-10-13T02:13:56Z. Candidate `native_stack_v1/candidate.py`, sha256 6562cc7e5cdcfe3f056eb5743feb67e413908d1c22cbc89428ba7a61a3a154d0; source manifest `native_stack_v1/manifest.json`. Certificate `native_stack_v2_cert.json`; predictions `native_stack_v2_cert_pred.npz`; full comparison `native_stack_v2_cert_comparison.json`; guard `native_stack_v2_guard.json`.

Seed 7100136, PB_ALL=1 versus exact b1d647e9 baseline:

- Bill 257,830,154,394 -> 257,328,286,683: -501,867,711 FLOPs, **-0.1946505%**.
- Requests 39,285 -> 39,097: **-188**, estimated **-0.94 ms** at platform 5 us/request.
- Prediction SHA 5eb243477612152e -> 063146b176eadc7d. Max absolute difference 1.3947487e-5; final squared difference 8.2813859e-12.
- Squared difference / base final MSE = **0.0541528%**, below half bill saving **0.0973253%**: satisfies the user bill-gain criterion.
- Actual final MSE on this seed **+0.1343734%**. Observed bill-times-MSE score change **-0.0605387%**. The single-seed MSE change is reported, not hidden behind the squared-difference bound.
- RLF activated8reads; D96 activated3products/12packets; lambda-only activated3reads (1,329,152 scalar products saved).
- Completed guard: RSS3.405GB, footprint4.388GB, no swap growth; pressure observations all >=50%. Elapsed571.39s.

One measured line appended to the team log, requesting a worker extension. This does not establish public-board improvement or top-three status. Next isolate D96+RLF from the small reassociations to reduce rounding cost.

## Core ablation: smaller prediction distance, worse observed score

Candidate `native_core_v1/candidate.py`, SHA75e3415a8cdb754f48f3b0335863327fc65f5cb3e40d0eacbaa9d0051ab16d02, contains only RLF and D96. `native_core_v1_cert_comparison.json` measures seed7100136 PB_ALL=1: bill **-0.1824553%** (-470,424,895 FLOPs), requests **-143** (-0.715ms at5us/request). Final squared prediction change/base MSE **0.0454160%**, below half bill gain0.0912277%; final squared change6.9453018e-12. Prediction SHA5c742cead387eb66.

Actual final MSE **+0.4419866%**; observed bill-times-MSE score **+0.2587248%**. This meets the stipulated bill admission rule but is not a preferable observed-score candidate. Keep full stack as the current better measured candidate. Dropping the small reassociations did not solve rounding cost. First five layer predictions are unchanged; drift starts at layer5. A separate no-D96 ablation remains needed.

Guard `native_core_v1_guard.json` completed in513.17s, max RSS4.018GB, footprint4.255GB, no swap growth. Pressure gate paused while measured free memory was below50%, then resumed. No other numerical process overlapped.

## Sq control validated; paired8 begun

`harness_audit/run_sq_control_v1/results/sc_astractl_r7100584_ctl.json` reproduces true S51+gk11 control final MSE2.1750184674017685e-8 and all16 layer MSEs exactly. The comparator completed in the guarded process: `harness_audit/guard_sq_ctl_v1.json.comparison.json`. Guard completed261.54s, maxRSS2.733GB, footprint1.452GB, no swapgrowth; per-layer pressure>=50%. The upcoming eight fixed Sq requests use the validated trueS51 controls, not the older mislabelled S50 scout-card results. Launch entry and harness sources recorded in `harness_audit/sq_arm_launch_manifest_v1.json`.

## Current experiment queue and interpretation

The root owns one numerical slot. Source preparation is parallel across six agents. Sq paired8 is active; no other numerical experiment overlaps it. The remaining priority order is: star-moment beta control/paired8; guarded distribution toys followed by radial/common-factor controls and arms if their identities hold; no-D96 native ablation and byte-preserving HZT; the fixed late-band qold proposal if earlier screens do not supply enough gain. Ordering may change for short certificates or a source blocker; mathematical settings do not change in response to partial screen results.

- Star beta: `beta_rule/STAR_MOMENT_RULE_v1.md`, final pbase31f7ee3e; peer review cleared actual post-F4 state and birth-cache consistency. No-op control and runtime assertions remain mandatory. Native cost is unmeasured.
- Radial covariance: `physics_alternative/BIG_GAIN_RSCOV_V1.md`; independent toy required, screen thresholdmeanlog<=-1% because native work has a cost.
- Common mean/scale factor: `physics_alternative/COMMON_FACTOR_NEXT_V1.md`; independent toy and control required, screen threshold<=-2%; full finite-model K3/K4 subtraction uses derivatives of the finiteR6 map, not a claim of identical truncation to every existing base cumulant term. Five-node quadrature superseded the unrun three-node draft.
- Mean-direction beta sensitivity: exact rational Isserlis certificate is prepared but unexecuted; diagnostic only, no change to initial starbeta arm.
- No-D96 subset: `winograd_review/native_nod96_v1/candidate.py`, SHA4bc3a33bed5a1fdfe3a04f3cfd93b68a0b733791fae3589bbf6ddc35c137bd7b. RLF is an exact real-arithmetic identity but not byte-equivalent: shared leaf addition and decode change float32 association.
- HZT: source-reviewed unconsumed input-transform strip pruning, intended byte-equivalent. Toy/native result still pending.
- Qold late-band: strong saved gradient evidence is offset by a prior held-out qold test (+0.084% SE0.166, 25/50lower); the new L11–14 restriction is an unmeasured hypothesis, not a recovered established gain.

The other team has begun a fresh eight-network extension of the original algebra stack (reported by lead/scout-native in TEAM_LOG around02:14–02:17Z). Root does not submit or operate their workers. Public top-three improvement remains unestablished.

## Sq paired8: null; no gain
True S51+gk11, fixed Sq coefficient1 on layers0–14, seeds7100584–591: mean log-MSE **+0.0160626%**, SE **0.0243962%**, pooled **+0.0113882%**, 4/8 lower. Does not meet -0.15% screen. No native port, no TEAM_LOG gain line, and no settings changed after seeing results.
| Seed | Log-MSE change |
|---|---:|
| 7100584 | +0.0342863% |
| 7100585 | +0.0444429% |
| 7100586 | -0.0482758% |
| 7100587 | +0.0940157% |
| 7100588 | +0.1194909% |
| 7100589 | -0.0105236% |
| 7100590 | -0.0281879% |
| 7100591 | -0.0767476% |

Evidence: `harness_audit/guard_sq_arm_v1.json.comparison.json`, individual results under `harness_audit/run_sq_arm_v1/results`, and exact control comparison in `harness_audit/guard_sq_ctl_v1.json.comparison.json`. Guard completed in1729.20s, maxRSS3.258GB, footprint1.564GB, no swapgrowth.

## Short mathematics and exact prediction-norm audit

`rscov_math_norms_v1.log` / `_guard.json` completed in6.88s, maxRSS0.213GB. The radial covariance toy matches independent raw-joint autograd derivatives within2.66e-16, has small-r norm ratio4.000854 (quadratic order), identity-ReLU residual2.29e-18, symmetric exactzero diagonal and exact fallback. This is formula validation, not modelgain. Three independent exact-rational certificates also succeeded: mean-direction sensitivity48cases, conditionalvariance12, conditionalcorrelation12.

Lead-requested final |d|/|pred| now measured from saved predictions under the same numericalguard: basepredictionRMS0.9288412156471986; fullstack **3.09820416894e-6**, core **2.83729232124e-6**. Evidence `native_stack_v2_cert_norms_v2.json` and `native_core_v1_cert_norms_v2.json`; all priorbill/MSE conclusions unchanged. No estimator rerun was required.

## Short certificate batch completed

- `cfcov_toy_v1.log`/guard: fourunit independent connected-derivative and positive-mixture tests succeeded, maxlinear error4.025e-16, maxremainder error2.915e-16; smallstrength normratio3.998565, identitylimit4.684e-16. Four declared r/beta cases includingbeta0. Guard9.02s,maxRSS0.215GB. Mathematical support only, networkgain unmeasured.
- `hzt_toy_v1.json`/log/guard: affected output and control bytes equalparent, billedstrip savings checked; fusedtoy includescold/warm shapes and NaN-poisoned scratch. Guard11.54s,maxRSS0.195GB. Native networkcertificate stillneeded.
- `rlf_final64_toy_v1.json`/log/guard: all fused/unfused andHZrow/column flag combinations checked. OFF isparent byte/bill/request-identical. ON equals independentdeep-f32/final-f64 reconstruction bytes; actualweightedbill extra2*M*N andchargedrequests+3, including explicitdouble-loop writes tofloat32 outputs. Guard11.89s,maxRSS0.341GB. No underpricedtype behavior used.
- `star_k21_toy_v1.log`/guard and `harness_audit/certificate_star_k21_v1.json`: exact-rational Gaussian polynomial tests support both reference conventions for the offdiagonalK21 objective. This is extra diagnostic evidence; the queued beta selector stillminimizes its original J3 and doesnotchange based on screen results.

Next active native run: `native_nod96_v1`, fullalgebra stack minusD96, prepared SHA4bc3a33b. A tiny plannerprobe runs immediatelybeforeit under itsownfreshguard to check finaloverrides, which invalidated an earlier source-onlyYCP table.

## Independent fresh8 extension supports the algebra stack

Other team source: `O/scout_native/seal_s51k3_judge.out`, reported2026-10-13T02:56:33Z. Their candidate s51k3 SHA6bc5dc3e combines our original native_stack6562cc7e with their separately certified identity-cache cut idc1. Comparison against b1d baseline on preregistered7190008–015 has **mean log(bill*MSE) -0.2853%, SE0.1161%, 6/8lower**. Every network uses fewer requests. This is combined-stack evidence, not an isolated estimate of our contribution or a public-grade/top-three result. Root did not submit or operate their workers.

```text
7190008  MSE +0.4064%  bill -0.2880%  score +0.1172%  requests -63  pred changed
7190009  MSE -0.3001%  bill -0.1229%  score -0.4235%  requests -311  pred changed
7190010  MSE +0.2540%  bill -0.2220%  score +0.0315%  requests -188  pred changed
7190011  MSE -0.6000%  bill -0.2845%  score -0.8867%  requests -137  pred changed
7190012  MSE +0.1265%  bill -0.1914%  score -0.0652%  requests -164  pred changed
7190013  MSE -0.3189%  bill -0.1977%  score -0.5173%  requests -177  pred changed
7190014  MSE -0.0289%  bill -0.3090%  score -0.3383%  requests -66  pred changed
7190015  MSE -0.0154%  bill -0.1846%  score -0.2002%  requests -230  pred changed
MEAN -0.2853% SE 0.1161% LOWER 6 OF 8 -> FIRE
```

Plannerprobev1 produced data but had a final print-only TypeError (`flush` passed to json.dumps). Root corrected that print, preserved v1 artifacts, and reran fresh `planner_probe_v2`: guard completed2.17s; final imported planner constants verified. The noD96 certificate was started only after v2 completed.

## No-D96 ablation: rejected by the specified drift bound

`winograd_review/native_nod96_v1/candidate.py`, SHA4bc3a33bed5a1fdfe3a04f3cfd93b68a0b733791fae3589bbf6ddc35c137bd7b, is the original full algebra stack with onlyD96 removed. Native7100136 PB_ALL=1 evidence `native_nod96_v1_cert_comparison.json`: bill **-0.086143994%** (-222,105,193FLOPs), requests **-338** (-1.69ms). RelativepredictionRMS **3.442983431e-6**, d2/baseMSE **0.066876052%**, which exceeds half bill gain **0.043071997%**. Thus the user admission criterion is false. ActualMSE **+0.471781478%**, observedscore **+0.3852311%**. No TEAM_LOGgainline.

D96 removal did not produce a byte-identical or better-supported subset; RLF and the smallstar/lambda changes also change rounding. Retain originalstack; this is an ablation of the same family, not additive supply. Guard completed227.89s, RSS4.477GB, footprint3.840GB, no swapgrowth.

The beta J3 selector is now in no-op control using the reviewed diagnostic-v2 snapshot. J3 choices/outputarithmetic remain identical to originalplannedarm; additional J21 telemetry is pinned before any armoutcomes. Sourceaudit and exactrational certificate account for bothreferenceconventions, next-layerpair presence, and S51young-readscaling.

## Beta selector control established; eight-network arm running

Control `beta_rule/guard_starmoment_control_v3.json` completed in214.15s, maxRSS2.854GB, footprint1.291GB, no swap growth. All16 layer MSEs match the true S51+gk11 control exactly, finalMSE2.1750184674017685e-8. Evidence `beta_rule/guard_starmoment_control_v3.json.comparison.json`. An earlier launch (`control_v2`) ended before claiming a request because the copied run lacked empty claimed/results directories; only those directories and the preparer were corrected, with all9 request hashes and mathematical source hashes unchanged.

The pinned diagnostic-v2 arm is now running seeds7100584–591 under `beta_rule/guard_starmoment_arm_v2.json`; root owns the sole numerical slot. It retains the original J3 chooser and adds only predeclared J21 diagnostics. No coefficient/selector changes from partial outcomes are permitted.

Source audit clarification: RLF itself already moves young-lambda scaling to the left when m<n. Thus the earlier core ablation removed the separate read_lambda fragment but did not remove every lambda reassociation. `winograd_review/native_stack_rightscale_v1/candidate.py` restores the baseline right scaling inside fused reads only; +8,787,968 seed136 scalar FLOPs predicted (+0.003408% baseline bill), zero additional calls. It is an unmeasured rounding isolation; no gain claim.

## Source-only priorities during the beta cohort

RSCOV is the next full error screen. Current estimated FP64 native overhead is about0.58% before remaining scalar/vector work; using the other team's historical0.8 raw-to-board transfer assumption, about1.98% raw reduction would be needed for1% net gain. CFCOV's current literal overhead is about2.39%; even the unimplemented lower-cost design is about1.58%, so its2% raw screening threshold is not sufficient for promotion. These are source estimates, not native measurements.

The additive five-node law is a research diagnostic only. A radial fourth tensor with C=I and equal means can nearly perfectly fit the observed rank-one planes while falsely implying unobserved all-distinct moments. Good observed fit and valid positive-law parameters do not establish physical coherence. Prepared root bundle `check_additive_bundle_v1.py` pins the four reviewed math sources; execution remains pending behind the beta cohort.

Numerical conditioning candidates remain isolated: fullstack + rightscale restores only fused young-lambda placement; fullstack + final64 promotes only final reconstruction; fullstack + BLcancel (`winograd_review/native_stack_blcancel_v1/candidate.py`, SHA7d6d73a23d483579b18a50868d9271a8769d478f0b4fcd354dcba797ff78014e) removes an algebraically cancelling Z4 from bottom-left reconstruction. BLcancel is predicted to preserve ordinary bill/request count, but its root toy and native certificate are not yet run. It is an alternative to final64, not composed with it. These are ablations of the existing stack, not additional landed gains.

The conditional newborn-merge proposal is now closed source-only: bank audit identifies it as the same [young|newborn-common] NB-MERGE previously evaluated by scout-native, with a new price gate rather than a new identity. Its small corrected cold ceiling and request cost do not justify reopening the user's closed joins line. No numerical run was added.

The cheap radial mean-only idea is likewise not a new big-gain lead on present evidence: its leading remainder is exactly the prior T68 e44 term, r^2*sigma*He6(alpha)*phi(alpha)/1152. A full same-law resummation differs only at higher radial orders; there is no measured evidence those orders are large. A scalar-variance remainder must also avoid double-counting the existing squared linear mean shift. No new mean/variance arm was queued.

## Beta J3 paired8: null; no gain

Pinned true-S51+gk11 seeds7100584–591: mean log-MSE -0.025503704%, SE0.156628449%, pooled 0.027255724%, 4/8 lower. The stipulated -0.15% threshold is not met. No native port or gain announcement; no selector re-fit to this cohort.

| Seed | Log-MSE change |
|---|---:|
| 7100584 | +0.638502156% |
| 7100585 | -0.588695955% |
| 7100586 | +0.579986148% |
| 7100587 | -0.385115887% |
| 7100588 | -0.344238580% |
| 7100589 | +0.022097642% |
| 7100590 | -0.144146384% |
| 7100591 | +0.017581224% |

Evidence: `beta_rule/guard_starmoment_arm_v2.json.comparison.json`, frozen v2 manifest and results. Guard completed1648.66s, maxRSS3.722GB, footprint1.737GB, no swapgrowth. All56 layer choices and J21 diagnostic rows were checked by the same-process comparator. Diagnostic interpretation may explain this result but will not retroactively select a new rule.

## New mathematics and decoder toys completed

`additive_math_v1.log` / guard completed9.84s, RSS0.206GB. Additive observed-fit reference error6.66e-16 and raw tensor error6.94e-18; all8 Skellam law cases,4 exact F4 trace cases, and6 exact covariance-support cases succeeded. Artifacts include `star_algebra/skellam5_certificate_v1.json` and `star_algebra/covariance_support_certificate_v1.json`. These establish identities/necessary inequalities, not a network improvement or a complete physical moment model.

`rlf_oldroute_v1.json` executes the final baseline dispatcher with metadata operands: seven of the eight old trailing products take HZG depth6; shape(672,320,800) takes ordinary AB depth4. Thus RLF changes that old-product route from D4 to D6, beyond lambda reassociation and sum/decode association. Probe guard completed2.97s,RSS0.116GB; no prediction or numerical matrix product ran.

`rlf_canonical_toy_v2.json` / guard completed20.20s,RSS0.383GB. Independent integer tensor map, floating roundoff bounds, OFF identity, fused/unfused/HZ-strip cases, exact extra arithmetic and unchanged warm requests all checked. v1 ended on a JSON-only numpy.float32 serialization error; root changed only serialization of the bound to float, preserving numerical assertions and decoder/candidate sources, then used a fresh guard/output for v2. Native original-fullstack+canonical SHA45c3a26e remains unmeasured.

Next active certificate is `hzt_native_v1`, HZT-only candidate SHA357ef9bd on b1d baseline, native7100136 PB_ALL=1, RSS backstop7.5GB. No performance conclusion yet.

## Landed HZT byte-preserving bill cut

Candidate `winograd_review/native_hzt_v1/candidate.py`, SHA357ef9bd99f7bc324a5719e7b2d573a2dd793702756cfab9812d21db1652e3bd, seed7100136 PB_ALL=1: bill257830154394→257783733460, saving46,420,934 (**-0.018004463%**). Requests39285→39342, **+57** (+0.285ms at5us/request). Prediction SHA remains5eb243477612152e; all16 prediction-delta MSEs exactlyzero. Actual MSE unchanged; qualifies as a measured bill gain. HZT activated38A/43B transforms.

Evidence `hzt_native_v1_cert_comparison.json`; guard completed323.25s,maxRSS4.256GB,footprint4.215GB,no swapgrowth. User-required7.5GB native backstop applied; pressure gate observations remained>=50%. One team-log gain line appended at2026-10-13T03:59:24Z, requesting stacked byte/count extension. This is not yet a public-grade result.

## Canonical RLF decoder native ablation — no incremental gain

Candidate `winograd_review/native_stack_canonical_v1/candidate.py`, SHA45c3a26e524583a338489aa520f957cfda36c473b931bbc5b4bbf2b41371f394, replaces the original stack's all-level RLF reconstruction only. Native7100136 PB_ALL=1: bill257360835339, requests39097. Against b1d this is −0.1820264414% bill, −188 requests; prediction SHA0ead0ab3cc34a9b8, final deltaMSE9.0206339221e-12, d²/baseMSE0.0589868035%, relative prediction RMS3.2335311650e-6; actual final MSE+0.2627451098%. It meets the baseline admission inequality but repeats an already measured stack gain.

Against the original stack this adds32548656 bill (+0.0126486895%) with unchanged requests and worsens actual final MSE by0.1281994674%; relative prediction RMS2.0128018615e-6, d²/baseMSE0.0228254496%. Eight reconstructed reads and4784128 output elements match the predicted extra bill exactly. No new gain announcement; original stack remains preferred. Evidence: `canonical_native_v1_cert_comparison.json`, `canonical_native_v1_cert_secondary_comparison.json`. Guard completed336.015s, maxRSS3928195072, maxfootprint4534375848, zero swap growth.

## RSCOV prerequisite control and optional diagnostics completed

`harness_audit/additive_control_guard_v1.json.comparison.json` verifies all16 MSEs exactly equal true S51+gk11 t134 and the earlier Sq control, plus all1024 final FP64 prediction bytes equal Sq (SHA94757511cf9218a3ae60b14d62cbf3ab75172d6184f4df1422ec84f663e1261a). Seven observer rows L8–14, one physical bank read per layer, RDC2 masks applied, no arm. Source/model pins unchanged. Guard {"completed": true, "elapsed_s": 249.05409975070506, "max_rss_bytes": 2614263808, "max_footprint_bytes": 1557254080, "max_swap_growth_bytes": 0}.

Compact telemetry is in `additive_control_summary_v1.json`. The rank-one fourth-plane fit is almost entirely explained by the already-carried radial plane; it is not an identified new latent source. No row/diagonal support gap is negative beyond its screening envelope on these seven rows. Those necessary tests are loose and are not a PSD certificate. Full three/fourth tensor identification remains false.

After authoritative control exit0, launched unchanged fixed RSCOV1 at L8–14 on7100584–591 via `harness_audit/launch_rscov_additive_arm_v1.sh`, guard `harness_audit/rscov_additive_arm_guard_v1.json`, log `harness_audit/rscov_additive_arm_v1.log`, session8021. Arm manifest91642f4c pins the successful control and rechecks it after all8. Predeclared big-gain gate remains meanlog<=−1%; source-only cost and formula certificates do not establish benefit.

## Final RSCOV result and graceful wind-down — 2026-10-13T04:42:20Z

COMPLETED VALID NULL. Fixed RSCOV1 at L8–14 on all8 seeds7100584–591. Mean log-MSE change -0.0167022828%, SE0.0342610112%, pooled-0.0160824758%, 5/8 lower. Per-network log-percent changes: 7100584: -0.0205447743%, 7100585: +0.0982037804%, 7100586: -0.0301447591%, 7100587: -0.1948283614%, 7100588: +0.0205216595%, 7100589: -0.0612373119%, 7100590: -0.0545902010%, 7100591: +0.1090017051%.
Fails both the user−0.15% error-gain threshold and its predeclared−1% big-gain port gate. No port or coefficient/band/fallback retuning is proposed; no gain notice. Mathematical mechanism remains partly unresolved because activation/norm telemetry was not recorded in this arm.
All eight exact layers0–7 match their own true S51 baseline, borrowed-control hashes were rechecked, all8 completed. Authoritative session8021 exited0. Guard elapsed1810.502s, maxRSS2899738624, footprint1778044936, zero swap growth. Evidence: A/harness_audit/rscov_additive_arm_guard_v1.json.comparison.json; independent stdlib postcheck A/harness_audit/rscov_postcheck_v1.json (VALID). Comparison SHAee19856acd6430961d05def0feb6ff5569923a8a66bb0d4c80bdf1ff40d2a121. The fixed8 screen is CLOSED without a measured gain; RSCOV native port work should remain parked.

Goal paused at Matt's request. All agents wound down, existing numerical process completed, shared lane lock free; no new experiments started. Complete handoff: `OPUS_HANDOFF_20261013.txt`, SHA256 96b16b68ada6f1aa23cc3a56ba6f7ded1db86848a28eae311c7e70d9d6e59370. Final timestamped notice sent to Opus in TEAM_LOG.md.
