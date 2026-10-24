# Fable handoff — K21 / pure-closure Phase 2

Status: final canonical handoff. The H7B all-688 residual-wall recovery, exact sequential parity audit, three-call literal meter, package audit, authorized board submission, terminal graph reconciliation, rendering, and validation have all completed successfully. No scientific workstream is active. `P2R0F2P` is the sole open graph leaf.

Repository root: `/Users/matt/arc-white-box-estimation`

## 1. Mission and precise stopping state

The goal is a deterministic, sampling-free estimator for the Phase-2 width-1024, depth-16 MLP task with either:

1. raw final-layer MSE strictly below `1e-8` at full compute, preferred for its higher ceiling; or
2. any accuracy/compute combination whose exact official adjusted score is strictly below `1e-8`.

The currently banked public-board best is submission `#328326`:

- raw final-layer MSE: `2.9409640056599073e-8`;
- adjusted public score: `2.847207506864414e-8`;
- grading: terminal, successful;
- evidence: `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/BOARD_RESULT_328326.json`;
- receipt SHA-256: `1f8742c2724477aa13e4478e045c1d09a7e68a20819ef5cb5cc52c293906cc93`;
- submitted estimator SHA-256: `2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76`;
- submitted archive SHA-256: `4817aaec22207a1903040b3b35f76c1ab12f0cdce599684041cedb9525ee28be`;
- passing meter receipt SHA-256: `b6aedc76b190702ac0b4bcf2c567d40e0f533b861ccde95bcb17ac21627eb365`.

This improves prior pilot-bank submission `#328317` by `5.622354%` in raw MSE and `5.623483%` in adjusted score. Its prior receipt remains useful as a protected-pilot comparison: `public_q24_native_transport_energy_pilot_bank_v1/BOARD_RESULT_328317.json`, SHA-256 `b7064870bc74fffca92caf681608e86a9ea808c807222271e70f6ee5fbaa60f7`.

The immediate stopping operation is a separate all-688 final-fit upgrade, node `P2R0H7B`. Its first literal meter was valid in every respect except residual wall: `0.49450173741206527 s` against a `0.4 s` cap. The user explicitly authorized one independently sealed mechanical residual-wall recovery and, on a clean pass, submission before handoff.

That recovery is now package-valid. Its scientific outputs are exactly unchanged; the mechanical optimization meters first-call scratch creation and then persistently reuses fully overwritten fixed-width scratch arenas.

- recovery branch: `public_q24_native_transport_energy_final_workspace_v4`;
- estimator: `public_q24_native_transport_energy_final_workspace_v4/estimator.py`, SHA-256 `2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76`;
- independent three-call sequential byte-parity receipt: `public_q24_native_transport_energy_final_residual_opt_audit_v1/SEQUENTIAL_PARITY_RESULT.json`, SHA-256 `81a1054b84073439ebaf20af87c6689d6bf99c845ff3a9245f2947b131bd89d9`;
- literal three-call meter: `public_q24_native_transport_energy_final_workspace_v4/METER_RESULT.json`, SHA-256 `b6aedc76b190702ac0b4bcf2c567d40e0f533b861ccde95bcb17ac21627eb365`;
- residual walls: `0.3215303928591311 s`, `0.34646319644525647 s`, and `0.35143496841192245 s`, all strictly below `0.4 s`;
- FLOPs by call: `2,132,366,833,051`, `2,123,756,968,456`, and `2,132,363,680,256`, all below `2^41`;
- literal meter walls by call: `24.89165858272463 s`, `24.72982758283615 s`, and `24.954987000208348 s`;
- maximum RSS across the three calls: `3,421,503,488` bytes;
- every call finite, exact-byte-parity, `float32[16,1024]`, and conjunctively passing;
- archive: `public_q24_native_transport_energy_final_workspace_v4/submissions/q24-native-all688-workspace-v4-2fd2d40d.tar.gz`, SHA-256 `4817aaec22207a1903040b3b35f76c1ab12f0cdce599684041cedb9525ee28be`;
- archive has exactly `estimator.py` and `manifest.json`;
- package validation: `public_q24_native_transport_energy_final_workspace_v4/PACKAGE_VALIDATION_RESULT.json`, SHA-256 `2d5e0d6a3182743dfce2fbdd36a4677524d2498c7839dd7791b45fe249b65e34`, status `PASS`;
- bank-ready receipt: `public_q24_native_transport_energy_final_workspace_v4/BANK_READY_RECEIPT.json`, SHA-256 `7539087bdb7908bfd0011029532cd7e9274a6a8ebb9e6520f2858306efeaec53`, status `READY_FOR_ROOT_SUBMISSION`.

All paths in this block are relative to `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/`.

Terminal board result:

- submission id/status: `328326` / `graded` (`Graded successfully`, terminal watcher);
- raw final-layer MSE: `2.9409640056599073e-8`;
- adjusted score: `2.847207506864414e-8`;
- board receipt: `public_q24_native_transport_energy_final_workspace_v4/BOARD_RESULT_328326.json`, SHA-256 `1f8742c2724477aa13e4478e045c1d09a7e68a20819ef5cb5cc52c293906cc93`.

Do not confuse the following three accuracy numbers:

- `3.107235383605394e-8` is the protected 40-network pilot raw MSE, a legitimate predeclared development validation result with `40/40` wins and worst network `4.093294327260766e-8`.
- `2.9736792457285063e-8` is the all-688 final model's **in-sample observer_fit diagnostic**, not held-out and not checkpoint-selection evidence.
- `2.9409640056599073e-8` from terminal board submission `#328326` is the authoritative accuracy of the submitted all-688 artifact.

The terminal graph receipt records no live experiment, meter, package process, submission watcher, training process, or scientific workstream. A fresh session must still verify the process state before launching work rather than assume an old census remains current.

## 2. Score law and engineering constraints

The Phase-2 score is computed per MLP and only then averaged:

`s_m = MSE_m * max(0.1, F_m / 2^41)`

`S = arithmetic_mean_m(s_m)`

Residual time is unpriced (`lambda = 0`) but remains a validity gate. Relevant limits:

- exact FLOP budget `2^41 = 2,199,023,255,552` per MLP;
- residual-wall cap `0.4 s`;
- total wall cap `120 s`;
- RSS cap `8 GiB`;
- output must be finite `float32[16,1024]`.

The exact score law receipt is:

`pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_adjusted_score_pareto_v1/RECEIPT.json`

SHA-256 `12e82f5b952c6b2786c834f028360cb25bea483f6ee0261a6ee1b198d4ff81e2`.

Never reconstruct an official aggregate as `mean(MSE) * max(0.1, mean(F)/2^41)` unless every row has identical `F`; per-row FLOP variation and its covariance with error matter.

Useful constant-F break-even points from the receipt:

- raw `3.0e-8` needs `F <= 733,007,751,850`, about one third budget;
- raw `2.5e-8` needs `F <= 879,609,302,220`, about 40% budget;
- raw `2.0e-8` needs `F <= 1,099,511,627,775`, about half budget;
- raw `1.5e-8` needs `F <= 1,466,015,503,701`, about two thirds budget;
- at essentially full budget, raw must be below `1e-8`.

The first H7B meter used `2,132,366,833,051` FLOPs, leaving `66,656,422,501`; total wall was `25.436657208018005 s`; maximum RSS was `3,218,259,968` bytes. Its only failed gate was residual wall. Evidence:

`pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_bank_v1/METER_RESULT.json`

SHA-256 `c8405b52b03c367a1c09b80c4ad4ab97fca0017707e96bb93d66acdf830b1949`.

This retry was reasonable because the pilot's materially identical runtime path had already passed residual wall at `0.3627213165 s`; source-file differences were payload/weights and artifact bindings, not a different scientific path. Preserve exact output parity and science semantics during any mechanical runtime repair.

## 3. How the estimator works

This is deterministic distribution propagation, not Monte Carlo and not a learned sampler.

At each layer the host estimator propagates an approximation to the activation distribution:

1. Linear layers transform the mean, covariance, and connected third-/fourth-order cumulant state.
2. ReLU moments are evaluated with an Edgeworth/Hermite expansion.
3. `Q24` means Hermite orders `0..24`, i.e. 25 expansion terms. `Q32` means orders `0..32`, i.e. 33 terms. These are truncation orders, not sample counts, widths, or bit widths.
4. Covariance is repaired to be PSD while preserving the required diagonal.
5. Third-order structure is carried with a capped CP3/full-birth representation plus cubic residual structure.
6. Fourth-order structure is carried with capped signed CP4 factors plus quartic/pairing residual structure.
7. Every layer's predicted mean is emitted, with the final layer being the primary raw score.

The current learned `J` correction is native to the Q24 host rather than a detached post-hoc file. It consumes lawful target-free Q24 quantities (`raw20`, scales, transport and energy/coherence features), carries four signed physical recurrent states through an H96 cell, uses exact `Phi * (state @ W)` transport plus energy terms, and adds a learned mean correction to the Q24 rollout. Architecture receipts report 36 input features, four channels, hidden width 96, and 25,600 parameters. Deployment uses zero truth inputs.

## 4. H7B all-688 final-fit evidence

The exact 688/688 target-free native tape was generated and the preregistered fixed six-epoch fit completed. There was no validation split and no checkpoint selection.

Core artifacts:

- generation preregistration: `public_q24_native_transport_energy_final_v1/GENERATION_PREREGISTRATION.json`, SHA `427a99e2b2cfe7701b8b3e1465f20d34c45583bc84165df24a61d14992d8ebeb`;
- generation manifest: `public_q24_native_transport_energy_final_v1/GENERATION_RUN_MANIFEST.json`, SHA `6b8ea7d9df9c4bb6169ce6d870f8604c891e1315461def7f073de05a33d314d4`;
- generation seal: `public_q24_native_transport_energy_final_v1/GENERATION_SEAL.json`, SHA `8959fd274df3120003b71229a5453b5ee2bc1c875c4e9a139d2e72d104135a6b`;
- fit preregistration: `public_q24_native_transport_energy_final_v1/FINAL_FIT_PREREGISTRATION.json`, SHA `6ae44ffd8ff1b34dd4eaae95aa10d2d4743d18691cbfe154a8b1f383f9b09e84`;
- fit manifest: `public_q24_native_transport_energy_final_v1/FINAL_FIT_RUN_MANIFEST.json`, SHA `23c35e2345941bcde7537accc72c52c31c45537b533201c9be36dfd86675892c`;
- fit seal: `public_q24_native_transport_energy_final_v1/FINAL_FIT_SEAL.json`, SHA `5b8eb05d29889f041fb11a39e0ab4ce496b6ebe8ebf9e92f0d68340d9124696f`;
- terminal fit result: `public_q24_native_transport_energy_final_v1/FINAL_FIT_RESULT.json`, SHA `e8261895c5d8f5f1feb1e15f257612d1d1b36a90e8bde1ab05e2844100a8ba08`;
- final model: `public_q24_native_transport_energy_final_v1/FINAL_MODEL.pt`, SHA `21ad1dede3373c90d2694158b6599c4c172d7181c72634a6ec8e362449b7d1f8`;
- initial exported estimator: `public_q24_native_transport_energy_final_bank_v1/estimator.py`, SHA `5e4d4e7b661739c613012390fb38c8972c22d0f05d181f79a34caa7b09d7a16f`;
- exact parity PASS: `public_q24_native_transport_energy_final_bank_v1/PARITY_RESULT.json`, SHA `8914a42967da3309055101bf0d2ccefc76b2542378aac68dfa0757056543f2f6`;
- initial no-submit receipt after residual failure: `public_q24_native_transport_energy_final_bank_v1/HANDOFF_BLOCKED_RECEIPT.json`, SHA `b487e1f551c44cfc716a2d455585af066c3133b0e251647fb9feda2e94dbb95f`.

All paths in this section are relative to:

`pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/`

The user's later explicit retry/submit instruction supersedes only the no-retry operational stopping point; it does not authorize scientific retuning, target access, or unrelated submissions.

## 5. The remaining raw-MSE gap and the strongest causal evidence

Relative to the current board raw MSE `2.9409640056599073e-8`, reaching raw `<1e-8` requires a `2.941x` MSE reduction, removal of `65.9975%` of endpoint squared error, or an RMSE reduction from about `1.71492e-4` to below `1e-4`.

The strongest positive scientific evidence is the exposure-trained residual-state experiment:

- same-checkpoint teacher-forced raw MSE: `8.207963477637236e-9`, below target;
- autonomous raw MSE: `3.4709697955756664e-8`;
- validation host raw MSE: `9.165478898808242e-8`;
- receipt: `public_exposure_trained_residual_state_v1/RESULT.json`;
- SHA-256: `fe2f2f3c8cfa890d416b520f443fbbe475ea8fabafbd16956006a9154b100440`.

Interpretation: the lawful local innovation model already has enough capacity when the previous true residual state is supplied. The main unresolved bottleneck is autonomous reconstruction and propagation of that hidden residual state, especially preserving sign/alignment through depth. It is not simply a lack of local nonlinear features.

The latest rank-16 covariance result sharpens that conclusion. `P2R0F2N` achieved OOF `3.230119894372402e-8` versus frozen J `3.2334958229470984e-8`, a real but only `0.1044049%` gain. All four folds improved, yet `86.6938%` of paired row movement canceled and every inherited hard breach survived. Gain correlated `0.948` with correction-to-negative-error alignment but only `0.024` with correction amplitude. A truth-derived, parameter-free projection oracle onto N's learned rank-16 factor geometry reached only `2.8928320722187982e-8`, capturing `10.535%` of J endpoint error rather than the `69.076%` needed. Therefore rank scaling and score retuning on this axis are not licensed.

Evidence:

- `public_full688_rank16_cross_unit_covariance_cv_v1/CV_RESULTS.json`, SHA `37187da1fabdc84951a5bc68ef3f12fba1d4b940bc19027c8426f59f1b61be37`;
- `public_full688_rank16_cross_unit_covariance_cv_v1/VERIFICATION.json`, SHA `9b6a00d819eb219c8eada6242d96821a76abbfda2076e21f263d079113cb7645`;
- `public_full688_rank16_cross_unit_covariance_cv_v1/N_VS_J_OOF_DECOMPOSITION_AND_ONE_GLOBAL_ORACLE.json`, SHA `007ba020154fed4ca2dd63f16fd10b4961c85f4bd8f5b719cc131acafb53f82a`.

N's receipt contains one design-only, unimplemented proposal for a materially different axis: one full-rank nonlinear weight-space round trip,

`delta = R * a * W^T * tanh(b * (W * qhat) / sqrt(width)) / sqrt(width)`

with `qhat` derived only from J's lawful local features. This should not be launched directly. First freeze the exact normalization, parameterization of `a`/`b`, compute ledger, nested OOF training, and strict gates; then run exactly one arm. This is the current best graph-ready successor because it tests actual full-rank network geometry rather than another diagonal or low-rank covariance approximation.

## 6. Closed recent experiment families — do not repeat

These are terminal evidence, not ideas awaiting another seed or hyperparameter sweep:

| Family | Terminal raw result | Conclusion | Receipt and SHA-256 |
|---|---:|---|---|
| J native transport/energy/coherence, 688-row OOF | `3.2334958229470984e-8` | Best established autonomous OOF parent, but misses target and has a `3.4085e-7` hard row. | `public_full688_transport_energy_coherence_cv_v1/CV_RESULTS.json` — `70a0917258a2d656ab9593a67b3a04483af153d8ef59162be9570a6a62b9b5b4` |
| D dynamic four-state residual | `3.2969476163272226e-8` | Worse than J; no retune. | `public_full688_dynamic_multistate_residual_cv_v1/CV_RESULTS.json` — `ddaf1ede02733eb4429dcaf003ff65f8d9fba2460352e8deba4fb54a3c111dbf` |
| H state-independent Duhamel | `3.908100560611721e-8` | State-independent innovation is insufficient. | `public_full688_state_independent_duhamel_cv_v1/CV_RESULTS.json` — `55fd5ec3c50cbe70fad9c9f83e88625afc374cb5dd4e32261c1a1a1518727104` |
| I typed diagonal error variance | `3.5165106535091496e-8` | Diagonal uncertainty state is insufficient. | `public_full688_typed_diagonal_error_variance_cv_v1/CV_RESULTS.json` — `ab810134e83924b698eca0647f97a44ba3ef884653bb3fb3e6516862f00b445f` |
| K full-Gram pooled context | `3.2444190155738505e-8` | Global Gram/context does not materially beat J. | `public_full688_full_gram_pooled_context_cv_v1/CV_RESULTS.json` — `2a8a569599f3d9b8d0efe373a659367447dc652a0e24e9785a8d72db52abcd44` |
| M backward adjoint | `3.233766009119748e-8` | Regresses `0.00836%` versus J. | `public_full688_backward_adjoint_state_cv_v1/CV_RESULTS.json` — `1c8fc334b1ba20fba82a82ced63361c9ff42ad079bb9a7ec54a5b1c9d7e2ce7b`; decomposition SHA `8ee24497e1bd763ec1f4c001af4a0ae74b558c7ee888fddf351f2ad0249b7fdb` |
| N rank-16 cross-unit covariance | `3.230119894372402e-8` | Tiny alignment-limited gain; its own truth oracle remains `2.893e-8`; close rank scaling. | receipt set in Section 5 |
| O bounded D/J/I/K/M ensembles | best held `3.221703035370728e-8`; same-row per-unit oracle `3.1796003925380795e-8` | Candidate errors are too collinear; no output-space ensemble successor. | `public_oof_ensemble_screen_v1/RESULT.json` — `0bb7770e7d497ad9cd86e9293958f0a46a55fa424410a83c97c2ba03407e4484`; final receipt `91c2777288a5fe2a63ce836df035dd783d7c050cc3eda46611cca966f7e84d5c` |
| Gaussian secant transport | `3.4708951907076616e-8` vs tangent `3.4709697955756664e-8` | Only `0.002149%` improvement; mean-curvature mechanism closed. | `public_gaussian_secant_transport_discriminator_v1/RESULT.json` — `133e2d05b4f619c7f770e2cbdbf75d8399ce39a58338b47c8d2d99c0dfdc538a` |
| Cheap CP1536 recurrent H128 | `9.421320969930728e-7` at about `209.905B` FLOPs | Projected adjusted floor `9.421e-8`; not competitive. | `public_cp1536_h128_p4_recurrent_discriminator_v1/RESULTS.json` — `ccc1335cb65c3c66ed08c217d0c522b0a3ec26b8a93f2d5850603b9ca1dd29ca` |
| CP16384 + dynamic4 D3 | `7.429439793913384e-7` | Host/interface capacity failure; close spelling. | `public_cp16384_dynamic_multistate_pilot_v1/runs/protected160_40_v1/TERMINAL_RECOVERY.json` — `404bb2a4f1bb3eebb7f3d961bcad0aa098d1e4c866fcd3e139cec9138c42d792` |
| CP16384 + L1-seeded dynamic4 D4 | `7.35166326177063e-7` | Later transport capacity still insufficient; close spelling. | `public_cp16384_l1_seeded_dynamic_multistate_pilot_v1/TERMINAL_AUDIT.json` — `cbffd2482e220d76353e73d0a716a5bc6c28d0b5f96fd6285342189434ed487b` |

Also do not repeat generic scalar damping, clipping, extra trees, simple ridge, generic powers-of-W summaries, or another unconstrained ensemble sweep. Their branches are already closed in the canonical graph. A new experiment must test a materially new state object or a clearly different compute/accuracy Pareto architecture.

## 7. Solved numerical-instability incident

Do not reopen the old Q24 tail blow-up unless new exact evidence contradicts this receipt chain.

Root cause: the correlation-gauge covariance repair normalized by a requested variance that could hit or nearly hit `1e-30`. A nearly dead coordinate was thereby amplified into a dense covariance perturbation and produced late-layer blow-ups.

Surgical repair trigger:

```python
if float(fnp.min(variance)) <= 2.0e-30:
```

On that branch the implementation uses a raw-covariance diagonal load followed by congruence rescaling. Repaired source:

`fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py`

SHA-256 `26aca6ed1fb52ec431132f87739fe729ec6855cc2fa2c80642fadd1c4c786e5d`.

The terminal 688-row observer_fit scan found 11 affected role ranks:

`[55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]`

All 11 repaired rollouts were finite and below the predeclared `5e-7` tail threshold. Evidence:

`public_observer_fit_candidate_truth_score_v1/PUBLIC_SCAN_RESULT_688.json`

SHA-256 `cde60657ada853c13802fb1f241d6da205708f91b2ea552481445b5679bec2cf`.

No public-name join was used.

## 8. Non-negotiable data and scientific firewalls

- `observer_fit` targets are authorized for development under their recorded receipts.
- `observer_selection` remains unopened for these runs. Do not open it.
- Do not join public board names, identities, scores, or aliases to development rows.
- Preserve target-first split semantics and exact `mlp_id` / `role_rank` alignment.
- Deployment may not consume truth-derived inputs.
- A same-row fit/oracle is capacity analysis only, never protected performance.
- Freeze hypotheses, seeds, split law, checkpoint law, cost ledger, and gates before target metrics.
- No checkpoint selection on validation metrics unless a new, explicitly authorized protocol permits it.
- Do not use private/held board identities or the sealed split.
- Preserve the dirty worktree. Existing edits belong to the user or another agent; never reset or erase them.
- Use `apply_patch` for file edits.
- Submission or other external mutation requires recorded user authority. The existing authority covers the one H7B recovery submission described above, not future candidates.

## 9. Canonical problem graph

Canonical artifacts:

- `pure_closure_writeup/problem_graph/graph.json`
- `pure_closure_writeup/problem_graph/problem-graph.html`
- `pure_closure_writeup/problem_graph/problem-graph.md`
- reconciler: `pure_closure_writeup/problem_graph/reconcile_20260824_postboard.py`

The final post-board canonical state has:

- graph SHA-256 `32d04b3c3f9493b8f261ff5dd2ad13048674ffe9c5a46b6ee399c9b4804045bb`;
- HTML SHA-256 `282fa6cb0969bc04aebe79aa89148384ac485f01c27f665203ea5dc27c53b4bc`;
- Markdown SHA-256 `aff3f94e8f55b28b65d43a9a0ae44d383195296df08c1ac6a6c0bef977013a98`;
- reconciler SHA-256 `aa8f222eb35e79d514f0f7253132abae8aa6d44cb94267480b5dda3d96a0f060`;
- renderer SHA-256 `29190a0c41fa8ea1cabbe1daad0abfd17d6a9fd936e572d596a5cba8c2d88f09`;
- 1,072 nodes, 1,325 edges, 207 gates, and 399 leaves;
- zero active agents/workstreams in `active_work`;
- one open leaf, `P2R0F2P`.

The canonical graph closes N as measured, binds its terminal decomposition, binds H7B's meter/package/board result, and exposes `P2R0F2P` as the sole licensed successor. All validators passed with the final hashes above. Re-run this complete sequence before relying on the graph after any future mutation:

```bash
python3 pure_closure_writeup/problem_graph/reconcile_20260824_postboard.py
python3 pure_closure_writeup/problem_graph/render_graph.py
python3 pure_closure_writeup/problem_graph/validate_graph.py
python3 pure_closure_writeup/problem_graph/validate_leaf_completeness.py
python3 pure_closure_writeup/problem_graph/validate_layout.py
python3 pure_closure_writeup/problem_graph/validate_reconciliation.py
python3 pure_closure_writeup/problem_graph/validate_evaluation_split.py
python3 pure_closure_writeup/problem_graph/validate_fable_coverage.py
```

Final validator summary: graph structure `OK` at `1072/1325`; 399 leaves with exactly one open; zero live graph workstreams; layout partitions all nodes into the three non-overlapping phase boxes; reconciliation is current and idempotent; evaluation split validates 64 unique seeds at digest `87eca84751643fac517a9f25c2b9e334bf076871a750f2caee20ac170ccd42d9`; Fable coverage validates 22 decision-relevant families, 11 explicit supersessions, and 125 mapped graph nodes. The HTML is up to date with the final graph.

## 10. Recommended continuation after the clean stop

First verify this handoff against disk; do not trust prose over an exact receipt. Confirm no inherited process or watcher is live and confirm the graph/HTML/Markdown hashes and validators.

Then continue the highest-ceiling unresolved path toward autonomous hidden-residual reconstruction. The most evidence-backed next node is the single design-only full-rank nonlinear `W/W^T` round-trip operator from N's decomposition. Before any target access:

1. work within sole open leaf `P2R0F2P`, whose claim is explicitly different from rank/diagonal covariance and backward-adjoint families;
2. specify exactly how `qhat` is constructed from lawful J inputs;
3. freeze dimensionless normalization, scalar or low-count parameterization of `a` and `b`, initializations, seed, optimizer, epoch/checkpoint law, nested grouped fourfold OOF split, and no-retune gate;
4. derive and independently audit a conservative exact compute ledger under `2^41`, including the native Q24 host and correction;
5. require all 688 autonomous OOF rollouts finite, OOF raw `<1e-8`, every fold `<1.1e-8`, every network `<1e-7`, zero observer_selection access, zero identity joins;
6. run one arm only, verify independently, close it on any failed conjunct, and update all graph views immediately.

In parallel at the design stage, consider compute reduction only if it is a genuinely material host replacement or Pareto hybrid. Small savings on the current `~2.132e12`-F host cannot turn a `~3e-8` raw method into adjusted `<1e-8`; a raw `3e-8` method needs roughly `<=7.33e11` constant FLOPs. Any mixed proposal must show both an accuracy mechanism and enough cost movement to cross the exact per-row score law.

Do not begin by retuning J, increasing N's rank, adding another output ensemble, or repeating a closed cheap-host transfer.

## 11. Goal Loop

Token budget: unset; there is no explicit cap.

```text
Continue the K21 / pure-closure Phase-2 estimator research from FABLE_HANDOFF_20260825.md until either (a) a legal deterministic submission achieves raw final-layer MSE <1e-8, preferably, or (b) a legal accuracy/compute combination achieves adjusted score <1e-8; preserve every recorded data/science firewall, close or decompose each tested leaf with exact receipts, keep graph.json/problem-graph.html/problem-graph.md reconciled after every terminal result, and never submit or otherwise mutate external state without fresh user authority.
```

## 12. Exact startup prompt for a fresh no-context Fable session

Use this prompt exactly:

```text
Continue the active K21 / pure-closure Phase-2 research in /Users/matt/arc-white-box-estimation. You have no prior conversational context. Before taking any action, read /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/FABLE_HANDOFF_20260825.md completely, then inspect the canonical problem graph and every receipt that the handoff marks as required. Treat exact receipts as authoritative over prose. The handoff records terminal H7B submission #328326 and the final canonical graph hashes; verify both, verify that no inherited process or watcher is live, and verify that graph.json, problem-graph.html, and problem-graph.md reconcile and pass every listed validator.

Preserve the dirty worktree and use apply_patch for edits. Do not open observer_selection, do not join public names/identities/scores to development rows, do not use deployment truth inputs, and do not repeat any experiment family the handoff marks closed. Do not submit or mutate external state without fresh recorded authority. Continue the highest-ceiling unresolved path toward raw final-layer MSE <1e-8, while retaining an accuracy/compute path to adjusted <1e-8 under the exact per-row law. The current evidence says the central bottleneck is autonomous hidden-residual state reconstruction: teacher forcing reaches 8.207963477637236e-9, while autonomous methods remain near 3e-8. The sole open graph leaf is P2R0F2P, the design-only full-rank nonlinear W/W^T round-trip residual operator. Start by auditing and preregistering that exact node—without target access—using the N-vs-J decomposition as its authority. Run only one leakage-safe arm after its hypothesis, normalization, parameterization, split, checkpoint, resource, and strict no-retune gates are frozen and independently checked. Close or decompose every terminal leaf and update graph.json, problem-graph.html, and problem-graph.md after every terminal result. Give concise updates at meaningful milestones, not per row.
```

## 13. Receipt audit notes

Every completed-artifact SHA cited above was recomputed locally. The H7B v4 source, parity, three-call meter, archive, package validation, bank-ready receipt, terminal board receipt, reconciler, renderer, graph, HTML, and Markdown artifacts all exist and agree. All eight graph commands in Section 9 passed on the final state. There are no unresolved placeholder tokens in this handoff.
