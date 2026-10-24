# Retained small validated component: frozen response-row reduction

September6,2026 PDT. The one explicitly authorized no-refit adjusted-score
exception is complete. Preserve this component and all its predecessors.
Do not reinterpret this as permission to tune/refit response rows or reopen
any other closed family. The old raw-only closeout remains unchanged.

## Confirmed outcome and decision

The exact fixed candidate improves BOTH independent panel groups. Numbers
below use the same14 MLPs, stored independent4M truths, servedfloat32
outputs, native FlopScope0.12/Whest0.16, and complete same-configuration
per-net bills. Noise-corrected raw subtracts mean(mc_se15^2), without
clipping; adjusted is that per-net raw times max(.1,F/2^41), then averaged.

| Group | Current bank adjusted | Response adjusted | Improvement |
|---|---:|---:|---:|
| Public8 |5.437274909258601e-9|5.2532154360922335e-9|3.38514%|
| Virgin6 |4.409194462222713e-9|4.275608385711634e-9|3.02972%|
| Combined14 |4.996669003386079e-9|4.83424098592912e-9|3.25073%|

Combined mean noise-corrected raw:
bank2.2592877636382216e-8, response2.2454337457623213e-8 (0.61320% lower).
Mean complete FLOPs: bank485374855176.7857,
response472413324017.2143 (2.67042% lower). Each role is one fresh
process serving14 MLPs sequentially, so first-call initialization is
included once identically. Older independently cold-initialized panel
numbers are not silently substituted for this matched baseline.

Panel result: response_rows_panel_v1/score/result.json, SHA256
a31981681305de1828c48d8b6e8830c87ed979789c7d600b949217814f479cdd.
Response acquisition resultSHA
8b8433a13d2448f785f468cb67d2f793745495ffc632d76408e5a0e4f33e2de4;
bank acquisition resultSHA
aea8a0f265d29dd923ea032064df86f796f9a99b0b363d5e604f761e71d22f29.
Both complete output/bill sets were authenticated before any panel truth
access. Both native launch records explicitly pin all three BLAS controls
to1. Score-only launch emitted a denied-nice warning in the restricted
shell, then completed exit0; it performed no estimator/card execution.
All actual card acquisitions ran at recorded nice10.

Development35/100 separately improved4.603252607224289e-9 to
4.436052721303617e-9 (3.63221%) against their stored1e9 truths. This is
corroborating development evidence, not another independent panel. See
RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md for complete raw/bill/parity
receipts and the native diagnostic thread-inventory qualification.

**Retain as a component; do not submit.** Its local4.834e-9 panel value
does not cross the below4e-9 milestone. Official bank#329926 is unchanged
at4.982231460183304e-9. No official score can be inferred from this panel.
Strict goals remain below4e-9,3e-9,2e-9 banks and final below1e-9.

## Faithful model/runtime and numerical checks

Candidate: candidate_response_rows_current_runtime_v1.py, SHA256
5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172.
Frozen response P controls all16 full-feature hooks; Q is unreachable.
The full final head is retained with no outgoing layer15 birth. Threshold
.01, every CP factor, every marginal diagonal and active covariance/star
coordinate remain fixed. The omitted radial source is restored before
learned feedback. D4/bulk-copy runtime is unchanged.93 other top-level
source nodes were byte-identical to the bank; payload serialization is
byte-faithful. No training, parameter edits, threshold search or refitting.

Full16-layer archived output/head parity passed on35/100; repeated35 is
byte-identical. The separate unwrapped NumPy-free persistent test on
35/100/500/3016500/35 also passes: finite/nonnegative properfloat32
outputs, repeat byte identity, maximum native-vs-remote output difference
3.09944e-6, full budget, wall33.985--34.882s, residual<=.144213s,
peak combined sampled RSS7240744960bytes (<8e9).
Qualified receiptSHA
de480339927c292bad2c6d8d30af459cd15d6052496bc87c6209104981c4ed9c.
These are local macOS checks, not an official Linux guarantee.

The first remote attempt's generic100MiB server-allocation rejection is
preserved. The successful v2 uses the previously documented official4GiB
per-array setting without changing estimator/accounting, as recorded in
RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md. The total-memory and wall gates
were not relaxed. Native14-call peak RSS was7884537856bytes response and
7640449024bytes bank, both below the predeclared8GiB native gate.

## Contract and frozen unsubmitted package

Whest0.16.0 `validate --json` completed exit0 with ok=true and
output_shape=[2,4]. This is the generic shape/guard-path contract check,
NOT a full-size numerical or runtime validation. Full1024x16 evidence is
the separate native/remote/panel work above.

Whest0.16.0 package completed exit0, ok=true, creating only the new local
response_rows_current_runtime_5a07832f_whest016.tar.gz (755010bytes).
ArchiveSHA256
4846fee44edd58ae734b3f35fa7c2a106ef6a8b8cfd9f19ccdcdb69471291a3d.
Read-only archive inspection found exactly:

- estimator.py,1811146bytes, SHA identical to the validated source above.
- manifest.json,528bytes, SHA
  475e59aeeb3bcd69016b68fc07211d7f9327aeac6264fa9751706f762d448bbe.

No key, labels, unrelated files or numerical library is bundled. No upload,
forum post, email, git commit, promotion, deletion or frozen-file overwrite.
Final resource check: no Python/Whest workers; swap792.75MiB of2048MiB.
Future combinations must preserve/coadapt their own full state/head
compatibility and revalidate the actual combined raw-times-cost objective.
The gain cannot be added arithmetically to another arm's percentage.
