# Fixed whole-pair projection: qualified execution, failed score objective

September 7, 2026 PDT. Phase 2 only. Preserve all source, synthetic tests,
acquisitions, predictions, bills and older closeouts. No tuning or refit.

## Measured verdict

The prospective actual-score pilot FAILED on BOTH development networks.
These are stored1e9 development references, not the independent14 panel
and not official scores. All values below are from the SAME unwrapped
configuration for raw and complete per-call bill, with initialization once
equally. The repeated35 call is excluded from both two-net means.

| Network | Parent raw MSE | Projected raw MSE | Parent complete FLOPs | Projected complete FLOPs | Parent adjusted | Projected adjusted |
|---|---:|---:|---:|---:|---:|---:|
|35 cold|2.1099028082828634e-8|1.1133888419226767e-7|476670806147|384352970841|4.573526314361569e-9|1.9460199341401736e-8|
|100|1.997488978344223e-8|1.3774019320118412e-7|473227916215|384826542393|4.298579128245664e-9|2.4104375505956108e-8|
|Mean|2.053695893313543e-8|1.2453953869672589e-7|474949361181|384589756617|4.436052721303617e-9|2.1782287423678924e-8|

Actual complete compute saving:90359604564 FLOPs,19.025102873980828%.
Mean raw ratio:6.064166515704869. Correct mean adjusted ratio:
4.910285966411552; per-net adjusted ratios4.254966081706745 and
5.607521645366008. Adjusted is the mean of per-net
MSE*max(.1,F/2^41), NOT the product of mean raw and mean cost ratios.
The advance gate was both distinct nets improving and at least1% mean
adjusted improvement. It was not met. No independent14 or release run.

The actual cost-saving hypothesis held approximately; its raw-neutrality
assumption did not. Error first changes at layer8 as expected afterpost7.
This does not identify the mechanism of raw degradation or measure a
full-size coefficient/reference tensor error. It does not prove that all
global-kernel compression methods are impossible.

## Compatibility, numerical and resource evidence

Four fresh sequential nice10 OFFPY processes each served35/100/35:
parent_nullref, null, parent, candidate. No concurrent numerical worker,
MPS job, warmup, payload reset or refit. All three BLAS launch variables1.
Threadpool listings were empty: this attests explicit launch controls,
not independently enumerated Accelerate thread counts.

- Null insertion matches all16 outputs, pre-head30-feature arrays, returned
  head deltas and full per-call FLOP bills exactly on all3 calls.
- Unwrapped parent matches its diagnostic reference outputs and bills
  exactly. Diagnostic copies were checked to add zero metered FLOPs.
- Enabled candidate matches parent output bytes on layers0..7 and has
  byte-exact cold/repeated35 predictions. All16 served outputs are finite,
  nonnegative concretefloat32(16,1024), with materialization inside budget.
- Cold-minus-repeat initialization charge is1383913666 FLOPs in both
  actual score roles. No initialization cost is omitted or amortized away.
- Candidate native calls18.1539204167--18.3440241674s; maximum cumulative
  RSS5163843584bytes. All four roles passed120s/8GiB native gates.
- Candidate native residual.73344--.76754s is diagnostic, above the later
  production.4s threshold in this different native frame. No persistent
  NumPy-free, residual, officialLinux, contract or packaging claim follows.

The tiny original-cube certificate and exact source inverse had already
passed before any P2 execution. They established implementation identities,
not actual score neutrality. Candidate source retained all16 response P
heads and every other original payload/reader node unchanged.

Final acquisition completed10:06:27.153678UTC. All-role verification seal
was written10:06:46.173889UTC; truth access began10:06:46.178731UTC.
An independent source/JSON-only audit rechecked all small seals, role
ordering, parity metadata, bills, arithmetic and this chronology.

## Frozen identities

- Candidate candidate_whole_pair_kernel_post7_k512_v1.py:
  41663c0b58e313bc89a13a425c2e22e29a0e9b2bde740580a2dd846d65eb1b97.
- Parent candidate_response_rows_current_runtime_v1.py:
  5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172.
- Checker check_whole_pair_kernel_fixed_dev_v1.py:
  de67d33312fbacc501c0cb73cd791d677b0c52dd9ba0bc6c1e468ef8ea223b93.
- Root admission WHOLE_PAIR_KERNEL_ROOT_ADMISSION_V1.md:
  29845e60b14fb168842ecec10baf121ec1ae014da7b12ed0b482de29f41abf92.
- whole_pair_kernel_fixed_dev_v1/score/result.json:
  da27ddecf4d4fdda120f82d22896042e20721e5a5fa48b7517ebf4911ba07b05.
- parent_nullref/result.json:
  a2570aee398eea4139cd2f6938e9d0f935514f4db2e88498527bbb4fc5c18521.
- null/result.json:
  5d23a43df7dd954ba53d6e1e49c936f28f2e8ba8121d93877ae0803745d66279.
- parent/result.json:
  a7d78e74b157f628d0d7881d552c22d0cf0a2085e05fb60e7b1f8ae653322c59.
- candidate/result.json:
  5b7b4d6c33bab5657b9d849237bc92a1d19b10e55b93a512541ae9f754c8eddc.

## Narrow closeout and next decision

Close this exact post7/cohorts0..6/512 complete-pair energy-selected,
FP32cross-RHS/FP64selected-Gram/fixed1e-3ridge/additive-star-repair,
unchanged-response-head construction. No automatic rank, cut, ridge,
precision, selection, head, training or repair sweep follows this verdict.
No panel, upload, forum post, commit, promotion, deletion or frozen edit.

Last confirmed official bank remains#329926 at4.982231460183304e-9.
Retained response component remains4.83424098592912e-9 on independent14;
it is preserved, not submitted above the below4e-9 milestone. Final goal
strictly below1e-9, with validated banks below4e-9/3e-9/2e-9, is unchanged.

The separate raw route has a newly identified NARROW old-bank rank35
mean13 reference: GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md.
That is source/provenance progress only. No cycle array test or numerical
admission occurred; complete retained-trajectory accuracy/cost evidence
is still missing. Reassess value before another substantial candidate run.
Post-test resource check10:07:48UTC: no Python/Whest workers; swap792.75MiB
of2048MiB, unchanged from preflight.
