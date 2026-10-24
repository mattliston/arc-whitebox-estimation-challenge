# Submission #337422: Message width 8 with teacher compression and loader repair

2026-10-20T23:43:33Z · submission · adopted · measured

Message width was reduced to eight using teacher compression and selective loader restoration. The graded result marginally improved the preceding public best.

Recorded measurement:

    Official submission record: {"submission_id": 337422, "date": "2026-10-20T23:43:33Z", "status": "GRADED", "score_public": 1.4226494934284554e-09, "n": 100, "failed_networks": 0, "utilization": 0.10834182489733393, "wall_mean_s": 101.347, "wall_max_s": 112.229, "resid_mean_s": 0.36, "resid_max_s": 0.385, "source_sha256": "e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05", "archive_sha256": "8b1d3c1e68810fac6c41723bbc9b9f9205b34a115a29939ea49f31a290a0bd82"}
    Ledger: #337421 GRADED/completed23:43:02Z (4:43PM Pacific),100/100complete,zero failures,all5setupsSUCCESS. Exactpublicscore1.423137754971646e-9, -.2869915256%versus337393. J2W1.416013781559781e-9 nowneeds-.5005821388%versus337421. K/BEST_GRADED_REFERENCE_v1.json atomicallyupdatedaftercheckingnoactiveoriginalparallel5/pinnedpriorityjudge; oldreferencepreservedK/BEST_GRADED_REFERENCE_337393_v1.json. SciencebaseD/science32_eh14_flatload_revert_parallel5_v1, SCI9af5617c, RELc7b39747. EvidenceA/lead_audit_2323_v1/public_monitor_20261020T234528Z/eval_337421_raw.json with100actualcompletionrecords. Allnewadjustednumbersmustuseactualpaired337421reference; historical337393reportsremainlabeledhistorical. VM exact337421samehostcontrolneedsdispatch, notinferredloaderoffset. #337422stillgrading,fiveinitialsetupsSUCCESS; nofinalscoreyet. LASTFIREDstill337422/8b1d3c1e. Matt's>=1%nextbanktargetunchanged despitegapnow.501%.
    #337422 finalGRADED100/100/zero failed; all5setupSUCCESS max4.301609s. Score1.4226494934284554e-9, actual-.0343088040% versus337421, NOT the local-.4284% projection. Public score50networkaccuracy versus100networkruntime distinction must be preserved. User correctly observes nearlysame score. J2W1.416013781559781e-9 requires-.4664333625%more. BEST pointer atomicallypromoted00:22:43Z afteredge6originaljudgecompleted00:20:50Z; backedup421pointer. ExactfinalevidenceA/lead_audit_2323_v1/public_monitor_20261021T002129Z/eval_337422_raw.json SHA5345808663a5b6794d55f2b16d7753106736101bebd13be2898f4a2e0da591d7. SCI777a73b6/RELe41095e4; LASTFIRED remains337422/8b1d3c1e. AllNEWgainsvs422; preservehistoricalreports. User: biggergainbeforenextfire, keepVMhumming;10UTC21slotsunused. No fire authorizedforcurrentedgealone.

Public score: 1.4226494934284554e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../337393/monitor_public_submission_v2.py](../337393/monitor_public_submission_v2.py)
- [../337393/audit_public_transfer_337422_v2.py](../337393/audit_public_transfer_337422_v2.py)
- [../337393/report_public_envelope_337422_v1.py](../337393/report_public_envelope_337422_v1.py)
- [../337393/report_public_runtime_v1.py](../337393/report_public_runtime_v1.py)
- [../337393/audit_public_transfer_337422_v1.py](../337393/audit_public_transfer_337422_v1.py)
- [../337393/report_public_envelope_337422_v2.py](../337393/report_public_envelope_337422_v2.py)
- [../337393/monitor_public_v1.py](../337393/monitor_public_v1.py)
- [../../instruments/single-submission-tool-records-one-archive-polls-evaluation-receipt/code/read_eval_summary_astra_v1.py](../../instruments/single-submission-tool-records-one-archive-polls-evaluation-receipt/code/read_eval_summary_astra_v1.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1253](../../records/SUBMISSION_LEDGER.md#L1253)
- [SUBMISSION_LEDGER.md#L1274](../../records/SUBMISSION_LEDGER.md#L1274)

<details>
<summary>All result files and records (90)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [package.json](package.json) · result
- [validate.json](validate.json) · result
- [validate_package.json](validate_package.json) · result
- [stage_a.json](stage_a.json) · result
- [CURRENT_SOURCE7_v1.json](../337393/CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](../337393/VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](../337393/PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](../337393/PLAN_v1.json) · result
- [RESULT32_v1.json](../337393/RESULT32_v1.json) · result
- [PUBLIC_337466_WATCHER_LAUNCH_v2.json](../337393/PUBLIC_337466_WATCHER_LAUNCH_v2.json) · result
- [ROOT_BEST337466_PROMOTION_v1.json](../337393/ROOT_BEST337466_PROMOTION_v1.json) · result
- [ROOT_337473_FAILURE_REVIEW_v1.json](../337393/ROOT_337473_FAILURE_REVIEW_v1.json) · result
- [FIRST_FINAL10_NEAR_TIE_AUTHORIZATION_v1.json](../337393/FIRST_FINAL10_NEAR_TIE_AUTHORIZATION_v1.json) · result
- [FINAL_337421_EVIDENCE_v1.json](../337393/FINAL_337421_EVIDENCE_v1.json) · result
- [PUBLIC337473_FINAL_RUNTIME_PAIRS_VS466_v1.json](../337393/PUBLIC337473_FINAL_RUNTIME_PAIRS_VS466_v1.json) · result
- [LAPTOP_FINISHED_QUEUE_AUDIT_20261020T233938Z.json](../337393/LAPTOP_FINISHED_QUEUE_AUDIT_20261020T233938Z.json) · result
- [HOSTED_SDK_VERSION_PARITY_SEAL_v1.json](../337393/HOSTED_SDK_VERSION_PARITY_SEAL_v1.json) · result
- [PUBLIC_337422_FULL100_RUNTIME_ENVELOPE_v1.json](../337393/PUBLIC_337422_FULL100_RUNTIME_ENVELOPE_v1.json) · result
- [PUBLIC_STANDINGS_20261021T0151_v1.json](../337393/PUBLIC_STANDINGS_20261021T0151_v1.json) · result
- [HOSTED337466_337473_RUNTIME_VERSIONS_v1.json](../337393/HOSTED337466_337473_RUNTIME_VERSIONS_v1.json) · result
- [RUNTIME_337422_VS_VALID337421_20261020T235314Z.json](../337393/RUNTIME_337422_VS_VALID337421_20261020T235314Z.json) · result
- [RUNTIME_337422_VS_VALID337421_20261021T003634Z.json](../337393/RUNTIME_337422_VS_VALID337421_20261021T003634Z.json) · result
- [FINAL_337422_EVIDENCE_v1.json](../337393/FINAL_337422_EVIDENCE_v1.json) · result
- [PUBLIC337473_FAILURE_INDEX27_v1.json](../337393/PUBLIC337473_FAILURE_INDEX27_v1.json) · result
- [NEXT_FIRE_ONE_PERCENT_TARGET_v2.json](../337393/NEXT_FIRE_ONE_PERCENT_TARGET_v2.json) · result
- [PUBLIC337466_RUNTIME_20261021T042835Z.json](../337393/PUBLIC337466_RUNTIME_20261021T042835Z.json) · result
- [LEAD_AUDIT_REFRESH_v1.json](../337393/LEAD_AUDIT_REFRESH_v1.json) · result
- [PUBLIC337473_PARTIAL_RUNTIME_20261021T053501Z.json](../337393/PUBLIC337473_PARTIAL_RUNTIME_20261021T053501Z.json) · result
- [PUBLIC337466_FINAL_PAIRED_VS422_v1.json](../337393/PUBLIC337466_FINAL_PAIRED_VS422_v1.json) · result
- [RUNTIME_337422_VS_VALID337421_20261021T001309Z.json](../337393/RUNTIME_337422_VS_VALID337421_20261021T001309Z.json) · result
- [RUNTIME_337422_VS_VALID337421_20261021T000355Z.json](../337393/RUNTIME_337422_VS_VALID337421_20261021T000355Z.json) · result
- [PUBLIC_STANDINGS_20261021T0308_v1.json](../337393/PUBLIC_STANDINGS_20261021T0308_v1.json) · result
- [PUBLIC_STANDINGS_20261021T0354_v1.json](../337393/PUBLIC_STANDINGS_20261021T0354_v1.json) · result
- [NEXT_FIRE_AT_LEAST_ONE_PERCENT_v1.json](../337393/NEXT_FIRE_AT_LEAST_ONE_PERCENT_v1.json) · result
- [PUBLIC_337422_TRANSFER_COMPOSITION_v1.json](../337393/PUBLIC_337422_TRANSFER_COMPOSITION_v1.json) · result
- [PUBLIC_337466_WATCHER_LAUNCH_v1.json](../337393/PUBLIC_337466_WATCHER_LAUNCH_v1.json) · result
- [PUBLIC337473_WATCH_LAUNCH_v1.json](../337393/PUBLIC337473_WATCH_LAUNCH_v1.json) · result
- [eval_337422_raw.json](eval_337422_raw.json) · result
- [eval_337422_raw-e984a614.json](eval_337422_raw-e984a614.json) · result
- [eval_337422_raw-22b2dbd0.json](eval_337422_raw-22b2dbd0.json) · result
- [eval_337422_raw-22faa352.json](eval_337422_raw-22faa352.json) · result
- [eval_337422_raw-ab9dc58a.json](eval_337422_raw-ab9dc58a.json) · result
- [eval_337422_raw-03bb062a.json](eval_337422_raw-03bb062a.json) · result
- [eval_337422_raw-bb45aba5.json](eval_337422_raw-bb45aba5.json) · result
- [eval_337422_raw-baac2d08.json](eval_337422_raw-baac2d08.json) · result
- [eval_337422_raw-6a56f067.json](eval_337422_raw-6a56f067.json) · result
- [eval_337422_raw-e4238f51.json](eval_337422_raw-e4238f51.json) · result
- [eval_337422_raw-d29f61ad.json](eval_337422_raw-d29f61ad.json) · result
- [eval_337422_raw-1eeface1.json](eval_337422_raw-1eeface1.json) · result
- [eval_337422_raw-26f39751.json](eval_337422_raw-26f39751.json) · result
- [eval_337422_raw-a2884ad2.json](eval_337422_raw-a2884ad2.json) · result
- [eval_337422_raw-9966e980.json](eval_337422_raw-9966e980.json) · result
- [eval_337422_raw-7e87c722.json](eval_337422_raw-7e87c722.json) · result
- [eval_337422_raw-2e24b443.json](eval_337422_raw-2e24b443.json) · result
- [eval_337422_raw-2a78ff03.json](eval_337422_raw-2a78ff03.json) · result
- [eval_337422_raw-a9728da2.json](eval_337422_raw-a9728da2.json) · result
- [eval_337422_raw-f1b14b1c.json](eval_337422_raw-f1b14b1c.json) · result
- [eval_337422_raw-0f61cc24.json](eval_337422_raw-0f61cc24.json) · result
- [eval_337422_raw-190c8219.json](eval_337422_raw-190c8219.json) · result
- [eval_337422_raw-a5847907.json](eval_337422_raw-a5847907.json) · result
- [eval_337422_raw-1cee0259.json](eval_337422_raw-1cee0259.json) · result
- [eval_337422_raw-f2b67635.json](eval_337422_raw-f2b67635.json) · result
- [eval_337422_raw-5939ecd3.json](eval_337422_raw-5939ecd3.json) · result
- [eval_337422_raw-e7e8fae4.json](eval_337422_raw-e7e8fae4.json) · result
- [eval_337422_raw-5e91f92c.json](eval_337422_raw-5e91f92c.json) · result
- [eval_337422_raw-426d6f46.json](eval_337422_raw-426d6f46.json) · result
- [eval_337422_raw-278acd63.json](eval_337422_raw-278acd63.json) · result
- [eval_337422_raw-7137826c.json](eval_337422_raw-7137826c.json) · result
- [eval_337422_raw-60015b7d.json](eval_337422_raw-60015b7d.json) · result
- [eval_337422_raw-9c3be640.json](eval_337422_raw-9c3be640.json) · result
- [eval_337422_raw-d5304ecc.json](eval_337422_raw-d5304ecc.json) · result
- [eval_337422_raw-ed8268b2.json](eval_337422_raw-ed8268b2.json) · result
- [eval_337422_raw-b2f002cb.json](eval_337422_raw-b2f002cb.json) · result
- [eval_337422_raw-4e4e3c71.json](eval_337422_raw-4e4e3c71.json) · result
- [eval_337422_raw-8e50c974.json](eval_337422_raw-8e50c974.json) · result
- [eval_337422_raw-0e1fe214.json](eval_337422_raw-0e1fe214.json) · result
- [eval_337422_raw-708959b3.json](eval_337422_raw-708959b3.json) · result
- [eval_337422_raw-5064f71b.json](eval_337422_raw-5064f71b.json) · result
- [eval_337422_raw-a6998979.json](eval_337422_raw-a6998979.json) · result
- [eval_337422_raw-d5f74aec.json](eval_337422_raw-d5f74aec.json) · result
- [eval_337422_raw-dbcd7a64.json](eval_337422_raw-dbcd7a64.json) · result
- [eval_337422_raw-e5c821ba.json](eval_337422_raw-e5c821ba.json) · result
- [eval_337422_raw-954666e3.json](eval_337422_raw-954666e3.json) · result
- [eval_337422_raw-b7afa39f.json](eval_337422_raw-b7afa39f.json) · result
- [eval_337422_raw-53458086.json](eval_337422_raw-53458086.json) · result
- [eval_337422_raw-5fe6dab9.json](eval_337422_raw-5fe6dab9.json) · result

</details>

## Related work

- ← shipped in: [Eight-channel message distillation with loader repair: -.6774163425% Mac32 versus #337393](../../experiments/learned-correction/eight-channel-message-distillation-loader-repair-6774163425-mac32-versus/README.md)

[Research graph](../../graph/index.html#node=submission-337422) · [Copy provenance](../../PROVENANCE.json)

