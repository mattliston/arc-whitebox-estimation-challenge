# Historical release package preserves 109 files with per-file checksums and estimator dependencies

2026-10-22T00:44:46Z · tool · informational · measured

The historical package contains both designated sources, artifacts, grade and gate records, parent files, capture/fit/export code, coefficients and dependency documentation. The record reports byte-copy verification and a clean secret scan; the recorded MIT license choice still required owner confirmation.

Recorded measurement:

    - 2026-10-22T00:44:46Z [fable-lead] RELEASE PACKAGE: phase-two-release/ (109 files, 40 MB; 337641/, 337619/, common/; byte-identical copies verified per file, originals untouched; SHA256SUMS self-test PASS; secret scan clean) committed as 15b06285 on backup/main-pre-sync-20260902 and pushed to origin as BOTH refs/heads/phase-two-release and refs/heads/backup/main-pre-sync-20260902 (github.com/mattliston/arc-whitebox-estimation). Contents per the Rules s8 open-source requirement: artifact, estimator.py, manifest, packaging+gate records, hosted grade, build record, parent seal, readout capture/fit/export code + coefficients, parents (#337506 lineage), tails, write-up draft, DEPENDENCY_MANIFEST, MIT LICENSE (Matt to confirm the license choice).

## Programs

- [../../../release/337619/build/candidate_hgfh_r0p03_s15_cap4s8e9_clean_science_v1.py](../../../release/337619/build/candidate_hgfh_r0p03_s15_cap4s8e9_clean_science_v1.py)
- [../../../release/337619/build/tail_sm40_cap4_size8e9_xsg_off.py](../../../release/337619/build/tail_sm40_cap4_size8e9_xsg_off.py)
- [../../../release/337619/extracted/estimator.py](../../../release/337619/extracted/estimator.py)
- [../../../release/337619/seal_s1.5/candidate_first128_onpolicy_blocks_release_v1.py](../../../release/337619/seal_s1.5/candidate_first128_onpolicy_blocks_release_v1.py)
- [../../../release/337619/seal_s1.5/candidate_first128_onpolicy_blocks_science_v1.py](../../../release/337619/seal_s1.5/candidate_first128_onpolicy_blocks_science_v1.py)
- [../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_release_v1_before_govstrip.py](../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_release_v1_before_govstrip.py)
- [../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_science_v1.py](../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_science_v1.py)
- [../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_science_v1_before_govstrip.py](../../../release/337641/build/candidate_hgfh_s135_c4s8e9_clean_science_v1_before_govstrip.py)
- [../../../release/337641/build/tail_sm40_cap4_size8e9_xsg_off.py](../../../release/337641/build/tail_sm40_cap4_size8e9_xsg_off.py)
- [../../../release/337641/extracted/estimator.py](../../../release/337641/extracted/estimator.py)
- [../../../release/337641/seal_s1.35/candidate_first128_onpolicy_blocks_release_v1.py](../../../release/337641/seal_s1.35/candidate_first128_onpolicy_blocks_release_v1.py)
- [../../../release/337641/seal_s1.35/candidate_first128_onpolicy_blocks_science_v1.py](../../../release/337641/seal_s1.35/candidate_first128_onpolicy_blocks_science_v1.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [../../../release/common/parents/parent_release_f9381d09_first128_paid_nocensus_estimator.py](../../../release/common/parents/parent_release_f9381d09_first128_paid_nocensus_estimator.py)
- [../../../release/common/parents/parent_science_49bd2ba3_candidate_first128_paid_nocensus_science_v1.py](../../../release/common/parents/parent_science_49bd2ba3_candidate_first128_paid_nocensus_science_v1.py)
- [../../../release/common/readout/audit_capture_v1.py](../../../release/common/readout/audit_capture_v1.py)
- [../../../release/common/readout/client_capture_v1.py](../../../release/common/readout/client_capture_v1.py)
- [../../../release/common/readout/export_blocks_v1.py](../../../release/common/readout/export_blocks_v1.py)
- [../../../release/common/readout/export_first128_onpolicy_tensors_v1.py](../../../release/common/readout/export_first128_onpolicy_tensors_v1.py)
- [../../../release/common/readout/first128_onpolicy_hook_v1.py](../../../release/common/readout/first128_onpolicy_hook_v1.py)
- [../../../release/common/readout/first128_onpolicy_hook_v2.py](../../../release/common/readout/first128_onpolicy_hook_v2.py)
- [../../../release/common/readout/fit_blocks_full90_v1.py](../../../release/common/readout/fit_blocks_full90_v1.py)
- [../../../release/common/readout/lc_blocks2_fold0_v1.py](../../../release/common/readout/lc_blocks2_fold0_v1.py)
- [../../../release/common/readout/lc_blocks3_fold0_v1.py](../../../release/common/readout/lc_blocks3_fold0_v1.py)
- [../../../release/common/readout/lc_blocks_fold0_v1.py](../../../release/common/readout/lc_blocks_fold0_v1.py)
- [../../../release/common/readout/lc_gate_fold0_v1.py](../../../release/common/readout/lc_gate_fold0_v1.py)
- [../../../release/common/readout/prepare_capture_v1.py](../../../release/common/readout/prepare_capture_v1.py)
- [../../../release/common/readout/run_capture_pipeline_v1.py](../../../release/common/readout/run_capture_pipeline_v1.py)
- [../../../release/common/tails/sm40_cap4_size8e9.py](../../../release/common/tails/sm40_cap4_size8e9.py)
- [../../../release/common/tails/sm40_cap4_size8e9_xsg_off.py](../../../release/common/tails/sm40_cap4_size8e9_xsg_off.py)
- [../../../release/common/tails/xsg_off.py](../../../release/common/tails/xsg_off.py)
- [../../../release/common/tools/strip_gate_arm_v1.sh](../../../release/common/tools/strip_gate_arm_v1.sh)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7598](../../TEAM_LOG_2026-10-01_to_10-22.md#L7598)

<details>
<summary>All result files and records (79)</summary>

- [.gitignore](../../../release/.gitignore) · record
- [DEPENDENCY_MANIFEST.md](../../../release/337619/DEPENDENCY_MANIFEST.md) · record
- [LICENSE](../../../release/337619/LICENSE) · record
- [README.md](../../../release/337619/README.md) · record
- [BUILD_v1.json](../../../release/337619/build/BUILD_v1.json) · record
- [manifest.json](../../../release/337619/extracted/manifest.json) · record
- [SETUP_GUARD_v1.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/SETUP_GUARD_v1.json) · record
- [prefire.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/prefire.log) · record
- [PLAN_v1.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/PLAN_v1.json) · record
- [SUMMARY_v1.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/SUMMARY_v1.json) · record
- [server0.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/server0.log) · record
- [server1.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/server1.log) · record
- [server2.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/server2.log) · record
- [server3.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/server3.log) · record
- [server4.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/server4.log) · record
- [worker0.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker0.json) · record
- [worker0_events.jsonl](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker0_events.jsonl) · record
- [worker1.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker1.json) · record
- [worker1_events.jsonl](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker1_events.jsonl) · record
- [worker2.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker2.json) · record
- [worker2_events.jsonl](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker2_events.jsonl) · record
- [worker3.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker3.json) · record
- [worker3_events.jsonl](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker3_events.jsonl) · record
- [worker4.json](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker4.json) · record
- [worker4_events.jsonl](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5/worker4_events.jsonl) · record
- [setup5.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/setup5.log) · record
- [stage_a.log](../../../release/337619/gates/qual_fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1/stage_a.log) · record
- [RELEASE_337619.jsonl](../../../release/337619/hosted_grade/RELEASE_337619.jsonl) · record
- [SUMMARY.md](../../../release/337619/hosted_grade/SUMMARY.md) · record
- [package.json](../../../release/337619/packaging/package.json) · record
- [stage_a.json](../../../release/337619/packaging/stage_a.json) · record
- [validate.json](../../../release/337619/packaging/validate.json) · record
- [validate_package.json](../../../release/337619/packaging/validate_package.json) · record
- [EXPORTED_v1.json](../../../release/337619/seal_s1.5/EXPORTED_v1.json) · record
- [submission.tar.gz](../../../release/337619/submission.tar.gz) · archive
- [DEPENDENCY_MANIFEST.md](../../../release/337641/DEPENDENCY_MANIFEST.md) · record
- [LICENSE](../../../release/337641/LICENSE) · record
- [README.md](../../../release/337641/README.md) · record
- [BUILD_v1.json](../../../release/337641/build/BUILD_v1.json) · record
- [manifest.json](../../../release/337641/extracted/manifest.json) · record
- [SETUP_GUARD_v1.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/SETUP_GUARD_v1.json) · record
- [prefire.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/prefire.log) · record
- [PLAN_v1.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/PLAN_v1.json) · record
- [SUMMARY_v1.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/SUMMARY_v1.json) · record
- [server0.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/server0.log) · record
- [server1.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/server1.log) · record
- [server2.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/server2.log) · record
- [server3.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/server3.log) · record
- [server4.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/server4.log) · record
- [worker0.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker0.json) · record
- [worker0_events.jsonl](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker0_events.jsonl) · record
- [worker1.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker1.json) · record
- [worker1_events.jsonl](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker1_events.jsonl) · record
- [worker2.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker2.json) · record
- [worker2_events.jsonl](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker2_events.jsonl) · record
- [worker3.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker3.json) · record
- [worker3_events.jsonl](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker3_events.jsonl) · record
- [worker4.json](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker4.json) · record
- [worker4_events.jsonl](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5/worker4_events.jsonl) · record
- [setup5.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/setup5.log) · record
- [stage_a.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1/stage_a.log) · record
- [qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1.gates.log](../../../release/337641/gates/qual_fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1.gates.log) · record
- [RELEASE_337641.jsonl](../../../release/337641/hosted_grade/RELEASE_337641.jsonl) · record
- [SUMMARY.md](../../../release/337641/hosted_grade/SUMMARY.md) · record
- [package.json](../../../release/337641/packaging/package.json) · record
- [stage_a.json](../../../release/337641/packaging/stage_a.json) · record
- [validate.json](../../../release/337641/packaging/validate.json) · record
- [validate_package.json](../../../release/337641/packaging/validate_package.json) · record
- [EXPORTED_v1.json](../../../release/337641/seal_s1.35/EXPORTED_v1.json) · record
- [submission.tar.gz](../../../release/337641/submission.tar.gz) · archive
- [COPY_RECEIPT_v1.json](../../../release/COPY_RECEIPT_v1.json) · record
- [DEPENDENCY_MANIFEST.md](../../../release/DEPENDENCY_MANIFEST.md) · record
- [LICENSE](../../../release/LICENSE) · record
- [ARCHIVE_README.md](../../../release/ARCHIVE_README.md) · record
- [SHA256SUMS.txt](../../../release/SHA256SUMS.txt) · record
- [LEDGER_EXCERPT.md](../../../release/common/LEDGER_EXCERPT.md) · record
- [full90_BLOCKS_COEFFICIENTS_v1.json](../../../release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json) · record
- [WRITEUP_DISCLOSURE_DRAFT_v1.md](../../../release/common/writeup/WRITEUP_DISCLOSURE_DRAFT_v1.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-22-historical-release-package-preserves-109-files-per-file-checksums) · [Copy provenance](../../../PROVENANCE.json)

