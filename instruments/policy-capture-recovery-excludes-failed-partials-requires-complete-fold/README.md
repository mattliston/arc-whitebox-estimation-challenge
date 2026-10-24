# On-policy capture recovery excludes failed partials and requires complete fold-specific source custody

2026-10-21T07:59:40Z · tool · informational · measured

Recovery retained the clean source-specific anchor groups and replayed failed remaining captures in new directories. The fixed fit required complete custody and sealed each fold before its own held-out evaluation; the record explicitly distinguishes that policy from sealing all folds before any held evaluation.

Recorded measurement:

    Onpolicy558pipeline nowAUTHORIZED afterrootstatic/fitter/source/plan/guardreview5d5e392a. Controller189185/start3094204 launchedonce06:27:00Z, originalshared36/BLAS4,6families4jobs/family aftereachown3anchorbyteproof;540captures+18unhookedcontrols. ExactlyonefixedTRAIN72/held18secondfit permitted onlyafterall540custodypass; no candidateexport/nativevalidation/fire. Fitterf318432e selftestclean originalguard0f6a12c3. Jointfirst128+squaredprotocol d0b7008d accepted forimplementation/selftestonly; actualfitnotyetrootauthorized. Exactprice911fe385 +5,898,240FLOPs, noadditivegains. Physical29closed.
    [accuracy_research] 2026-10-21T06:57:06Z ONPOLICY REMAINING FAILED original reserve guards; sole189185 and all6remaining controllers confirmed absent11:55:28p.m., no secondfit/retry. FAILURE_SUMMARY e03e32a3 local100files rehashed, all24groups retired/no cleanup/swap;9reserve+15cancel, maxRSS5500809216, maxcgroup5503746048, minavailable18893729792. All48 partialpreds preserved, inadmissible; six anchors8b8de0e1 valid. Root reviewing separated scheduling; newlaunches held. FIRST-square selftestfb7861e9 clean; source04343d75 no fullfit yet.
    REC/root dynamic765140ce and ops native-empty640aaeca authorized the unchanged recovery controller749bb186 once. Launched2026-10-21T07:58:49Z (12:58:49 a.m. Pacific), PID216924/start3645159, --fit-after-capture, REC/RECOVERY_LAUNCH_v1.json0b20ee9ce49c08f9f077c81a1dff51e4a61b4f2e926eb24fbc9865d5fd131302. Original4jobs/family,max8native,shared36,6.5GBcap,19GBreserve; waves(fold0,fold1),(fold2,fold3),(fold4,full90). Exactnativeempty beforeeachwave/optionalfit unchanged. No duplicate controller/retry. All12truth COMPLETE and chosen58c32rootcustodyde2b +ownfresh4bb8d verified. Fresh preflight allnative/truth/guarddead,187.156GBavailable,parent9.349GB,max8033/local128,OOMswap0. Existing48failedpartial predictions remain excluded and preserved;606completed/630attempted ceiling. One original fixed fit only after540 clean custody; no export/native/fire.
    - Recovery CAPTURE_FINAL88e23463257e99231da6e6a76cb7776e6291fd2e6f6443f6386f25b55b7b13e8 is successful: all six own-source families,540 valid captures+18 controls, all174 replacement native guards clean;48 discarded failed predictions preserved and630 attempted ceiling unchanged. Compact37-file local custody: recovery_pairs_v1/CAPTURE_CUSTODY_COLLECTED_v1.json02b10079618e453aeb2b2a53d6d28fc2624962098a107e8e3e1953afe2281fde.
    - Correction to earlier shorthand: old frozen fitter seals EACH fold before its OWN held18, then nextfold; its original full90 loop runs unconditionally before final pass decision. No adaptive selection and all final gates passed. Do not claim all5 old seals before any held evaluation or full90 conditioned on CV pass. New OP1 code has stronger all5seals-before-held protocol. Both treat historical777a pretrained estimator as fixed prior, not fully nested historical pretraining.

## Programs

- [code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/run_onpolicy_recovery_v1.py](code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/run_onpolicy_recovery_v1.py)
- [code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/export_final_tensors_v1.py](code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/export_final_tensors_v1.py)
- [code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/audit_onpolicy_recovery_v1.py](code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/audit_onpolicy_recovery_v1.py)
- [code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/fit_onpolicy_second_step_recovery_v1.py](code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/fit_onpolicy_second_step_recovery_v1.py)
- [code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/DISPATCH_ONCE_v1.py](code/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1/DISPATCH_ONCE_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6752](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6752)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6779](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6779)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6940](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6940)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7272](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7272)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7308](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7308)

<details>
<summary>All result files and records (15)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_EXPORT_REVIEW_v1.json](ROOT_EXPORT_REVIEW_v1.json) · result
- [ROOT_RECOVERY_REVIEW_v1.json](ROOT_RECOVERY_REVIEW_v1.json) · result
- [EXPORT_RECIPE_v1.json](EXPORT_RECIPE_v1.json) · result
- [STATIC_REVIEW_PACKET_v1.json](STATIC_REVIEW_PACKET_v1.json) · result
- [CAPTURE_FINAL_v1.json](CAPTURE_FINAL_v1.json) · result
- [RECOVERY_PLAN_v1.json](RECOVERY_PLAN_v1.json) · result
- [RECOVERY_READY_v1.json](RECOVERY_READY_v1.json) · result
- [RECOVERY_LAUNCH_v1.json](RECOVERY_LAUNCH_v1.json) · result
- [UNCHANGED_FIT_MATH_v1.json](UNCHANGED_FIT_MATH_v1.json) · result
- [CAPTURE_CUSTODY_COLLECTED_v1.json](CAPTURE_CUSTODY_COLLECTED_v1.json) · result
- [ROOT_STATIC_REVIEW_v1.json](ROOT_STATIC_REVIEW_v1.json) · result
- [PIPELINE_FINAL_v1.json](PIPELINE_FINAL_v1.json) · result
- [FIT_CUSTODY_COLLECTED_v1.json](FIT_CUSTODY_COLLECTED_v1.json) · result
- [SECOND_FIT_RECIPE_v1.json](SECOND_FIT_RECIPE_v1.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-21-policy-capture-recovery-excludes-failed-partials-requires-complete-fold) · [Copy provenance](../../PROVENANCE.json)

