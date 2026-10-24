# Squared second-hidden readout: -.258929085% Mac32 and -.317014687% Linux32 versus #337466

2026-10-21T05:38:14Z · experiment · rejected · measured

A fixed squared-hidden readout passed its local 90-network proxy gate and improved full-32 native predictions on both hosts. Its actual increments over the hidden-complement parent differed by host and were measured directly.

Recorded measurement:

    Mac squared-hidden original32 FINISHED at05:35:38Z (10:35p.m.), root session78209 exit0 and77740sameidentityresumed. Output D/squared_hidden90_v1/science32_edge8_complement_squared_hidden90_parallel5_v1: adjusted-.258929085% vsactual337466, held25-.277938641%,20/32lower,SE.155393%,bill+.004156933%. Root fullguard/custodyreviewand exactincrementvsb54 are NOT done; Fablemustcomplete beforebank. Source155cde56/RELb9d5f036; VM32controller161958/strict161978/localcallback95817 running atlastcheck. Macslotnowfreeapartfromnormalownedqueue; inspect actualprocessesbeforeuse. Headfoldea86/f158 Mac32 stillnotrun; VMexactincrementvsb54-.060303695%/held-.012458222%,notbankalone.
    Squared-hidden2 PREREG2c5343c4, authentic777a perlayer128 f32 squares, independentfold-fit complement193 baseline, fixedridge.1/damping.5/90LOO. ProxyLOO−.150810577%,78/90lower,SE.0132385; train−.528910287%; physical9.0657%/maxlayer25.4402%. Passed the frozen−.15% floor narrowly. FITTEDc4a2c2e5; originalfitguard7dc0916b clean. Complete official15-layer bill+5,898,240FLOPs (+.00249421007% versus337466), statebytes/liveunits checked. No native adjusted gain claim.
    Squared-hidden155cde56 VM32 completed05:39:42Z. Independent audit D/squared_hidden90_v1/native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json SHAde66c2d7997dd9e2f67a80c223c9be2bd97e541231e218868110d92da8474f6a rehashed95files incl source,verified32preds/12unchangedguards,zerorescue/missing/skipped/ownermiss/swap/OOM,maxRSS5.452GB. Actualpairedvs337466 adjusted-.317014687%/held25-.394277969%,21/32lower,SE.155364%; exactincrementvsb54-.189770464%/held-.194191626%,22/32lower. No runtime/releasequalification claim. Root stillownsMacreview.
    Root squaredMac32 independent custody PASS: all32/12guards/source/zerorescue/compiles verified. Actual466-relative adjusted-.258929085%/held-.277938641%; exactb54increment-.066856581%/held-.182348126%,20/32lower. OriginalMacv4guards recorded0.805GiB systemswapgrowth in6groups, below original2GiBconditional/6GiBhard retirement limits, noerrors/failreads. It is NOT a zero-swap run. Root initial audit's erroneous zero-swap assumption preserved separately; no guard changed. Rootreviewbc5b9b15 in R21/ROOT_SQUARED_MAC32_REVIEW_v1.json. SquaredVM32 custody95files PASSde66c2d7:466-relative-.317014687%/held-.394277969%,21/32lower; b54increment-.189770464%/held-.194191626%,22/32lower; VMzeroSwap/OOM.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/squared_hidden90_v1/check_squared_tiny_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/check_squared_tiny_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/qualify_passed_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/qualify_passed_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/price_squared_hidden_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/price_squared_hidden_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/run_price_and_fit_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/run_price_and_fit_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/fit_squared_hidden_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/fit_squared_hidden_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/check_squared_import_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/check_squared_import_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/audit_native32_vm_custody_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/audit_native32_vm_custody_v1.py)
- [code/astra_oct20_bill_v1/squared_hidden90_v1/READOUT_TAIL_v1.py](code/astra_oct20_bill_v1/squared_hidden90_v1/READOUT_TAIL_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_oct20_bill_v1/hidden_complement90_v1/native_v1/hidden_complement90_per_layer/candidate_hidden_complement90_per_layer_science_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/native_v1/hidden_complement90_per_layer/candidate_hidden_complement90_per_layer_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py](code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)
- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6683](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6683)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6664](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6664)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6713](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6713)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6724](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6724)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_SQUARED_MAC32_REVIEW_v1.json](ROOT_SQUARED_MAC32_REVIEW_v1.json) · result
- [QUALIFY_RECIPE_v1.json](QUALIFY_RECIPE_v1.json) · result
- [PREREG_v1.json](PREREG_v1.json) · result
- [PIPELINE_RECIPE_v1.json](PIPELINE_RECIPE_v1.json) · result
- [STATUS_CURRENT.md](STATUS_CURRENT.md) · record
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-squared-second-hidden-readout-258929085-mac32-317014687-linux32-versus) · [Copy provenance](../../../PROVENANCE.json)

