# Learned edge hidden width eight: -.150665691% Mac and -.2338808% fresh adjusted versus #337422

2026-10-21T00:11:47Z · experiment · adopted · measured

A nonlinear teacher fit reduced learned-edge hidden width to eight. It produced negative paired results on Mac32, Linux32 and four fresh networks and later served as a parent for readout improvements.

Recorded measurement:

    Edge8 originalMac32 completed00:08:22Z (5:08PM Pacific), all12guardsgood, queue/fivejudgesresumed. SCI627c79b7, V/science32_msg8_edgehidden8_parallel5_v1. ActualpairedBEST337421all32-.5794739409%SE.170661,26/32lower;held25-.5513501482%SE.201007,20/25lower;raw-.17825143%,bill-.40192507%. Exactedgeincrementvs337422SCI777a all32-.1507%,held-.1600%,raw+.0046%,bill-.1553%; 21/32lower. Localprojection1.4148910425380463e-9 isONLY.0792887%belowJ2W. Matt explicitly sayskeepstacking/getbiggergainbeforefire; HOLD,no releasequalificationorupload. TenUTC21submissionsunused. Earlierfirst7VM-.2222%wasvsoldermsg8SCI23ff, includesloaderdifferenceanddifferenthost; actualMacfirst7vs777a-.1174%,notinterchangeable. Newbestreportsalwaysvs421.
    OriginalMacedge6SCI5b5f5f3e all12guardsgood, wrapper5473exit0/resumedoldqueue/fivejudges00:20:50Z. ActualvsBEST422all32 -0.169319149%, held25 -0.286418680%, first7+.248893%; rawall+.0378%,bill-.2071%,19/32lower. Edge8SCI627c79b7 all32 -0.150665691%,held -0.159975376%; thesearealternatives,notadditive. Userinformedlocalbank~.1693%,notyetfresh/runtimequalified,needs~.30additionalpp toJ2W. Edge8fresh4exact92–95started5:23:21PacificPID40646, unchangedguards/shared36, existingactual421+777controls.
    VMexact422Linux32controlnowcomplete: edge8-.2140704%held-.2012067;edge6-.1819533%held-.2612975; donotmixhostswithMacnumbers. OLDoffpolicy rank24+ .0366448%all/+.0975452%heldvs422, rejectstack417c/98a3; noedge6+oldrank24build. Layeroutputrepairfirst7+.10958%vsedge8, no independentpiece. Currentactual777a trajectory rank16SCI96b0e972/PID40842 andrank24SCIc84bc15e/PID40844 launched00:23:55Z aftergradient/tinycold-warm-cache/NumPyfreeimportchecks. Plan352d985d,shared36/v7/BLAS4,all128nonlinearfeaturesretained. Currentmsg4/6 nativePIDs37756/37878continue. VnewfixedCM4 first-affine output-sensitivefitapproved: samearchitecture/bill, actual777asavedstate/head/message targets,train4only/val3descriptive, onecountedguardedfit/twoBLAS1workers, thenchecks/native32. Rootpreparingboundededge6finalteacher7D FD (56trainingpredicts), noFDlaunchyet. Billcurrenttrajectorysourcesowner,Vfit/scoring,jointVMdispatch/publicmonitor/fresh.
    Fresh4 exact seeds7100092–95 now complete for edge8 and6 against exact777a: edge8-.2338808% adjusted (SE.139616,4/4lower;raw-.080521,bill-.153488), edge6-.1312268% (SE.381647,2/4lower;raw+.073572,bill-.204650). Edge8 is more consistent independent confirmation; no additive gain or runtime qualification. Mac32 strongestedge6-.169319/held-.286419 versusedge8-.150666/held-.159975.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/edge_hidden_nonlinear_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_edge_v1.py](code/check_edge_v1.py)
- [code/export_hidden_v1.py](code/export_hidden_v1.py)
- [code/fit_edge_hidden_v1.py](code/fit_edge_hidden_v1.py)
- [code/native_hidden8_v1/candidate_msg8_edge_hidden8_nonlinear_science_v1.py](code/native_hidden8_v1/candidate_msg8_edge_hidden8_nonlinear_science_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6413](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6413)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6420](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6420)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6421](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6421)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6426](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6426)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VS_BEST_337422_v1.json](VS_BEST_337422_v1.json) · result
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [VS_EXACT_MSG8_337422_v1.json](VS_EXACT_MSG8_337422_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337466: Edge width 8 with per-layer correction fitted on 90 networks](../../../submissions/337466/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-learned-edge-hidden-width-eight-150665691-mac-2338808-fresh) · [Copy provenance](../../../PROVENANCE.json)

