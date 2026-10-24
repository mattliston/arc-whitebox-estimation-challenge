# Float32 arithmetic, exact copies and radical edge: -0.276832224% adjusted versus #337393

2026-10-20T20:59:42Z · experiment · adopted · measured

The combined source without the added covariance coefficient produced the strongest local verified stack at that point. Quiet production tests passed original guards but showed mean wall +1.600%, leaving no guaranteed platform margin.

Recorded measurement:

    NEW bestlocal micro_copyall_ecut_astra (NOaddedKC, Gram+f32mean/carrier+exactcopyall+radicaledge) all32-.276832224% SE.092541,25/32lower;held25-.264199835%SE.105587,20/25lower;raw-.0713254%,bill-.2056499%. Frozenv4aqualification-.614878754% (not incrementalgain). FullreceiptVS_BEST_337393_v1.json. AuxiliaryKC+K31age4+edge complete only-.131591%vsbest,held-.138834%; notpromoted.
    New permanentnogovscience51aec0294ab23ced8280cfd104d6ebc97647585904721822c4b96be3049af6d5; release4b8baf51fe5ced80f46907aea7ab4461124b993d17a178d73c9c75230a6100b7. ASTexactparent4d2a8d40/82ca6d99 + finalemptyrescuemap. All32parentcumulativecalls0/controlnull, validno-rescuepath equivalence. K/MICRO_COPYALL_ECUT_NOGOV_BUILD_AUDIT_v1.json. ORIGINALrelease_min PASS underquietbox: setup5max3.509371083s, StageApass, approvedgconly,0newsignatures. Archivefaf49095644316e9a5738fecd39cdce33d85b2e03287b9cfb5238e01ef16afa9, tagastra_micro_copyall_ecut_nogov_4b8baf51_v1; qualificationqual_micro_copyall_ecut_nogov_4b8baf51_v1. NOTYETFIRED; preparedbankcandidate.
    Quietproductionranks35,200 versusbankfd: meanwall+1.600%,backend+1.841%,residualwall+1.088%,threadresidualinstructions+.360%,requests+.544%;bill-.210728%. All4predictionbytesexactscience; rank35bills exact,rank200commonwarmsequence3,424,256FLOPoffsetcancels. Norescueinanyprediction. Original7.5GB/free50guardsPASS. This is NOTguaranteedplatformruntimeheadroom; knownworkerjitterdominates. K/MICRO_COPYALL_ECUT_NOGOV_QUIET_COMPARE_v1.json. Primaryqueue+5dispatchers drained/paused20:52:49–20:57:18Z, all6resumed. Runtimeaudits foundno supported5–10% exactmarginwithinthisscorebudget; no repeatedGC/fusion/stripauditneeded.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_client_dual_v3.py](code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_client_dual_v3.py)
- [code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_run_dual_v3.py](code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/boundary_run_dual_v3.py)
- [code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py](code/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_nogov_astra.py](code/lead/runs/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_nogov_astra.py)
- [code/lead/runs/fable_oct19_kc_v1/qualify_micro_copyall_ecut_nogov_isolated_v1.py](code/lead/runs/fable_oct19_kc_v1/qualify_micro_copyall_ecut_nogov_isolated_v1.py)
- [code/release/common.py](code/release/common.py)
- [code/release/setup_clock.py](code/release/setup_clock.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6301](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6301)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6302](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6302)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6303](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6303)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MICRO_COPYALL_ECUT_NOGOV_BUILD_AUDIT_v1.json](MICRO_COPYALL_ECUT_NOGOV_BUILD_AUDIT_v1.json) · result
- [MICRO_COPYALL_ECUT_NOGOV_QUIET_COMPARE_v1.json](MICRO_COPYALL_ECUT_NOGOV_QUIET_COMPARE_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337415: Edge compression and exact copy removal with residual routing disabled; prediction failure](../../../submissions/337415/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-float32-arithmetic-exact-copies-radical-edge-0-276832224-adjusted) · [Copy provenance](../../../PROVENANCE.json)

