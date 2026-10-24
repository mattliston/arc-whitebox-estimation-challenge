# Runtime comparison stops on reference RSS before any candidate prediction; singleton protocol preserves guards

2026-10-21T08:44:25Z · incident · informational · measured

The verified reference exceeded its unchanged RSS guard during the third network, producing no candidate evidence. A separate fixed singleton-cold protocol retained the same 13 networks, original guards and actual invoice accounting.

Recorded measurement:

    The08:41:36Z summary's liveMacruntime statement is superseded. Polling confirmed isolated session40703 had stopped08:38:38Z on control_part01: exactverified466 reference reached7,510,163,456bytes RSS versus unchanged7,500,000,000backstop duringrank200. Footprint7,275,338,880/free81%/swapgrowth0; two referencepredictions35/150, zero49bd/f938predictions. Originalguard retiredentiregroup; queue77740restored/no cleanup errors. Preserveallfailedartifacts; NC/root_coverage13_v1/CONTROL_MEMORY_FAILURE_REVIEW_v1.json 80ae2669e6aeee74fc449b1944d7b529245bb48c69500635f8cfccd988173e1a. No usablepairedruntime, no candidatefailure/performanceclaim, no blindunchangedretry. Runtime lane preparingdistinct fixedsame13 singletoncoldpairs withoriginalguards andno partialreuse; warm/cold billing differences explicitlyhandled. FulloriginalMac32group qualification remainsmandatory and rootstaticreview 80c83304e2d7f1a9f01bc5f88426a7bb2c6d32e3db02cd65312f22c1f944d164 permits one49bdMac32 next afterlivepreflight. No release/fire; final10remains2used/8remaining.
    Root first f938 fixed13 session40703 failed08:38:38Z in unchanged337466 control part01 before any candidate: original RSS guard7510163456>7500000000 during rank200, footprint7.275GB/free81/swap growth0; two partial control rows preserved, no reuse. Root failure80ae2669; this is not candidate runtime evidence. Separate latent metadata defect found: H original isolation command map lacked final analyze command; it was never reached and did not cause RSS failure. Original packet preserved.
    Root requested a distinct fixed all13 singleton-cold protocol. `H/root_cold13_v1` READY cb1974b9/plan c6c5f16b/isolation392ccd07/runner99aaa9d5/analyzer1d090eae prepared, no native. All same13 ranks/order, fresh control→candidate for each, original125s/7.5GB/free50 guard and harness unchanged. STATIC_COMMAND_SCHEMA8491851d proves26 native+1 analyzer labels, exact command correspondence,129 pins and alloutputs absent. Exact byte anchors all13; oldownfe8 invoice equality onlyfive matching original cold ranks. Actual bills recorded for all26 arms; never transfer eight oldwarm bills/ratios. Cold-only evidence cannot qualify warm retained-cache behavior; original32 originalgroups remains separate.

## Programs

- [run_isolated_root_v1.py](run_isolated_root_v1.py)
- [analyze_coverage_v1.py](analyze_coverage_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7039](../../TEAM_LOG_2026-10-01_to_10-22.md#L7039)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7051](../../TEAM_LOG_2026-10-01_to_10-22.md#L7051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7053](../../TEAM_LOG_2026-10-01_to_10-22.md#L7053)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CONTROL_MEMORY_FAILURE_REVIEW_v1.json](CONTROL_MEMORY_FAILURE_REVIEW_v1.json) · result
- [COVERAGE_PLAN_v1.json](COVERAGE_PLAN_v1.json) · result
- [ISOLATION_v1.json](ISOLATION_v1.json) · result
- [ROOT_RESULT_REVIEW_v1.json](ROOT_RESULT_REVIEW_v1.json) · result
- [ROOT_REVIEW_v1.json](ROOT_REVIEW_v1.json) · result
- [STATIC_COMMAND_SCHEMA_v1.json](STATIC_COMMAND_SCHEMA_v1.json) · result
- [ISOLATED_PLAN_v1.json](ISOLATED_PLAN_v1.json) · result
- [COVERAGE_RESULT_v1.json](COVERAGE_RESULT_v1.json) · result
- [READY_ROOT_REVIEW_v1.json](READY_ROOT_REVIEW_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-runtime-comparison-stops-reference-rss-before-any-candidate-prediction) · [Copy provenance](../../../PROVENANCE.json)

