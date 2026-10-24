# Isolated release wrapper binds all arguments and retires only owned descendant processes

2026-10-21T10:16:11Z · tool · adopted · derived

Review found incomplete argument binding and unsafe research resumption in the unexecuted first wrapper. The replacement binds candidate, verdict and archive, validates process identities, and leaves research paused if ownership or retirement is uncertain.

Recorded measurement:

    Source-only review of root_release_v1/run_release_isolated_root_v1.py found incomplete argv binding and cleanup that could resume research after an early-exited bash left descendants. V1 SHA5408e45e preserved/unexecuted. Root requested and reviewed separate v2 SHA75f0c976d29da6ea94c72281376fbdeccb2a7cc6bd2e9a6b8de0cb45c5ddcd45; preparation17880bed. Full6argv/candidate/verdict/last-fired3848/archive binding; observe owned descendants/new-session groups; individually identity-checked signals only, no killpg; uncertain/missing-root/unowned-project/survivor leaves research paused and durable failure. Original release_min ec26493b and guard6523c552 unchanged. After root source review, all8 injected synthetic metadata scenarios plus18 negative bindings passed: root_release_v1/SYNTHETIC_FIXTURE_RESULTS_v2.json SHA0973006323161884d77eceaf9d93f71affd19e66d42d2c0ea5f3966af7288f52. No real process API, native/candidate import, actual release/check/manifest/verdict or fire ran.
    - Release manifest ead7e8bb, isolatedwrapperv2 75f0c976/8metadatafixtures09730063. One releaseexec97947 ran10:21:28–39Z, all5setupPASS/max3.321799625s vs5s; originalguardRSS1.913159680GB/free82/swap0, allsetupbills27,687,270,724. StageAall8gatesPASS; archive2regularmembers/exactf938. Prefireonlypreviouslyapprovedgcimport/zero neweinsums. Ownershipretirementclean/no cleanupsignals/77740restored. Rootartifactreview82a867b8 andindependentaudit2807390f agree; rootonefireauthorityc245dc23. Packager.14/.10/minPython3.11metadata identicaltofailedarchive, separatefromofficialsetup.16.1/.12.3+Python3.10compilation; no newexception.
    Root's unchanged release_min completed once at2026-10-21T10:21:39Z under reviewed isolation wrapperv2, with owned descendants retired and research queue77740 restored. Standalone runtime-lane artifact audit `H/root_release_v1/INDEPENDENT_RELEASE_ARTIFACT_AUDIT_v1.json` SHA `2807390fbe6bd1fa8c340610f79eb1e1743fbd84462e77d7c77805d5d1d44a02` verifies57 input pins, all5 official setup workers (maximum3.321799625s), original external setup guard (peakRSS1,913,159,680B, free82, swapgrowth0), all8 StageA gates, three recorded whest check exits0, exactf938 archive membership, unchanged release script/guard and exact priorfailed4685 reference. No candidate import/native or gate rerun by this audit. Five existing nonfatal rlimit notes remain visible. Source/package manifest differences from official setup versions were checked against the prior fired manifest and are unchanged. Root independent artifact review82a867b8 also passed.

## Programs

- [code/run_release_isolated_root_v2.py](code/run_release_isolated_root_v2.py)
- [code/run_release_isolated_root_v1.py](code/run_release_isolated_root_v1.py)
- [code/test_release_wrapper_metadata_v2.py](code/test_release_wrapper_metadata_v2.py)
- [code/audit_release_artifacts_runtime_v1.py](code/audit_release_artifacts_runtime_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7242](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7242)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7252](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7280](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7280)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SYNTHETIC_FIXTURE_RESULTS_v2.json](SYNTHETIC_FIXTURE_RESULTS_v2.json) · result
- [INDEPENDENT_RELEASE_ARTIFACT_AUDIT_v1.json](INDEPENDENT_RELEASE_ARTIFACT_AUDIT_v1.json) · result
- [RELEASE_ISOLATED_PLAN_v2.json](RELEASE_ISOLATED_PLAN_v2.json) · result
- [PREPARATION_REVIEW_PACKET_v2.json](PREPARATION_REVIEW_PACKET_v2.json) · result
- [ISOLATION_v2.json](ISOLATION_v2.json) · result
- [ROOT_FIRE_ONCE_AUTHORIZATION_v1.json](ROOT_FIRE_ONCE_AUTHORIZATION_v1.json) · result
- [ROOT_RELEASE_ARTIFACT_REVIEW_v1.json](ROOT_RELEASE_ARTIFACT_REVIEW_v1.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-21-isolated-release-wrapper-binds-all-arguments-retires-only-owned) · [Copy provenance](../../PROVENANCE.json)

