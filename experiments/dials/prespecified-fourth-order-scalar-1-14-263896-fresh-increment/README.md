# Prespecified fourth-order scalar 1.14: -.263896% fresh increment but worse than parent on 32 networks

2026-10-20T21:36:20Z · experiment · rejected · measured

A four-network cache-safe scalar screen selected a prespecified endpoint. Its full-32 combination with hidden-channel pruning harmed the parent, so standalone fresh percentages were not added.

Recorded measurement:

    All four coadapt fresh seeds7100092–95 complete: selected prespecified GSK.9/GK4 1.14 endpoint -.616946% vs bank, SE.288418,3/4 lower; incremental to same-screen microedge default -.263896%,SE.136026,4/4 lower. Both width-only caches explicitly reset and U reconstructed from all28 predictions. No off-grid fit: strong mixed responses make additive proxies unreliable. A/gate_coadapt_v1/FROZEN_SELECTION_GK4114_v1.json and RESPONSE_DIAGNOSTIC_v1.json. Independent baked GK4 source/cache audit PASS.
    Allthreepriority32 COMPLETE versus best337393: EH14+FG -.278827996%SE.099651,24/32lower,held-.286301936%19/25; GK4+FG -.168919680%held-.167024599%; EH14+GK4+FG -.169095778%held-.185924579%. ExtraGK4hurtsownparent;do notstack. EH14onlyessentiallytiesfallbackmicroedge-.276832%. Both oldfreshscreenlarger gainsdidnottransfer. FullVS_BEST_337393_v1 receipts saved. Freshdrop14-only -.3595569%vsbankbut+.3018815%vsEH14,0/4lower;closed. sqrt4alreadyclosed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v2.py](code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v2.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v2.py](code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v2.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/check_cache_ast_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/check_cache_ast_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/diagnose_response_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/diagnose_response_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v1.py)
- [code/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py](code/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6324](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6324)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6342](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6342)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FROZEN_SELECTION_GK4114_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/FROZEN_SELECTION_GK4114_v1.json) · result
- [RESPONSE_DIAGNOSTIC_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/RESPONSE_DIAGNOSTIC_v1.json) · result
- [README_v1.md](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/README_v1.md) · record
- [settings_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/settings_v1.json) · result
- [MANIFEST_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/MANIFEST_v1.json) · result
- [MANIFEST_v2.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/MANIFEST_v2.json) · result
- [CACHE_AST_PROOF_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/CACHE_AST_PROOF_v1.json) · result
- [bank_settings_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/bank_settings_v1.json) · result
- [V2_LOG_REPAIR_PROOF_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/V2_LOG_REPAIR_PROOF_v1.json) · result
- [RESULT_FRESH4_v1.json](../../join-sketch/join-sketch-oversampling-64-192-first-seven-1-627310/RESULT_FRESH4_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-prespecified-fourth-order-scalar-1-14-263896-fresh-increment) · [Copy provenance](../../../PROVENANCE.json)

