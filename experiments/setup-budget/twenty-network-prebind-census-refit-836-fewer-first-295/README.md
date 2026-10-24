# Twenty-network prebind census refit: 836 fewer first and 295 fewer warm requests

2026-10-16T03:38:35Z · experiment · adopted · measured

The fixed 649-key list contained 138 unused keys and omitted 141 observed keys. Swapping and reordering keys preserved predictions and lowered billed lazy bindings on out-of-sample boundary networks.

Recorded measurement:

    138unused/141unlisted;first48875to48039;warm45993to45698;risk19.64to14.88%;bound588to595;bill-.34%first/-.18%warm;setup4.439s,slow4.581s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_setup_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_vnh128_pbt_v1.py](code/candidate_vnh128_pbt_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3796](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3796)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SETUP_PREBIND_DECISION_v1.json](SETUP_PREBIND_DECISION_v1.json) · result
- [fragment_prebind_vnh128_v1.py.txt](fragment_prebind_vnh128_v1.py.txt) · result
- [diag_pbt_7103248.json](diag_pbt_7103248.json) · result
- [guard_diag_pbt.json](guard_diag_pbt.json) · result
- [quick_vngb128_s1.json](quick_vngb128_s1.json) · result
- [quick_vngb128_s1_guard.json](quick_vngb128_s1_guard.json) · result
- [quick_vngb128_s2.json](quick_vngb128_s2.json) · result
- [quick_vngb128_s2_guard.json](quick_vngb128_s2_guard.json) · result
- [quick_vngb128_s3_guard.json](quick_vngb128_s3_guard.json) · result
- [quick_vngb128_s3b.json](quick_vngb128_s3b.json) · result
- [quick_vngb128_s3b_guard.json](quick_vngb128_s3b_guard.json) · result

</details>

## Related work

- shipped in: [Submission #336389: census-refitted prebinding for the adaptive 128-column join](../../../submissions/336389/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-twenty-network-prebind-census-refit-836-fewer-first-295) · [Copy provenance](../../../PROVENANCE.json)

