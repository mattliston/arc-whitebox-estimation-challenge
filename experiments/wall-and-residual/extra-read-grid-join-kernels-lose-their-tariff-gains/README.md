# Extra read-grid and join kernels lose their tariff gains after runtime-trigger costs

2026-10-16T10:56:21Z · experiment · rejected · measured

Paired fresh-network and record-replay evaluation rejected both extra kernel increments on the release-qualified swap. Approximate emergency routes and request overhead outweighed lower untriggered bill.

Recorded measurement:

    readgridwarm-.153%,first-.052%,publicmix-.148%;replaynet+3.23%(+.06%withoutoneshednet);joinwarm-.295%,publicmix-.279%;risk55.67%,worst122.5s;governor+.324%bill/+4.07%MSE

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_pubmax_v1/build_pubmax_v1.py](code/opus_pubmax_v1/build_pubmax_v1.py)
- [code/opus_pubmax_v1/cand_pubmax_A_parity.py](code/opus_pubmax_v1/cand_pubmax_A_parity.py)
- [code/opus_pubmax_v1/cand_pubmax_A_release.py](code/opus_pubmax_v1/cand_pubmax_A_release.py)
- [code/opus_pubmax_v1/cand_pubmax_AB_parity.py](code/opus_pubmax_v1/cand_pubmax_AB_parity.py)
- [code/opus_pubmax_v1/cand_pubmax_AB_release.py](code/opus_pubmax_v1/cand_pubmax_AB_release.py)
- [code/opus_pubmax_v1/cand_pubmax_B_parity.py](code/opus_pubmax_v1/cand_pubmax_B_parity.py)
- [code/opus_pubmax_v1/cand_pubmax_B_release.py](code/opus_pubmax_v1/cand_pubmax_B_release.py)
- [code/opus_exact_v1/dm2/build_dm2_v1.py](code/opus_exact_v1/dm2/build_dm2_v1.py)
- [code/opus_governor_v1/candidate_GOV_v3.py](code/opus_governor_v1/candidate_GOV_v3.py)
- [code/opus_governor_v1/candidate_GOV_v4g.py](code/opus_governor_v1/candidate_GOV_v4g.py)
- [code/opus_smtable_v1/cand_GOV3_smtable_v1.py](code/opus_smtable_v1/cand_GOV3_smtable_v1.py)
- [code/opus_stack_v1/assemble_final_v2.py](code/opus_stack_v1/assemble_final_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4145](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4145)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4125](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4125)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PUBMAX_RESULT_v1.json](PUBMAX_RESULT_v1.json) · result
- [BUILD_pubmax_v1.json](BUILD_pubmax_v1.json) · result
- [fragment_grid_remainder_v1.py.txt](fragment_grid_remainder_v1.py.txt) · result
- [DESIGNATE_spec_check_v1.json](DESIGNATE_spec_check_v1.json) · result
- [spec_check_v1.json](spec_check_v1.json) · result
- [fragment_lazybind_v2.py.txt](../../arithmetic-kernels/unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · code
- [fragment_swap_drop12_v1.py.txt](../../arithmetic-kernels/wall-capped-read-grid-swap-12-zero-saving-plans/fragment_swap_drop12_v1.py.txt) · code
- [fragment_smallsm_v2.py.txt](../../arithmetic-kernels/grouped-join-old-bank-kernels-warm-bill-0-347/fragment_smallsm_v2.py.txt) · code
- [fragment_smtable_cap_v2.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_cap_v2.py.txt) · code
- [fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-extra-read-grid-join-kernels-lose-their-tariff-gains) · [Copy provenance](../../../PROVENANCE.json)

