# Encoder row scales and offsets: +43.0785360494% cross-validation error, zero of 90 wins

2026-10-21T08:54:28Z · experiment · rejected · measured

The fixed zero-added-operation encoder affine fit reproduced its baseline exactly but failed every training trust gate and all 90 validation comparisons. No damping, ridge or layer rescue sweep was performed.

Recorded measurement:

    Enc2 affine fixed fit retired1:52:15 a.m. Pacific. CV90+43.0785360494%,0/90wins, all five TRAIN trust gates failed; exact full90 independent first128 parent parity passed. FITTED28e6b4e7f55161edb85167692123d4e097b6b4beca9b912350734dc42f81660c; originalguard98c3e91ef8f672a6265f3fa5e998292164de57ca845a305c4a8d0a2d6190c4ab successful/whole-group-retired/child0/events0/swap0, peakRSS469.910MB, wall726.23s. E/CLOSURE_v1.json 1daddf834af9bb27a8b4f2718c7e97764c6444c83ede88cba1a0990e0738f227 closes with no full90refit/model/export/native and no damping/ridge/layer rescue. Exact receipts in E/fit_receipts_v1; sole small slot free.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py)
- [code/astra_oct20_bill_v1/enc2_affine_local90_v1/fit_enc2_affine_v1.py](code/astra_oct20_bill_v1/enc2_affine_local90_v1/fit_enc2_affine_v1.py)
- [code/astra_oct20_bill_v1/enc2_affine_local90_v1/price_enc2_affine_v1.py](code/astra_oct20_bill_v1/enc2_affine_local90_v1/price_enc2_affine_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7059](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7059)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-encoder-row-scales-offsets-43-0785360494-cross-validation-error) · [Copy provenance](../../../PROVENANCE.json)

