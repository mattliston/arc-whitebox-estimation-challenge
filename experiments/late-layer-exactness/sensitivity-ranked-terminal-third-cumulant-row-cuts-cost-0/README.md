# Sensitivity-ranked terminal third-cumulant row cuts cost +0.7..+3.7% MSE for -0.1% FLOPs

2026-10-20T08:38:43Z · experiment · rejected · measured

Terminal third-cumulant contributions were too large to remove cheaply among the rows retained by the existing threshold. Truth-free row rankings failed and a truth-conditioned ranking selected residual coincidences unavailable during prediction.

Recorded measurement:

    [fable-lead] 2026-10-20T08:38:43Z L15 K3 ROW CUT (kink side / sensitivity-ranked) DEAD — the parked bill lever priced natively (lead/runs/fable_oct19_kc_v1/native_l15_probe_v1.py + l15_rowcut_price_v1.py; nets 35/100/37/39, the shipped card): the terminal k3 term c_i = k3_i e_i/6 has rms 5.1-5.8e-4 per unit vs the output error rms 1.05-1.16e-4 (dropping every kept row's k3 = +1,360..+1,640% MSE): every kept row matters. Bill per row 0.0031-0.0036% (83-95 rows in the fired 3.0<z<=3.5 band = -0.295%). Truth-free rankings at -0.1% bill (28-32 rows): drop the lowest-|z| rows (z < 0.09-0.14) raw +0.7..+3.7%; lowest |e| (both tails) +2.2..+3.3%; at -0.3% bill +12..+47%. The pre-push proxy (diagonal transport sum_a W_ia^3 post_k3_a(L14)) correlates only 0.24-0.31 with the pushed k3 (|.| 0.08-0.13): no cheap k3 estimate. The truth-side 'oracle' ranking (by c_i^2 - 2 c_i r_i) shows -8..-16% for 30-90 rows, which is the selection of rows whose k3 happens to align with the error (|c| < 2|r|), not a wrong k3: unusable. Bill shelf now empty at the final layer too (row cut z 3.0, pair cuts, RTT floor, u-leg, kink side).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/fable_oct19_kc_v1/native_l15_probe_v1.py](code/lead/runs/fable_oct19_kc_v1/native_l15_probe_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/l15_rowcut_price_v1.py](code/lead/runs/fable_oct19_kc_v1/l15_rowcut_price_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6154](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6154)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-sensitivity-ranked-terminal-third-cumulant-row-cuts-cost-0) · [Copy provenance](../../../PROVENANCE.json)

