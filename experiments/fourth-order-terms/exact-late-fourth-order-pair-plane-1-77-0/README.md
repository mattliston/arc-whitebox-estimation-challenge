# Exact late fourth-order pair plane: +1.77% / -0.69% MSE, with earlier-only substitution worse

2026-10-20T06:16:34Z · experiment · rejected · measured

Exact fourth-order pair planes at layers 10-14 did not consistently improve native predictions on networks 35 / 100. Half-strength substitution was also inconsistent and layers 12-14 substitution gave +2.45 / +3.10%.

Recorded measurement:

    - [fable-lead] 2026-10-20T06:16:34Z K22 PLANE ORACLE DEAD (native, truth K22 through the pair map's k22_override at L10-14): +1.77% / -0.69% (half strength +0.14 / -0.72; L12-14 +2.45 / +3.10); scaling the card's own k22 object x2 / x0.5 explodes (it is not a free plane). Night tally of native tests on the shipped card: K31 rest, K21 plane parts, kernel order, own-cumulant transport, terminal readout refit, all dial families (pe34, GNN gains, k4_15, dcv floor, bfix, K31R rows, rowcut) = nothing >= 0.1% that holds on 32 ranks; the card is at a flat optimum for every cheap change; the leaders' -17% is structural propagation accuracy (exact late k3+k4 jointly -5.5/-14%, exact late covariance -26/-34%) that no cheap producer reaches. Last two lanes (k31rl_12, bfix_020) running; no fire candidate at the 0.005e-9 bar tonight.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rel_nosync_sci_v1.py](code/candidate_rel_nosync_sci_v1.py)
- [code/native_cov_oracle_v2.py](code/native_cov_oracle_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6142](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6142)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-exact-late-fourth-order-pair-plane-1-77-0) · [Copy provenance](../../../PROVENANCE.json)

