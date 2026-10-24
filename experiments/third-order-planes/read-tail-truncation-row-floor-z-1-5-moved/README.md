# Read-tail truncation row floor z 1.5 moved to 1.25/1.75/2.0, global or per layer: all worse (+0.82 to +1.73% adjusted)

2026-10-19T14:28:48Z · experiment · rejected · measured

The read-tail truncation's gate floor (Phi(-1.5) in the plane reads, now _RTT_Z) removes 32-64 response rows per layer at layers 2-14. Lowering it to 1.25 cuts more rows from the reads (25% of the bill) but the rank-2 completion's error costs 7x the saving; raising it to 1.75 or 2.0 buys raw that saturates at about -0.8% while the bill keeps growing; 1.75 at layer 14 only is null (the raw gain is spread over the inner layers): z 1.5 is a sharp optimum and the family is closed in every form.

Recorded measurement:

    z 1.25 (28 ranks vs stk1): adjusted +1.725% (se .40, 8/28 lower) = raw +2.062% for bill -0.309%, held-out 21 +1.375%; z 1.75: raw -0.789% for bill +1.614% (adjusted +0.822%, 7/28); z 2.0: raw -0.821% for bill +2.501% (+1.671%); 1.75 at L14 only: raw -0.040% (se .07), bill +0.118%; rtt125 part01 bill -0.07/-0.11/-0.84%, requests +1,095 on the first network

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/fable_oct19_kc_v1/candidate_rtt125_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_rtt125_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_rtt175_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_rtt175_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_rtt20_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_rtt20_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_rtt14_175_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_rtt14_175_sci_v1.py)
- [code/lead/runs/fable_oct19_audit_v1/probe_rtt_v1.py](code/lead/runs/fable_oct19_audit_v1/probe_rtt_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5933](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5933)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5938](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5938)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5941](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5941)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5957](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5957)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5961](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5961)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-read-tail-truncation-row-floor-z-1-5-moved) · [Copy provenance](../../../PROVENANCE.json)

