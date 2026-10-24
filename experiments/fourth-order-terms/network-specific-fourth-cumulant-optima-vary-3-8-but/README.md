# Network-specific fourth-cumulant optima vary by 3.8%, but internal features cannot predict them

2026-10-19T23:02:00Z · experiment · rejected · measured

The spread of per-network optimal band-2 k4 scales is real structure (slopes on the two truth halves correlate 0.80; a perfect per-network dial would be worth raw -0.50%), so a conditional dial was built from the card's own per-layer features (variance dispersion at layers 3/8/12). In-sample it read raw -0.47% on 28 ranks, the honest leave-one-out arm only -0.17% (se .28), and a fit on 64 fresh networks showed the features correlate 0.09-0.33 with the optimum with sign-flipping coefficients and a worse out-of-store result; a search over 168 internal features found nothing above a spurious |r| 0.3-0.4. The radial k4 is one scalar per layer per network, so all per-unit structure sits in the K4L term.

Recorded measurement:

    band 2 per-net optima sd 4.5%, range -6.5%..+10%; perfect per-net dial raw -0.68% (band 2) / -0.35% (band 3) on 7 ranks; truth-half slope corr +0.80; true optimum sd 3.8%, truth-noise 1.4%; best LOO linear dial cv_vv at L8-10 r -0.45 for raw -0.08%; k4cv in-sample raw -0.47% all28 (held-out-21 -0.41%, 20/28), adjusted -0.38%; k4cv2 raw -0.48% (held-out -0.37%, 16/28); k4cvloo raw -0.17% (se 0.28), held-out-21 -0.06%, 12/28 -> adjusted -0.09% / +0.03%; release form rel_k4cv2 vs l1314x (32 ranks) raw -0.31%, bill -0.02%, adjusted -0.32% (18/32) in-sample; 64 fresh nets (2 min/net): feature corr 0.09-0.33 (store in-sample 0.67), out-of-sample +0.04..+0.14% worse, optimum sd 0.048; 168 features best |r| 0.3-0.4 (spurious ~0.3); grow + re-adapt (births floor 0.005 at L1-6) raw -0.75% for +1.44% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/fable_oct19_kc_v1/probe_k4feat_v1.py](code/lead/runs/fable_oct19_kc_v1/probe_k4feat_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/k4feat_analysis_v1.py](code/lead/runs/fable_oct19_kc_v1/k4feat_analysis_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_k4cv_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_k4cv_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_k4cv2_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_k4cv2_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_k4cvloo_sci_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_k4cvloo_sci_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_rel_k4cv2_v1.py](code/lead/runs/fable_oct19_kc_v1/candidate_rel_k4cv2_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/probe_fresh_k4_v1.py](code/lead/runs/fable_oct19_kc_v1/probe_fresh_k4_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6005](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6005)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6009](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6009)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6015](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6015)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6017](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6017)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6019](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6019)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6025](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6025)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6027](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6027)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6029](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6029)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6041](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6041)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6063](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6063)

<details>
<summary>All result files and records (3)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [RESULT32_v1-d5add8d1.json](RESULT32_v1-d5add8d1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-network-specific-fourth-cumulant-optima-vary-3-8-but) · [Copy provenance](../../../PROVENANCE.json)

