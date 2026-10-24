# Missing second-order full-tensor diagram has cosine .04-.10 to error and negligible one-layer value

2026-09-25T09:03Z · experiment · rejected · measured

Two proposed diagrams were already present. The remaining covariance-dressed full third-order tensor had poor alignment and prohibitive arithmetic cost.

Recorded measurement:

    - 2026-09-25 02:03 [physicist] H2 FALSIFIER: FAIL (leadergap RESULT.md section O addendum; refit6b/carry/t3_screen.py, t3_screen.json; B2 ep1 card, r35, MPS; brute-force certificate 1e-15). Of the three second-order candidates, the two plane diagrams ARE the card's j1 (f''' g g C K21) and j0 (f'' f'' g C K21), and the (2,1,1) K4 feed is in the card isotropically (b, birth weight radial/3). The one missing diagram T3 = f''_a f''_b g_c C_ab K3_abc (full carried K3 x one covariance bond; 10-22k bank columns) is 40% of j1's size but orthogonal to the distinct error: cosine .099/.079/.066/.063/.036 at L6/8/10/12/14, one-layer cut 0.5/0.3/0.2/0.2/0.0% (j1: cos .40-.49, cut 7-11%). Bar was cos>=.3 with cut>=5%; FAIL at every layer (and exact T3 would cost ~30,000 U/layer anyway). loopR4 now running on MPS.
    - 2026-09-25 02:03 lead: RIVALS-EDGE HYPOTHESES:
      - H2 CLOSED (physicist §O addendum): two of the three candidate diagrams are already in the card (f'''ggCK21 = j1, f''f''gCK21 = j0; the (2,1,1) feed is isotropic via b/radial). The one missing diagram T3 = f''_a f''_b g_c C_ab K3_abc (full-bank computed on MPS, brute-force certified 1e-15) has cosine .04-.10 with the distinct error and a one-layer cut <= 0.5% (j1: .40-.49 / 7-11%). It would also cost ~30,000 U/layer.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/carry/t3_screen.py](code/fable_physicist_v1/refit6b/carry/t3_screen.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L934](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L934)
- [TEAM_LOG_2026-09-22_to_09-26.md#L935](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L935)
- [TEAM_LOG_2026-09-22_to_09-26.md#L936](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L936)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [t3_screen.json](t3_screen.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-missing-second-order-full-tensor-diagram-has-cosine-04) · [Copy provenance](../../../PROVENANCE.json)

