# Late Gaussian restart cannot generate the inherited fourth cumulants accurately enough

2026-09-25T06:42Z · experiment · rejected · measured

Restarting the true network from exact Gaussian mean and covariance late in depth misses inherited cumulants. The required accuracy demands nearly exact propagation from the first layers.

Recorded measurement:

    - 2026-09-24 23:42 lead: PHYSICIST GAUSSIAN RESTART (MPS, net 35; net 100 running). Replace the true state at layer l0 by a Gaussian with the true 32M mean/cov, run the TRUE network forward, compare per-unit k3/k4 at L12-15 to the independent 32M truth. This is the ceiling of any construction starting at l0.
      - k4 at L14 is INHERITED: births after L12 give 2% of it, after L10 8%, after L6 30%, after L2 61%.
      - Best k4 error at L13-14 with EXACT physics after l0 (plus the best isotropic term), by start layer: L12 24-25%, L10 21-22%, L6 15%, L4 12%, L2 7.7%, L1 5.3%. The spec is 7%; the card is 25%.
      - k3 needs a start at L1 to reach 1.5%.
      VERDICT: late carriers and last-layer contractions are dead on physics; only a near-exact FULL-DEPTH fourth-order carrier (from L1-2) could meet the spec. The cost estimate goes on the physicist's page.
    - 2026-09-25 00:03 [physicist] k3/k4 PAGE: fable_physicist_v1/leadergap/RESULT.md section N. ANSWER: NO deterministic per-unit k4 source within ~7% at L13-14 for 10-20% of the bill, and neither proposed form can in principle.
      (1) GAUSSIAN RESTART (refit6b/carry/gauss_restart_mc.py + _eval.py; nets 35/100, 16M samples/row, MPS; true 32M mean+cov at l0, true net forward, vs independent 32M truth, noise removed by halves; control l0=0 exact within 1.2%): k4 made after L13/L12/L10/L6/L2/L1 = 0 / 0-2 / 5-8 / 25-30 / 61 / 73% of the L14 truth. Best k4 error at L14 with exact physics after l0 + the isotropic term: L12 25.1/22.3%, L10 22.5/20.2%, L6 14.6/16.2%, L4 11.6/12.7%, L2 7.7/7.7%, L1 5.3/5.2% (spec 7%, break-even 11%, card 27/25%). k3: 1.5% needs exact from L1; our bank (6-7%) = exact from ~L5-6. The last-two-layer contraction and any late carrier sit at the isotropic floor.
    - 2026-09-25 00:05 lead: k3/k4 SOURCE LINE CLOSED (physicist leadergap §N; Gaussian restart nets 35/100, 16M/row vs 32M truth).
      - L14's per-unit k4 is made early: after L12 0-2%, after L6 25-30%, after L2 61%, after L1 73%. The first ReLU alone makes 27%.
      - Best k4 error at L14 with exact physics from l0: L12 25/22%, L10 22/20%, L6 15/16%, L4 12/13%, L2 7.7%, L1 5.3%. Spec 7%, break-even 11%, the card 27/25%.
      - k3 at 1.5% needs exact physics from L1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/carry/gauss_restart_mc.py](code/fable_physicist_v1/refit6b/carry/gauss_restart_mc.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L764](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L764)
- [TEAM_LOG_2026-09-22_to_09-26.md#L765](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L765)
- [TEAM_LOG_2026-09-22_to_09-26.md#L766](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L766)
- [TEAM_LOG_2026-09-22_to_09-26.md#L767](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L767)
- [TEAM_LOG_2026-09-22_to_09-26.md#L768](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L768)
- [TEAM_LOG_2026-09-22_to_09-26.md#L788](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L788)
- [TEAM_LOG_2026-09-22_to_09-26.md#L789](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L789)
- [TEAM_LOG_2026-09-22_to_09-26.md#L793](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L793)
- [TEAM_LOG_2026-09-22_to_09-26.md#L794](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L794)
- [TEAM_LOG_2026-09-22_to_09-26.md#L795](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L795)
- [TEAM_LOG_2026-09-22_to_09-26.md#L796](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L796)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-late-gaussian-restart-cannot-generate-inherited-fourth-cumulants-accurately) · [Copy provenance](../../../PROVENANCE.json)

