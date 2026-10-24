# Preserving top covariance modes in a Gaussian restart improves kurtosis but not skew accuracy

2026-09-25T09:03Z · experiment · rejected · measured

Hybrid restarts with exact leading modes still failed the required third-cumulant accuracy. Complement tests placed the missing skew in mixed cumulants between leading and remaining modes.

Recorded measurement:

      - H1 FAILS on k3 (analyst interim, r35; r100 queued): subspace-Gaussianized restart at l0 = 6, R = 64: k4 .092/.087 (passes <= .11) but k3 .082/.073, unchanged from R = 0 (fails <= .03). The top modes carry ~half of k4's non-Gaussian content but none of k3's per-unit spread; even R = 256 leaves k3 .031-.038. k4 alone is worth ~-5%. H1 closes unless r100 disagrees.
    - 2026-09-25 02:36 [physicist] Section O addendum 2 (leadergap RESULT.md): the analyst's H1 subspace restart on BOTH nets fails on k3. l0=6, R=64: k4 .092/.087 (r35) .086/.089 (r100) PASS vs .11, k3 .082/.073 and .078/.072 FAIL vs .03 (R=0: .085/.077, .080/.070). R=256 gives k3 .03-.04. Top modes carry the kurtosis-relevant content, not the skew. With H1 and H2 both failed, section O's read stands: sum of small things; the path is bill + refits after real structural changes (loopR4 -0.18%, c0 flat).
    - 2026-09-25 02:37 lead: H1 CLOSED on both nets (analyst + physicist): the top-64 subspace kept exact fixes most of k4 (.15 -> .09) but none of k3 (.07-.08 vs the .03 bar); the k3 content is spread over hundreds of directions. Both rival-edge hypotheses (H1, H2) are closed. The reading stands: the edge is many small differences our one-component tests can't isolate. Remaining path: bill (S25 package with R320, the S26 k-split package) + refits only after real structural or physics changes; the weight-average probe is pending.
    - 2026-09-25 02:41 analyst: H1 FALSIFIER FAILS (fable_analyst_v1/RESULT.md section 40; subspace_restart_mc.py, MPS lock with-block; r35 + r100, 8M per half, common random numbers; scored with the physicist's gauss_restart_eval.py 'hyb'). Keep the TRUE state in the top-R covariance modes at pre-l0, Gaussianize the complement, run the true network: at l0 = 6, R = 64 (58% of variance) gives k4 .086-.092 (passes <= .11) but k3 .072-.082 (fails <= .03; the full restart is .070-.085), R = 256 (91%) k3 .031-.038. Controls: R = 1024 = 0, R = 0 = the physicist's restart. COMPLEMENT TEST (r35): Gaussianize ONLY the top 64 (or 256) modes, keep the rest true: k4 .149-.154, k3 .072-.086 = as bad as the full restart. So the bulk's own non-Gaussianity gives nothing, the top modes' own content gives ~half of k4 plus k3's level, and the per-unit k3 spread lives in MIXED top x bulk cumulants. A top-mode subspace carrier misses them by construction. H1 closes.
    - 2026-09-25 02:43 [physicist] Section O addendum 2 REWORDED per the analyst's complement test (their §40): Gaussianizing ONLY the top 64/256 modes (rest true) is as bad as the full restart (k4 .149-.154, k3 .072-.086, r35 l0=6), so the bulk's own cumulants give nothing and the per-unit k3 spread sits in MIXED (top,top,bulk)/(top,bulk,bulk) cumulants, not in the complement. H1 closed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/subspace_restart_mc.py](code/fable_analyst_v1/subspace_restart_mc.py)
- [code/fable_physicist_v1/refit6b/carry/gauss_restart_eval.py](code/fable_physicist_v1/refit6b/carry/gauss_restart_eval.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L937](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L937)
- [TEAM_LOG_2026-09-22_to_09-26.md#L953](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L953)
- [TEAM_LOG_2026-09-22_to_09-26.md#L954](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L954)
- [TEAM_LOG_2026-09-22_to_09-26.md#L956](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L956)
- [TEAM_LOG_2026-09-22_to_09-26.md#L967](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L967)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../../third-order-planes/gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-preserving-top-covariance-modes-gaussian-restart-improves-kurtosis-but) · [Copy provenance](../../../PROVENANCE.json)

