# Active reduced-output audit, 2026-10-18

Source: `astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py` (parent supplied SHA prefix `95c87176`). Read-only source audit, no SDK/model imports, judges, training, submissions, or numerical processes. No closed rank/basis route revisited.

## Outcome

No exact reduced-output mechanism found that plausibly saves at least 1 GF net. The one clear unnecessary full output is the pair map's `pk22`/`post22`, but its **entire removable arithmetic is less than about 0.42 GF**, before paying any replacement contraction. This cannot fund the requested margin alone and does not justify a native experiment.

## Exact consumers and bounds

1. **Scalar fourth-moment output.** Active nonlinear body is line 844, confirmed by its binding assertion at 17341. At lines 935/964/968/969/977, the third Hermite channel becomes `pk22`, then `post22`, widens to binary64, receives a replacement diagonal, and is used solely by `sum(post22)`. An exact-real rewrite could contract this channel directly and apply its diagonal correction. The active Hermite fusion at 18329–18349 costs roughly 32 scalar arithmetic operations per produced entry (row-only work excluded); covariance and fourth-moment channels are already computed on 36 blocks of 128² entries, i.e. 589,824 entries each, not a full 1024². Fourteen noninitial hidden layers therefore cost at most about 0.264 GF for this complete channel. Even crediting ten full-plane arithmetic passes to its postprocessing adds only 0.147 GF. Thus gross deletion ceiling is ~0.411 GF / 240 GF = 0.171%; actual net gain is smaller because a scalar contraction still must read the arbitrary correlation and cumulant planes. This reassociates floating point reductions; it is not byte identical.

2. **Final covariance diagonal.** The final layer already calls `_dc_cov_diagonal` (22551), which computes `pushed = C_live @ W_live` and reduces `sum(W_live * pushed, axis=0)` (12492–12502). It never constructs `W.T @ C @ W`. With a general dense symmetric `C_live`, every column quadratic form is necessary; the remaining first product has no thin output discarded afterward.

3. **Final star diagonal and rank-one propagation.** `_flms_star_diagonal` already contracts `sum(P1*T2, axis=1)` (11103–11113). The active star ranks are all -1 (13531); `_ss_factors` uses only `rows = A.sum(axis=1)`, `rows @ A`, and two matrix-vector products (14665–14676, beta 0.85 at 19843). Holding star as a reduction-aware expression could remove some dense assembly at 22725–22800. However those are only O(15*1024²) elementwise passes. The cubic CP matrix read, target pair calculation, and BD change all have independent consumers. This does not eliminate a large GEMM and cannot independently supply 1 GF.

4. **Old-bank reconstruction.** It is not consumed solely by the response-plane read. The active core also needs its per-row signed cube (22612–22619), omitted-row squared-energy and all-row weighted column sums (14422–14448), and NP moments (15383–15425). Existing OTT already omits selected tail rows and repairs them (16216–16255). Exact cubic diagonals from `U F` would require a third-order rank-space tensor (rank 384: 9,511,040 symmetric degree-three coordinates), so replacing reconstruction with a small Gram contraction does not cover the actual consumer. No new exact thin-output cut identified.

5. **Birth direction plane.** `_bd_update` builds three dense directions (22828–22884); its returned `new32` is sliced to selected birth columns, but its full `delta` also feeds `born`, `cp_post`, and the star's two reductions. Dropping the unselected columns without separately preserving those reductions changes the recurrence. Even a careful combined reduction has only a few dozen full-plane passes of gross savings (~0.5 GF upper scale across 15 layers), not a new large matrix multiplication cut.

6. **Feature planes.** `rho`, `skplane`, and `postrho` (201–224) are consumed by row/column sums, squares, absolute values, clipping, and one matrix-vector product. Separable scale factors can move outside some reductions, but dense arbitrary planes still must be read; clipping and abs prevent one universal linear contraction. Expected savings are another O(15*1024²) small-pass count, with changed rounding, rather than ≥1 GF.

These findings are specific to the active definitions and their consumers. They do not assert that all exact arithmetic optimization is exhausted.
