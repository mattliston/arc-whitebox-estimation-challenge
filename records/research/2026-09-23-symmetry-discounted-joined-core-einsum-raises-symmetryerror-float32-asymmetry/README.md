# Symmetry-discounted joined-core einsum raises SymmetryError from float32 asymmetry

2026-09-23T14:14Z · incident · failed · measured

Weighted Gram and covariance symmetry optimizations reduced work, but inferred symmetric float32 output violated the runtime tolerance. The joined-core and layer-zero einsums were withdrawn and a call-site gate was added.

Recorded measurement:

    - 2026-09-23 08:05 [engineer] SYMMETRIC-BILLING MEASUREMENT on S6b+coldbind3 (no builds; sym_census.py, sym_price.py, sym_time.py, sym_server_check*.py, sym_elem_census.py in fable_engineer_v1). (1) Symmetric-output products = 5.4% of the bill (r35 cold / r100 warm alike): covariance second product w^T(Cw) 4.3% (14/net; symmetric output, NOT a Gram), old-tier joined core Fj diag(w) Fj^T 0.78% (3/net, 384 x 4.6-6.1k x 384, weighted Gram), layer-0 W^T W 0.24% (_resident_gram512 already computes 3 of 4 blocks), old core sandwich 0.06%. Everything else is non-symmetric: star + C w + w^T A 17.2% (square), pushes/reads 68% (rect). No (P o P) L (P o P)^T read exists in S6b. Elementwise ops with a symmetric square operand: 0.22% of the bill (half-price SymmetricTensor ops would save <= 0.1%). (2) Prices, units of 2mkn, server path (bill identical in-process): covariance second product 0.551 -> 0.435 with one level of upper blocks (deeper is worse: 256-blocks run dense), at +100 ms server CPU each; Cholesky route (chol 0.167 + L^T w 0.551 + Gram 0.396-0.5) = 1.11-1.22 vs 1.10 now: no gain, and C is singular when units die. Weighted Gram 0.591 -> 0.503 as einsum('ia,a,ja->ij'), 240 -> 30 ms. Layer-0 Gram 0.435 -> 0.396 with einsum diagonal blocks, 243 -> 103 ms. (3) Server path: discount holds; symmetric einsum is BLAS-backed (30 ms vs 17 ms plain GEMM vs 130-240 ms for our Winograd paths); S+S on a SymmetricTensor bills 0.5/element, astype drops the symmetry. SAVINGS: covariance upper blocks -0.91% bill at +1.4 s server CPU/net (2.5 Gflop per CPU-second, ~4x pad07's rate); weighted-Gram einsum -0.13% at -0.63 s; layer-0 -0.02% at -0.14 s. All three: about -1.05% bill at +0.6 s/net; the two win-win items alone -0.15% at -0.77 s. Not bit-exact (rounding level; einsum is the more accurate path).
    - 2026-09-23 07:14 [engineer] --symout BUILT (lead-approved, E-43; symout_fragment.py + 2 count-asserted call-site edits; rounding-level): covariance w^T(Cw) by upper blocks (3 x 512-wide _mm, lower block mirrored; the two star products stay batched at P=2), layer-0 Gram with einsum diagonal blocks (_sym_gram; _resident_gram512 untouched because _cov_compact passes a weighted pair), old-tier joined core as einsum('ia,a,ja->ij') (its SymmetricTensor output passes the finite check, _mm and dense adds unchanged). File fable_engineer_v1/candS6b_cb3so_mem.py (cfe454f0) = S6b recipe + --coldbind3 --symout + mem,yright,arena. In-process (OpenBLAS env) vs cb3: util r35 cold -1.159%, r100 -1.082%, r37 -1.072%, r39 -1.037%; final MSE -0.092/-0.267/+0.109/-0.034% (pooled -0.067%); max prediction change 1.1e-5; handles -654..-1126 (the P=3 batch frames are gone). Expected server CPU +0.6 s/net (+1.4 covariance, -0.63 old core, -0.14 layer 0). Needs the porter's VALID40 vs cb3 (bar: pooled raw within +-0.05%) and engineer2's CPU gate. setupbind v3 (setupbind_fragment_v3.py) skips the P=3 batch frames when --symout is present.
    - 2026-09-23 07:45 [engineer] --symout v2 (symout_fragment_v2.py; the builder now points at it): the covariance's second product is 2 products, top half-rows [G11 G12] (512x1024x1024) + G22 (512x1024x512), G21 mirrored: 0.428 units (v1's 3 blocks 0.435), 38 ms/layer faster than v1. File fable_engineer_v1/candS6b_cb3so2_mem.py (b05b80dd) = S6b recipe + --coldbind3 --symout + mem,yright,arena. In-process vs cb3: util r35 cold -1.210%, r100 -1.139%, r37 -1.124%, r39 -1.086%; final MSE -0.098/-0.138/+0.117/-0.129% (pooled -0.056%); handles -0.6..-1.1k. Compute is NEUTRAL (product_profile, concurrent pair, cold r35: wall 72.21 vs 72.24 s, process CPU 63.84 vs 63.82 s; the covariance's two products +1.78 s, the smaller batch -1.79 s). Server path (engineer2's census, r35 cold): requests +1.6k (+3.7%), residual +23 ms, server peak RSS -428 MB; under load 17 the local server CPU read +2.8..+4.7 s (ABBA) and tracks round trips, not compute: engineer2's quiet gate decides. Expected platform cost about +0.8 s/net (1.6k requests x 0.5 ms). v1 (candS6b_cb3so_mem.py) superseded.
    2026-09-23 08:14 porter: S8 (4f91e8f4) BLOCKED. The symout joined-core einsum('ia,a,ja->ij') raises SymmetryError on r100 (both envs) and r42 (.venv): float32 core asymmetry 1-2.5e-6 at L13 vs flopscope atol 1e-6 (margin diagnostic: 1 in 6 nets raises). Tested fix sent to the engineer: X=Fj*sqrt(w), einsum('ia,ja->ij',X,X) = exactly symmetric, same half-price billing (+4.6k flops), equal or better accuracy. The .venv runs flopscope 0.10.0, the OpenBLAS env 0.12.0.
    2026-09-23 08:15 porter: pre-fire einsum gate fable_porter_v1/einsum_site_diff.py <cand> <ref=last fired>: FAIL on any new repeated-operand einsum or symmetric-tag call. S8 (symout v2) FAILS (3 new sites); S7 vs S6b PASSES. Waiting for symout v3 (covariance blocks only) to rebuild S8.
    2026-09-23 08:17 porter: S8 v3 (stack_S8_so3_mem.py 5116cee9 = engineer's candS8so3_mem.py byte-for-byte; symout v3 covariance blocks only): einsum gate PASS vs S7, validate OK; parity + VALID40 running (~08:50). S8 v2 (4f91e8f4) withdrawn.
    - 2026-09-23 08:27 [engineer] --symout v3 BUILT (symout_fragment_v3.py; lead's CRITICAL rule: no inferred-symmetric f32 einsum): covariance upper blocks only (2 plain _mm + G21 mirrored); the joined-core and layer-0 einsums are withdrawn and the builder's call-site edits removed. Files: fable_engineer_v1/candS8so3_mem.py (5116cee9) = fired S7 recipe (#332026, 9011916a) + --symout; candS8so3sb_mem.py (43a1adab) = + --setupbind (v3). Diff vs the fired S7 file: purely additive, 39 lines (_sym_upper, _flms_cov_star), no existing line changed. Symmetry audit (AST): einsum/outer/as_symmetric/symmetrize call sites identical, 90 in both files; the only repeated-operand einsums (einsum('i,j,ij->ij', m2, m2, cov) in _nonlinear_pair_batch; einsum('ij,ij,j->ij', bp, bp, _e2_bb) in _predict_binary) are in S7 and cannot be inferred symmetric (plain cov / asymmetric index pattern); outer(x,x) sites are all in S7 and exactly symmetric. In-process OpenBLAS vs fired S7: util r35 cold -1.021%, r100 -0.966%, r37 -0.981%, r39 -0.927%; final MSE +0.108/-0.104/+0.116/-0.021% (pooled +0.028%); max prediction change 1.1e-5; handles -183. .venv (numpy 2.2.6, where v2 raised): r35 then r100 run clean. Server path (engineer2's census, cold r35, S7/v3/v3/S7 at load 19-26): requests 44,920 -> 47,445 (+2,525, +5.6%), round trips +2,609, residual +7 ms on the pair means with +-90 ms between repeats; quiet estimate +10-13 ms. setupbind v3 on top: bit-identical, cold -0.42%, warm unchanged.

## Programs

- [sym_census.py](sym_census.py)
- [sym_price.py](sym_price.py)
- [sym_time.py](sym_time.py)
- [sym_elem_census.py](sym_elem_census.py)
- [symout_fragment.py](symout_fragment.py)
- [candS6b_cb3so_mem.py](candS6b_cb3so_mem.py)
- [../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer_v1/setupbind_fragment_v3.py](../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer_v1/setupbind_fragment_v3.py)
- [symout_fragment_v2.py](symout_fragment_v2.py)
- [candS6b_cb3so2_mem.py](candS6b_cb3so2_mem.py)
- [einsum_site_diff.py](einsum_site_diff.py)
- [stack_S8_so3_mem.py](stack_S8_so3_mem.py)
- [candS8so3_mem.py](candS8so3_mem.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer_v1/symout_fragment_v3.py](../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer_v1/symout_fragment_v3.py)
- [candS8so3sb_mem.py](candS8so3sb_mem.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L234](../../TEAM_LOG_2026-09-22_to_09-26.md#L234)
- [TEAM_LOG_2026-09-22_to_09-26.md#L249](../../TEAM_LOG_2026-09-22_to_09-26.md#L249)
- [TEAM_LOG_2026-09-22_to_09-26.md#L255](../../TEAM_LOG_2026-09-22_to_09-26.md#L255)
- [TEAM_LOG_2026-09-22_to_09-26.md#L262](../../TEAM_LOG_2026-09-22_to_09-26.md#L262)
- [TEAM_LOG_2026-09-22_to_09-26.md#L263](../../TEAM_LOG_2026-09-22_to_09-26.md#L263)
- [TEAM_LOG_2026-09-22_to_09-26.md#L265](../../TEAM_LOG_2026-09-22_to_09-26.md#L265)
- [TEAM_LOG_2026-09-22_to_09-26.md#L269](../../TEAM_LOG_2026-09-22_to_09-26.md#L269)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-symmetry-discounted-joined-core-einsum-raises-symmetryerror-float32-asymmetry) · [Copy provenance](../../../PROVENANCE.json)

