# JSK v1 — a cheaper and better join basis (2026-10-17 22:xxZ, [opus-lead])

## The defect

Each join (L7/L10/L13) compresses the joined cohorts J (live rows x K columns, K = 6144/5568/4672 on store rank 35) plus the
previous old tier into a rank-384 frame Qn. Today Qn = orth(G[:, rows]) with G = J W J^T + Qp core Qp^T (W = lam^2) and rows =
the 384 largest Gram diagonals: a Nystrom basis, one product of 2 m K r flops. The 10-16 NT hunt showed the exact rank-384
eigenbasis buys the whole no-join prize (-1.30% xMSE natively for NTF64) but bought it by widening the sketch (more Gram columns,
each 2 m r flops) plus an eigh: rates .64-1.14, "join subspaces exhausted".

Offline on the dumped joins of rank 35 (joinsketch/join35_L{7,10,13}.npz; metric = relative Frobenius error of the (2,1) plane
of the projected content, at the join and after one push through w32):

| join | today (Nystrom) | exact rank-384 | gap |
|---|---|---|---|
| L7 | .0271 / .0307 | .0232 / .0260 | 17% |
| L10 | .0148 / .0175 | .0127 / .0147 | 17% |
| L13 | .0123 / .0137 | .0099 / .0107 | 24% |

Adaptive pivots (block greedy) move it by 1-2% (consistent with the 10-16 "3-4 rank units"). The core's accuracy (Nystrom vs
exact F W F^T vs a one-third subsample) does not move the basis at all.

## The lever: subsample the columns, spend the saving on a power step

The weighted column energies lam_c^2 |J_c|^2 are concentrated (top 10% of columns = 34% of the energy). The exact top-384
subspace of the Gram of the heaviest kc columns is within 1-3% of the full one for kc >= 1536 (L7: .0235 at 3072, .0238 at 2048,
.0244 at 1536; L10: .0128 at 1536; L13: .0102 at 1536). So:

1. start Y1 = G0[:, rows] on the kc0 heaviest columns (cost kc0/K of today's product), plus the old-tier term Qp core Qp[rows]^T;
2. one power step Y = G1 Y1 on the kc heaviest columns (two products of 2 m kc r), plus Qp core (Qp^T Y1);
3. QR as before; Fj = Qn^T J as before; the core for the next join = Fj1 W1 Fj1^T (exact for the kc columns, PSD).

Costs in units of today's sketch product and plane errors (offline, rank 35):

| join | plan (kc0 -> kc) | cost | plane / pushed | closes |
|---|---|---|---|---|
| L7 | 1024 -> 2048 | 0.83 | .0247 / .0278 | 62% |
| L7 | 2048 -> 2048 | 1.00 | .0245 / .0277 | 67% |
| L10 | 1024 -> 1536 | 1.01 | .0131 / .0153 | 81% |
| L10 | 1024 -> 1024 | 0.83 | .0134 / .0157 | 67% |
| L13 | 768 -> 1536 | 1.15 | .0109 / .0119 | 58% |
| L13 | 1024 -> 1024 | 0.99 | .0114 / .0124 | 38% |

No eigh anywhere (flopscope bills eigh at 9 n^3; a client-side eigh would cost ~30 ms of residual per join). More than one power
step without re-orthonormalisation blows up numerically (4 steps: x2 error); re-orthonormalising costs a 0.49 GF QR per step, so
v1 uses exactly one step. Importance-sampled columns lose to the deterministic heaviest columns. Gaussian or heaviest-column
starts lose to the Nystrom-rows start at equal cost.

## Build

`jsk/make_joinsketch_v1.py <parent> <out>`: textual patch of BOTH live join bodies (`_astra_bcc_push_impl`, the live one, and the
OTT-captured `_shared_old384_push` fallback): the sketch block and the Nystrom core solve are replaced; the core is computed after
Fj from Fj[:, cols1]. `_JSK_PLAN = {7: (1024, 2048), 10: (1024, 1536), 13: (768, 1536)}`. candidate_v47j.py = v40ua + JSK v1
(sha8 1b331b5b).

Expected bill: products 11.9 GF dense vs 12.3 today (joins) -> ~-0.1%; +2 requests per join. Expected error: if closure maps to
the NT native results (NTB32 ~64% -> -0.75%, NTF64 ~85% -> -1.30%), -0.7..-0.9% final MSE at zero bill. Judge: store ranks
(cross-half U, final layer) vs bnd_v34y / store32_ctl_v34y_clean; call on >= 7 ranks, stack on 32.

## Results

(filled in below as they land)

### 22:4xZ — v47j (JSK v1, no whitening) FAILS natively: rank 35 raw +28.3% (U 1.108e-8 -> 1.422e-8), bill -0.25%, residual .239 s
Per-layer U: L8 +3.8, L10 +7.1, L12 +11.5, L13 +20.6, L14 +16.3 — the error grows from the first join on. Cause found offline
(joinsketch/join_basis_v6.py): FLOAT32. The power step squares the spectrum (Y2 singular-value ratio s384/s1 = 1.4e-9 at L7,
1.4e-11 at L10, 1.9e-12 at L13 vs 3e-5 / 3e-6 / 1e-6 for today's Y1); in float32 the products round the trailing directions
away and the QR returns noise for them: f32 plane errors .0289 / .0319 / .0437 vs today's .0271 / .0148 / .0123 — exactly the
native degradation. In float64 the scheme is as good as predicted (.0248 / .0132 / .0107).

Fixes that restore the float64 quality IN float32 at every join: (A) QR of the start Y1 before the step (billed 491/415/415 MF),
(B) Nystrom-factor whitening Y1 <- Y1 L^-T with L L^T = Y1[rows] (cholesky 19 MF + inv 113 MF + one 960x384x384 product ~140-280
MF). Cholesky-QR with an f32 Gram fails (not PD). Flopscope bills (probe_bill_v1.py, f32): matmul dense 2mnk, qr 2(2mnk-2k^3/3),
cholesky ~n^3/3, inv ~2n^3, solve 2n^3/3 + 2n^2 nrhs; gathers/sums/argsort negligible (< 8 MF each). f32 Cholesky of Y1[rows] is
fine (eigen ratio 1.2e-5, residual 4e-7).

v48j = JSK v2 (fix B with a 1e-6 trace jitter and a QR fallback), sha8 b4df8fcc. Expected bill ~+0.15% (whitening + exact cores),
expected error -0.7..-0.9%.

### 22:57Z — v48j on 8 store ranks: final-layer raw MEAN +1.68% (lower on 1/7), bill +0.12%, ADJUSTED +1.80%: a loss
Rank 35 was the lucky one (-0.72%); ranks 150/200/250/259/277/287: +0.9/+5.1/+0.6/+1.6/+1.8/+2.6%. Per layer the L7 join helps
(L8-L10 -0.5..-1.6% on most ranks), the L10 join is neutral, the L13 join hurts (L13-L14 +0.7..+4%). v49j (v3) rank 35 +3.66%.

### 23:0xZ — cause: the core, not the sketch (join_core_v8/v9/v10.py)
Today's Nystrom core Qn[rows]^-1 R^T is exact to 0.1% (Y = G[:, rows] over ALL columns). My cores: v48j Fj1 W1 Fj1^T = 71% of the
true trace at L7 and 54% at L10 (it also dropped the inherited old-tier Gram); v49j's generalized-Nystrom core 65%. The next join
then weights the old tier (the majority of the content at L13) at half strength and its basis under-serves it. The sketch generalizes:
rank 200 (worst native) basis improves at all joins in f32 with the whitened step (-10/-11/-11%).
Zero-cost completion: rescale the subsample core to the full joined trace (rel err .085/.111) + the inherited overlap core_old overlap^T.
v50j = JSK v4 (sha8 8bbf1b9f): sketch of v48j + full-trace core + fold of the old tier as columns when its Cholesky succeeds.

### 23:2xZ — CLOSED: kernel rounding
v50j (full-trace core) ranks 35/200: adjusted +3.75% / +6.60%. Cause = the fast kernels' rounding (Frobenius rel 4-8e-6, 10-25x BLAS;
probe_kernel_noise_v1.py). Emulated offline (join_basis_v13/v14.py): at eps 3e-7 (max-normalised rms, ~the kernel's level) every
whitened power-step variant is 2.5-4x WORSE than today's Nystrom (which is unmoved up to 1e-6): the step's gain lives in the tail
directions it lifts, and after the step those sit at the noise floor. Regularised/shifted variants either collapse or equal Nystrom;
sparse multi-row test vectors are 8-10% worse than coordinate rows; a BLAS-accurate JSK costs +0.6% bill for ~-0.4% MSE. Closed.

### 00:11Z — v51j (JSK at L7 only, today's Nystrom at L10/L13): 7 ranks adjusted MEAN +0.01% (se .48): zero. Closed in every form.
