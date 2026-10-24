# fable_porter_v1: native gain-mixture b fix (port, parity, refit-to-native pipeline), 2026-09-22

**Verdict.** The native port of the lead's b fix is faithful to the torch research card (`fable_refit_v1/model_bank_bfix.py`).
Gamma 0 is bit-exact to the bank. Pooled over VALID40, native final MSE matches torch within 0.02% at gamma 0.5.
Single nets differ by up to about 1%, which is the card's own torch-vs-native fidelity and is equally present at gamma 0.

## The fix and where it sits
In the live loop (the last `_predict_binary` of the bank candidate), right after
`b = ((g * bpre + radial * d * vv / 6) if _FULL_LOWER_MOMENTS else ...)` the builder inserts

    if _BFIX_GAMMA:
        b = b + _BFIX_GAMMA * (radial_new - radial) / 6 * m

and appends the module global `_BFIX_GAMMA = <gamma>` at the end of the file. The quantities are the torch card's:
`radial` is the incoming coefficient; `radial_new` already includes the omitted-radial source; `m` is the mean after the learned hook
(`m = m + dm`). The insert sits before `trace_post`, so, as in torch, the fix reaches the innovation, the births and the star.
The builder locates the loop by AST and asserts each dependent statement occurs once, in the expected order.
Cost: 2n FLOPs + 3 scalar ops per layer over 15 layers, i.e. 75 extra flopscope requests per net (about 0.1 s platform wall).
Measured util is unchanged at 5 decimals (r35 0.18828, r100 0.1846; VALID40 mean 0.188870 vs bank 0.188858).

## Scripts (all in this directory)
- `build_bfix_variant.py <base_cand.py> <gamma> <out.py>`: the patch (refuses to overwrite).
- `build_refit_candidate.py <params.pt> <gamma> <out.py>`: pinned bank (sha d79463f3...) + fix at gamma + `fable_refit_v1/export_params.py`
  literals from a refit .pt (`dict(p, a, c, e)`); rejects a .pt with any other key, since the exporter would silently drop it.
- `eval_refit.sh <params.pt> <gamma> <tag>`: build + paired VALID40 vs the bank (`fable_refit_v1/paired_eval.py`, 2 jobs x 1 thread, nice 10);
  appends one summary line to `eval_service.log`. About 31 min.
- `same_candidate.py <a.py> <b.py>`: same program and bit-equal literal values (decoded), ignoring literal formatting.
- Diagnostics: `trace_native.py`, `trace_torch.py`, `compare_traces.py`, `trace_source.py`, `trace_select.py`, `compare_select.py`.
- Built candidates: `cand_bfix_g0.py` (= bank), `cand_bfix_g05.py`, `cand_bfix_g085.py`; round-trip checks `cand_refit_orig_g{0,05}.py`.

## Parity: native vs torch final MSE
Torch values are the lead's `fable_refit_v1/bfix_test_m.log`. Native values are from `fable_refit_v1/score_variant.py`, logged in `parity_g*_r35_r100.log`.

| gamma | rank | native | torch | native vs torch | native util |
|---|---|---|---|---|---|
| 0 | 35 | 1.648017e-08 | 1.648466e-08 | -0.03% | 0.18828 |
| 0 | 100 | 1.680305e-08 | 1.683410e-08 | -0.18% | 0.1846 |
| 0.5 | 35 | 1.676271e-08 | 1.677682e-08 | -0.08% | 0.18828 |
| 0.5 | 100 | 1.910860e-08 | 1.914059e-08 | -0.17% | 0.1846 |
| 0.85 | 35 | 1.981686e-08 | 1.976179e-08 | +0.28% | 0.18828 |
| 0.85 | 100 | 2.589896e-08 | 2.580660e-08 | +0.36% | 0.1846 |

Gamma 0 on r35 equals the bank exactly: 1.648017257382719e-08.

VALID40 pooled (`pooled_parity_valid40.txt`; native = the paired run below, torch = `bfix_test.py` at 2 threads):

| gamma | pooled native/torch | per-net sd | largest single-net gap | native above torch |
|---|---|---|---|---|
| 0 | 1.00048 | 0.26% | 0.61% | 23/40 |
| 0.5 | 1.00018 | 0.39% | 0.99% (r37) | 22/40 |

The fix's own effect agrees: gamma 0.5 raises pooled raw MSE by 5.79% natively and by 5.83% in torch, with 4/40 wins in both.

## Why the remaining single-net gaps are not the port
- **Layer traces on r100** (`cmp_g{0,085}_r100.log`). At gamma 0.85, the fix coefficient and `b` agree with torch to 2e-6 at L0
  (float32 level). Up to L14 they agree to at most 3e-4. The change the fix makes to `b` (gamma 0.85 minus 0) agrees to 6e-7 at L1-L2.
- **Positive controls.** Replacing the post-hook mean by the pre-hook mean in torch moves that change by 7.8e-6 at L1, about 10x the observed
  native-vs-torch gap there. The omitted-radial source is 10-112% of the radial increment at L4-L14 (`trace_source_g085_r100.log`),
  against an observed coefficient gap of 3e-5 to 2e-3. So either porting mistake would have been visible.
- **Selections on r37**, the worst net (`sel_cmp_g{0,05}_r37.log`). All 45 unit selections contain the same units in native and torch
  at gamma 0 and 0.5; only the order of near-tied units differs. The divergence is diffuse (top 10 units hold 12% of it) and grows in L10-L15.
  At the final layer it is 3.4% of the error vector at gamma 0 and 4.2% at gamma 0.5. The native-vs-torch discrepancy is equally large
  at gamma 0 (3.42% of the error vector on r100 at both gamma 0 and 0.85).
- The torch card itself moves up to 0.05% per net between thread counts (r37, r39).

## Refit-to-native round trip
`build_refit_candidate.py fable_refit_v1/params_orig.pt 0.5` gives the same program as `cand_bfix_g05.py`. All 205 top-level statements
are equal. The only textual difference is the re-compressed `_GNN_LITERAL` blob, whose decoded float32 bytes are identical. The same holds at gamma 0
against the bank-exact `cand_bfix_g0.py`. The key guard was tested: a cumulant-head .pt is refused before any file is written.

## Baseline: plain gamma 0.5 (bank parameters, no refit) vs bank on VALID40
`paired_valid40_bfix_g05_plain.json`: raw_ratio 1.0579, adj_ratio 1.0580, adj_wins 4/40, util 0.188870 vs 0.188858, wall 1844 s.
A gamma-0.5 refit must recover this 5.8% before it gains anything.

## Caveats
- A refit .pt does not record gamma. Pass the gamma the refit was trained at.
- Expect native-vs-torch single-net final-MSE differences with sd about 0.4% (up to about 1%). Pooled VALID40 differences are about 0.05% or less.
- Bank rows are bit-reproducible (r37 matches `fable_refit_v1/paired_sanity.json` exactly). Caching them would halve each paired eval,
  but that is not implemented because `paired_eval.py` was specified as is.
- The b-fix patch and `fable_refit_v1/build_cumhead_variant.py` compose syntactically (checked on `cand_bfix_g05.py`), but that combination has not been scored.

---

## Update 2026-09-22 19:30: bank cache, cached paired evaluation, generic (3,1)-plane plumbing
The lead closed the K21/b-fix line (partition, additive and offb all refuted), so no further b-fix ports were made.
`eval_refit.sh` and `build_refit_candidate.py` (v1, b-fix based, uncached) are superseded by the tools below; they are kept unchanged.

### Bank VALID40 cache and cached paired evaluation
- `bank_valid40_cache.json` (made by `make_bank_cache.py` from the bank side of `paired_valid40_bfix_g05_plain.json`). Per net: final MSE,
  unrounded util (= adjusted/MSE), exact flops, adjusted, wall; pins the bank and scorer sha256. Means: MSE 1.9425e-8, util 0.188858, adjusted 3.6666e-9.
- `paired_eval_cached.py <cand.py> <cache.json> <jobs> <out.json> [ranks]` runs only the candidate side and reports the same fields as
  `fable_refit_v1/paired_eval.py`. Self-test on r37 and r39: candidate rows bit-identical to the uncached run. VALID40 takes about 16 min instead of 31.

### Generic plumbing for a plane correction to the pair map's (3,1) input
`build_t31_variant.py <base> <formula.py|none> <layers> <out>` patches the live loop so that at the chosen layers (subset of 1..14) the
pair map is a copy of `_nonlinear_pair_batch` with `T31 = t4 rho + offdiag(X / (v_a s_a s_c))`, where X = `_t31_extra(l, ctx)` comes from a
formula file. X is a raw-unit correction to K31[a, c] = kappa4(a, a, a, c). ctx is the loop's locals at the pair-map call, plus a
per-network carry dict. An optional `_t31_commit(l, ctx)` runs after each layer's births. The contract is written out in `t31_formulas/TEMPLATE.py`.
The builder refuses formula imports, numpy/torch names and layers outside 1..14.

Verification on rank 35 (`t31_bill_*_r35.log`, exact flopscope totals):

| candidate | final MSE | flops vs bank | charged ops vs bank |
|---|---|---|---|
| bank | 1.648017257382719e-08 | 0 | 0 |
| plumbing off (no layers) | 1.648017257382719e-08 | 0 | 0 |
| X = 0 at L1-14 | 1.648017257382719e-08 | +88,137,728 | +84 |
| direct control: T31 doubled inside the pair map | 2.821351269695783e-08 | +29,360,128 | +14 |
| X = radial * vv_a * Cpre_ac at L1-14 (should equal the direct control) | 2.821351269695783e-08 | +117,526,556 | +112 |
| transposed X (orientation control, must differ) | 2.910092049641958e-08 | +146,857,012 | +112 |

The plumbing costs 6n^2 + 4n flops and 6 ops per enabled layer: +81.8M flops (+0.019% adjusted) for layers 2-14.
Doubling the bank's own K31 plane raises final MSE by 71% on r35, so the (3,1) input is a strong lever in both directions.

One-command service: `bash eval_t31.sh <formula.py> <layers> <tag> [ranks]` builds the candidate, runs smoke and cost on rank 35
(`t31_bill.py`), then the paired VALID40 against the cache, and appends one line to `eval_service.log`. That line includes the exact mean flops change per net.
Self-test with X = 0 at L2-14 on r37 and r39: raw_ratio 1.0, flops change 81,842,176 per net as predicted.

### Cost reference for the physicist (bank, rank 35, `bill_by_function_bank_r35_callers_v2.log`, kernel frames attributed to callers)

| component | share of the 414.0 GFLOP bill |
|---|---|
| (2,1) plane read (`_shared_old384_read` + `_slr_inner`) | 26.2% (~108 GFLOPs, util ~0.049) |
| young/newborn column push | 20.4% |
| old-tier reconstruction | 13.0% |
| star push | 12.2% |
| old-tier push | 10.7% |
| covariance push | 8.1% |

The pushed columns `pp` span the full roster: about 2k columns at L1 and 17-23k at L10-14. A cube-read with the same young/old split would cost
about as much as the existing read. That is +26% bill, so it needs a final-MSE cut above 21% just to break even on adjusted.

### Refit literals on any base
`build_refit_candidate_v2.py <params.pt> <base.py> <out.py>` (torch runtime) exports p/a/c/e (all or none) into any base, e.g. a T31 variant.
Every other .pt key K replaces the base's `_<K>_LITERAL` (same shape; pure literal), and the builder asserts nothing else changed. Tests:
- A .pt holding only `x31` on the zero-table formula base: only `_X31_LITERAL` changed.
- `params_orig.pt` on the bank: same program as the bank.
- `params_orig` plus `x31` on the T31 base: learned literals bit-equal, `_X31_LITERAL` replaced.
- A wrong-shape table is refused and no file is written.

---

## Update 2026-09-22 20:00: physics patch for the stacking order, and the per-row K31 scale
Stacking order set by the lead: bank, then engineer2's `build_e2.py`, then the physics patch, then the engineer's kernel fragments.

### Builder
`build_physics.py <base> <out> <k31row: none | c0,c1[@layers] | table.json> <x31 formula|none> <x31 layers>` makes exact string edits with
count asserts. The same edits apply to the bank and to the `build_e2.py` output (both pair-map T31 forms are recognized).
- **Per-row K31 scale** in the live pair map at configured layers: `T31 = (t4 * r)[:, None] * rho`, with `r_a = c0 + c1 mu_a k3_a / v_a^2`.
  Only the T31 statement changes. A module layer slot set by the live loop selects the per-layer constants.
- **Extra (3,1) plane**, now in place: the pair map gains `extra31=None` instead of being copied, and `T31 += offdiag(X/(v_a s_a s_c))`.
  The formula contract is unchanged (`t31_formulas/TEMPLATE.py`).
- **Newborn handle**: `_PHYS_HANDLES['newborn_common']` = the newborn push's `W[previous,:]^T base[previous,:]`, current at the pair-map call.
- Unconfigured layers run the original statements. `eval_physics.sh <k31row> <formula|none> <layers> <tag> [ranks]` is the one-command service.
- Engineer2 has been told which lines are anchors. The e2 base used here is `base_e2_v5_snapshot.py`, rebuilt from their `build_e2.py`;
  it is byte-identical to their `cand_e2_v5.py` (sha 24f3c465...).

### Checks (rank 35, `phys_bill_*_r35.log`, exact flopscope totals)
| candidate | final MSE | flops | ops |
|---|---|---|---|
| bank | 1.648017257382719e-08 | 414,040,784,973 | 43,563 |
| bank + patch, disabled | identical | identical | identical |
| bank + k31row (1, 0) @1-14 | 1.648017257382719e-08 | +602,112 | +84 |
| bank + k31row (0.85, 0.5) @1-14 | 1.623723963140016e-08 (-1.47%; torch -1.27%) | +602,112 | +84 |
| bank + plane control (X = model K31) | 2.821351269695783e-08 (= direct T31 doubling) | +117,526,556 | +112 |
| e2 base | 1.648017257382719e-08 | 411,629,632,327 | 40,615 |
| e2 + patch, disabled | identical to e2 | identical | identical |
| e2 + k31row (1, 0) | 1.648017257382719e-08 | +602,112 | +84 |
| e2 + k31row (0.85, 0.5) | 1.623723963140016e-08 (bit-identical to the bank version) | +602,112 | +84 |
| e2 + zero plane @1-14 | 1.648017257382719e-08 | +88,137,728 | +84 |
| e2 + plane control | 2.821351269695783e-08 | +117,526,556 | +112 |

### Per-row K31 scale, provisional constants (c0 0.85, c1 0.5, layers 1-14): paired VALID40 against the cached bank rows
`paired_valid40_phys_k31r_085_05_L1_14.json`: adj_ratio 0.98606, raw_ratio 0.98605, adj_wins 38/40, util 0.188863 vs 0.188858.
The mean flops change is +11.7M per net, because selections move slightly on some nets. Per-net raw change: median -1.03%, range -3.84% to +0.22%.
Candidate: `cand_phys_k31r_085_05_bank.py`.

---

## Update 2026-09-22 21:10: stacked base = bank -> engineer2 build_e2 (v5) -> K31 row factor (PARKED, not fired)
The lead confirmed the fixed row-factor constants: c0 0.85, c1 0.5, layers 1-14. The physicist's per-layer schedule was less robust (35/40 wins).
- **Candidate:** `cand_stack_e2v5_k31r.py` (sha fbee9331...) = `build_physics.py base_e2_v5_snapshot.py ... 0.85,0.5@1-14 none ''`.
  The e2 base is engineer2's `build_e2.py` output at v5 (sha 24f3c465..., byte-identical to their `cand_e2_v5.py`), because engineer2 has not yet announced a final builder.
- **Check 1, MSE bit-equal to the bank-based physics variant:** r35 1.623723963140016e-08 and r100 1.660905741633846e-08 (bank r100 1.680305e-08, so -1.15%);
  all 40 VALID40 nets bit-equal as well.
- **Check 2, flops:** r35 411,630,234,439 (bank-based variant 414,041,387,085); r100 414,446,332,502 (416,857,482,588). VALID40 mean util 0.187767 vs
  bank 0.188858, i.e. the bill is 0.578% lower (-2.40 GFLOPs per net).
- **Check 3, paired VALID40 against the cached bank rows** (`paired_valid40_stack_e2v5_k31r.json`): adj_ratio 0.98034 (-1.97%), raw_ratio 0.98605,
  adj wins 40/40, raw wins 38/40. Per-net adjusted change: median -1.59%, range -4.41% to -0.32%.
- **Row factor, native vs torch on VALID40** (`k31row_native_vs_torch_valid40.txt`, physicist's rows): pooled 0.98605 vs 0.98565; per-net correlation 0.953;
  difference sd 0.33%, the known native-vs-torch per-net noise. The two native raw losses (r37 +0.22%, r199 +0.16%) are nets where torch gains only 0.19% and 0.10%.
- **Ready for engineer2's v7** (`base_e2_v7_snapshot.py`, sha 737bdcc9..., their current `build_e2.py` output). Their hold patch changed the T31 line to
  `_e2_hold.append(fnp.multiply(...))`; `build_physics.py` now recognizes that form (earlier outputs regenerate byte-identically). On r35, v7 with the
  patch disabled equals v7 exactly (flops 411,629,632,327, ops 40,615), and the row factor on v7 gives 1.6237239631400155e-08, the same as on the bank and on v5.
