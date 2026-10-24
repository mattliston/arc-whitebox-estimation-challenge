# P2R0C1D5 post-Fable CP3 Pareto audit receipt

Status: **CONDITIONAL_OPEN_PREREG_READY_ACCURACY_COMPUTE_HYBRID**  
Date: 2026-08-26  
Graph parent: `P2R0C1D`  
Proposed child: `P2R0C1D5`  
Canonical `active_work`: **empty**; no implementation or heavy gate is active

## Terminal verdict

Exactly one materially new target-free route survives this audit:

> Apply a genuine depth-4 Strassen--Winograd scalar-operation reduction only
> to the 58 deterministic CP3 dense contractions in the exact `#328439`
> estimator, retaining the complete rank-15,360 carrier and every consumer.
> Batch the two channels only at preregistered memory-safe ranks and run them
> sequentially above those ranks.

This is not ready to deploy. It is preregistration-ready and has passed a tiny
synthetic bill/algebra discriminator, but no production candidate exists and
no cold meter, full prediction-drift roster, target score, or board result has
been run. The route changes float32 summation order and therefore is an
accuracy/compute hybrid, not an exact-output-parity cut.

The honest fixed-protocol static result is:

| quantity | exact value |
|---|---:|
| incumbent CP3 contraction bill | 966,086,623,232 |
| depth-4 Winograd kernel bill | 594,693,332,992 |
| memory-safe batching pack bill | 20,971,520 |
| candidate CP3 bill | 594,714,304,512 |
| **net scalar-operation saving** | **371,372,318,720** |
| conservative maximum candidate RSS | 7,902,887,936 bytes (7.3601 GiB) |
| depth-5 sequential conservative RSS | 9,353,728,000 bytes (8.7113 GiB; closed) |

The earlier exploratory `512.147B` depth-7 figure is rejected. It counted
only arithmetic multiplications/additions for classic Strassen, omitted
FlopScope's stack/block materialization, and did not satisfy the memory bound.
Classic Strassen also has five B-side pre-adds, not four. None of those early
numbers is a candidate claim.

## Exact standing-source and board binding

The source is
`public_noeigh_card_v1/estimator.py`, SHA-256
`373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7`.
Its build receipt binds that output hash. Its immutable watch line records
submission `#328439`, graded adjusted
`2.2842888692472472e-8`, raw `2.9409159800763974e-8`.

Static AST verification finds:

- `_CP3_CAP=16384`, `_PRODUCTION_WIDTH=1024`, `_PRODUCTION_DEPTH=16`;
- exactly two `fnp.matmul` calls in `_cp_push`;
- exactly two `fnp.einsum("ir,jr->ij",...)` calls in `_cp_read`;
- the separate final `_cp_k3` matrix-vector operation, explicitly excluded;
- loop layers 1--15 with the final branch before `_cp_read`.

The full per-layer shape/call roster is frozen in `PREREGISTER.md`. It totals
30 push products and 28 read products. The read calls are GEMM-equivalent
contractions, not source-level `matmul` calls.

## Why the bill is actual work rather than TD52 pricing

The installed, hash-bound FlopScope sources establish all parts of the bill:

- `_flops.matmul_cost` delegates to canonical einsum pricing and explicitly
  returns `2*m*k*n-m*n` for a 2-D product;
- the live `fnp.matmul` wrapper routes ordinary/batched products through the
  same einsum cost engine;
- `stack` and `block` set cost to output element count;
- default weights for `matmul`, `einsum`, `add`, `subtract`, `stack`, and
  `block` are 1.0, and the float32 dtype rate is 1.0;
- all indexing in the kernel is basic integer/slice indexing, which the live
  ndarray wrapper treats as free views.

The recursive cost includes all 15 Winograd additions and all stack/block
copies at every node. It does not use `tensordot`, high-rank singleton axes,
TD52, a symmetry discount, a dtype discount, or a custom weight. A post-prereg
synthetic depth-2 `8x8 @ 8x16` execution under the installed meter billed
exactly 3,596 operations versus the independently computed 3,596. Its maximum
absolute float32 difference from ordinary NumPy matmul was
`4.291534423828125e-6`, confirming that real-arithmetic identity is not
floating-point parity.

## Resource and numerical bounds

The incumbent common-driver meter's three calls `[35,279,35]` had maximum
outer wall `24.954987000208348` s, residual `0.35143496841192245` s, and
process RSS `3,421,503,488` bytes. They are context only: that meter predates
the candidate and cannot certify it.

The conservative depth-4 workspace schedule counts retained recursive operand
stacks, deepest dependency temporaries, the leaf result, recombination, output
banks, the extra paired-read buffers, and the duplicated common push input.
The frozen batching thresholds (push through rank 10,240; read through rank
11,264; sequential above) keep the algebraic maximum at 7.3601 GiB with
687,046,656 bytes of 8-GiB margin. Depth 5 exceeds 8 GiB even sequentially and
is closed.

Wall and residual cannot be proved statically. The trim receipt's paired
18.8-ms median improvement per roughly 1,100 removed Python dispatches makes
the proposed roughly 2,731-wrapper depth-4 schedule plausibly near the
0.4-second wall, not safely below it. Therefore the first production action
must be exactly three cold no-target common-driver calls, with every residual
strictly below 0.4 s and the maximum <=0.385 s for deployment readiness. Any
failure closes this implementation form.

The candidate changes association. Its no-target prediction gate is final RMS
delta <=`1.714909904361275e-6` and max absolute delta
<=`8.574549521806375e-6` against the exact incumbent on a prebound weights-only
roster, plus candidate repeat byte parity. These thresholds are scaled from
the board aggregate RMSE but do not reveal or prove target accuracy.

## Score implication: projection only

The official score applies the multiplier per row before aggregation, so the
board aggregate adjusted/raw ratio is not an exact common FLOP measurement.
For planning only, treating that ratio as an equivalent constant-F multiplier
and subtracting the static net saving gives:

- projected multiplier `0.6078464296189547895`;
- same-raw projected adjusted `1.7876252782987673e-8`;
- raw break-even for adjusted `<1e-8`: about
  `1.6451523794042476e-8`;
- required raw reduction from #328439: about `44.0599%`.

These are projections, not evidence. The receipt does not infer that raw MSE
stays unchanged, improves, or meets the threshold. Exact candidate comparison
requires literal per-row costs and a later target-authorized score.

## Dedupe and closures

- Fable/Phase-1 Strassen sources are bound below. They apply classic
  7-product Strassen to the stochastic sample-forward path, generally at
  depth 3. They neither touch the deterministic Phase-2 CP3 P/Q push/K21 read
  roster nor use the 15-add Winograd schedule. They are numerical precedent,
  not Phase-2 evidence.
- TD52/tensordot singleton-axis families are explicitly excluded; this route
  reduces the actual scalar algebra under ordinary matmul pricing.
- t7 trimming and no-eigh are already banked in #328439 and are the parent,
  not candidate savings.
- CP3 caps, carrier drops, and rank truncations change the represented state;
  existing evidence reports multi-error-scale output changes. This route keeps
  all births and ranks.
- CP4 halving, star rank-64, corrector trims, and other small component cuts
  do not have this ceiling and are not silently composed here.
- The three most recent accuracy-science receipts bound below close IG-LC3 as
  a production exact reset, close covariance-resummed sparse base under the
  present TT16 circuit absent an external certificate, and close whole-tensor
  CP8/16. None supplies the missing 44.1% autonomous raw improvement, and none
  is being combined into this one candidate.

## Bound authorities

All paths are repository-relative. `verify_receipt.py` recomputes every hash.

| authority | SHA-256 |
|---|---|
| `public_noeigh_card_v1/estimator.py` | `373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7` |
| `public_noeigh_card_v1/BUILD_RECEIPT.json` | `3fd515406b6eec954523e7d553aba769999998318fb29fd987ce4637e2bbb0d3` |
| `public_noeigh_card_v1/SUBMIT_WATCH.log` | `6f2339963c1272d9eee46e20897b06fc87f5b321126e2e1976a565df71f3c781` |
| `public_q24_native_transport_energy_final_workspace_v4/METER_RESULT.json` | `b6aedc76b190702ac0b4bcf2c567d40e0f533b861ccde95bcb17ac21627eb365` |
| `public_trim_card_v1/PAIRED_RESIDUAL_RECEIPT.json` | `2eb85dba23858669b532887c9c7d774bf8f9581fec5609ba4133b6976c28b777` |
| `public_adjusted_score_pareto_v1/RECEIPT.json` | `12e82f5b952c6b2786c834f028360cb25bea483f6ee0261a6ee1b198d4ff81e2` |
| `public_adjusted_score_pareto_v1/ERRATUM_001_EXECUTABILITY.json` | `042ec26fa003dbe2fda47ef69b801104d6e35e46198a6927a6eacc600ac8f8a0` |
| `public_host_compute_value_attribution_v1/V_ATTRIBUTION_RECEIPT.json` | `7a44a817d3f5de9e2c7b5ee0c0b9e68277ceb3e0408d9750dae26db2437e3e06` |
| FlopScope `_flops.py` | `bbdad83348b7ce7b63c9d8da66512c40c69e055c653d24687e62132bd34220c2` |
| FlopScope `_array_ops.py` | `e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c` |
| FlopScope `_pointwise.py` | `7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7` |
| FlopScope `_dtype_billing.py` | `d6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8` |
| FlopScope `_budget.py` | `ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01` |
| FlopScope `_einsum.py` | `0f25b8f5709c5a96404f20c38afc9b705568ab2654ab6c5fc81f3de292b73284` |
| FlopScope `_ndarray.py` | `fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74` |
| FlopScope `default_weights.json` | `f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a` |
| Phase-1 `327723/estimator.py` and Fable clean Strassen source | `c3c6981b4687724e8bbabf55c8103586b589bfbf0b2755bf3e71a584fef64ec9` |
| Fable spend Strassen source | `4dee5200cb0c1c72eb5e9c9d8c0cecc989423363fdcd08b232e0e6bf4ead7820` |
| Fable spend7 Strassen source | `f6dc80663b9e35743e48141796e21087861bd1ea2e83015c3ebab38b9616b6d6` |
| IG-LC3 constraint receipt | `d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba` |
| covariance-resummed sparse-base receipt | `41cecf538bee6a3feadcea88c8d82c35ed4489756bc500084118d11b7b16f561` |
| symmetric-CP prior audit | `1683ddb0400767516f74f3eed68e3f23d43ed71f6685fd2493dbf55a99026594` |

Local artifacts at seal time:

| artifact | SHA-256 |
|---|---|
| `PREREGISTER.md` | `5b7d8d778c684082cc07b04040f7e9e4b9fc8d60c1ab6298aaa7703e36306740` |
| `verify_receipt.py` | `e2dfbb7a35f460e37a8d590cd773f0c6954a734221dc15872446febceb0f9bc3` |
| `STATIC_VERIFICATION.json` | `2f5147738874d637e9b1f7d4623d6ff87ca6dfd415a4fc529ab87e5d7a7aaf53` |

Reproduction:

```bash
python3 pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_receipt.py
pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/python pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_receipt.py --micro
```

The first command must return `PASS_STATIC`; the second must additionally
return `PASS_TARGET_FREE_MICRO`. Neither command reads targets.

## Graph-ready disposition

Add `P2R0C1D5` as a conditional child of `P2R0C1D` with status
`open_preregistered_unexecuted`, evidence this receipt, and gate summary:
"depth-4 CP3 Winograd; static net save 371.372B; production RSS/residual/drift
unmeasured; same-raw adjusted projection 1.788e-8; still needs about 44.1% raw
gain." Do not place it in `active_work` until an implementation/meter run is
explicitly launched.

