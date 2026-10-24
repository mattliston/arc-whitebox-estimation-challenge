# Phase 2 continuation receipts — September 4 PDT

Previous turn made concrete progress: full14-net, native, and remote gates
completed for the D5 full-bank candidate. Its upload remains unapproved after
automatic permission review rejected disclosure of the code to AIcrowd.
There has been no retry or workaround. Goal~4e-9 remains incomplete.

Development raw MSE below always means rank35 / rank100, width1024depth16,
against stored1e9 post means. These are screens, not the required14-net gate.

## Closed bounded variants

- Signed-pair cap4096, honestly coadapted200rows, fixed160/40 split,
  C64,40epochs: closed-loop5.53288e-8 /5.40290e-8, versus full-bank learned
  2.21231e-8 /2.11385e-8. Unlearned utilization.184676, but raw loss overwhelms
  saving. No panel or upload warranted.
- Low-rank CP plane READ, retaining all factors and exact scalar diagonal:
  exactSVD rank64/128 both fail. Removing the diagonal before SVD does not
  fix it: rank1289.7903e-8 /1.1302e-7 versus full host2.7972e-8 /2.9453e-8.
  This closes this read-compression form, not an impossible plane transport.
- Physical activation probability instead of standardized gate in GNN
  messages, same frozen200rows and40epochs: held-out one-step5.93605e-9
  versus6.01057e-9, but closed-loop2.25120e-8 /2.15028e-8 is worse than
  original2.21231e-8 /2.11385e-8. Not promoted. Inference and trainer match.
- Half precision has no billing advantage: installed official FlopScope.12
  dtype rates give float16=float32=1, float64=2. No compute experiment run.
- Complete binary cubic `3 lambda P^2 Q + kappa Q^3` represents negative
  diagonal remainders using the same two factor banks and two rank-N plane
  products. Algebra certificate passed. Full unlearned2.79799e-8 /2.95247e-8
  is effectively unchanged versus two signed cubes. Discriminant clipping
  was not an important raw-error source on these two nets.

## Compact-neuron path: measured useful cost lever

`radial_binary_gateprune.py` first tested dropping cold cross-neuron state
while keeping all means and all univariate diagonal moments. Gate<.001
gave2.79629e-8 /2.94772e-8, effectively unchanged. Gate<.01 gave
3.25649e-8 /3.42971e-8. These first dense emulations were not cost claims.

`radial_binary_compact.py` is the actual compact reference: active and birth
counts round UP to32, only active-active CP plane entries are read, full
factor pushes permit exact scalar third-moment reads for every neuron,
and cold covariance/star diagonals remain. At .001 raw2.79925e-8 /2.94442e-8;
at .01 raw2.93709e-8 /3.32566e-8.

FlopScope port at .01: `candidate_binary_compact_g01_unlearned_v2.py`.
Rank35 raw2.93826e-8, utilization.233989, wall23.85s, matching its reference.
It uses active input blocks plus exact cold diagonals in covariance/star
pushes, and padded compact matrix products. Initial v1 tried unsupported
indexed writes and failed before a full run; v2 uses documented, metered
`fnp.put` scatter operations. That is a normal public API, not a bypass.

Keeping full covariance/star state but pruning only the multi-step cubic
carrier (`radial_binary_sparsecarry.py`) gives at gate.01, birthfloor.001:
2.75715e-8 /2.90177e-8. Larger gates .05 or .1 lose too much raw before
coadaptation. This variant is not metered or coadapted yet.

Cold-state diagonal-shadow experiment: preserve the next-layer diagonal
reads of omitted covariance and cubic-star matrices. At gate=birthfloor.01,
2.94334e-8 /3.16415e-8 is a small mixed gain versus compact. At .05 both
thresholds, raw1.455e-7 /1.276e-7. This stronger arm also removes many nearly
linear POSITIVE births; it is not a clean test of stronger negative-gate
pruning alone. No further shadow implementation undertaken yet.

## Exact Winograd optimization

`build_direct_transform_v1.py` writes each operand's four intermediate sums
directly into its child-transform slots. The old implementation used four
sums then seven copies; the new one uses four sums plus three copies. The
input and output parity buffers are disjoint in the existing audited arena.
No arithmetic order or approximation changes. Tiny D6/D3 tests pass with
exactly the previous numerical errors.

Actual numpy-free client/server, full learned D6/4096:

- New candidate `candidate_binary_learned_d6_direct_v1.py`.
- Utilization .298482952 versus .312744994, a4.56% saving.
- Wall65.78s under concurrent load, residual.286452587s (passes.4s).
- Output SHA256 `1aaefe9b72de8b60f8aacb2e776daa566cae57bf13f68b4456de9022b7e500ad`
  is IDENTICAL to the old D6/4096 remote output.
- Source SHA256 `31777abfd8b0b12b0bb192960d74513c1be8e085ab296e01bee1dca7b262ac6d`.
- No14-net run or new peak-RSS audit for this exact version yet.

Composed with compact-neuron .01, unlearned D5:
`candidate_binary_compact_g01_unlearned_direct_v1.py`, rank35 raw2.93826267e-8
EXACTLY unchanged, utilization.228172300 versus.233989185. Not a submission
by itself; it needs a matched learned head and all resource/panel gates.

## Currently running at this entry

`chain_binary_compact_training_v1.py --mode compact`, exec session79845,
completed200rows in358.5s and entered one MPS training job. Honest one-step
profile L10=5.62167e-9, L15=8.44578e-9. No current-layer target leakage.
Expected payload `payload_binary_compact_g01_v1.npz`, tag
`ASTRA_BINARY_COMPACT_G01_V1_C64_E40`, fixed160/40split,C64,40epochs.
Revalidate the live handle before waiting or taking further action.

No other experiment process remains running at this entry. No commits,
promotion, historical-file edits, or deletion. Latest checked swap6482MiB,
unchanged. Actual date September4 PDT despite September5 UTC after midnight.
