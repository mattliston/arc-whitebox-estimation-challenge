# Phase 2 receipts — September 4, 2026 PDT, evening

Dates follow the actual clock and file mtimes. No Phase 1 work, commits,
promotion, frozen-file changes, or deletion. The goal remains approximately
4e-9 adjusted. Upload approval is still outstanding; there has been no retry
or workaround of the rejected upload.

## Compact .01: fully validated improvement

`candidate_binary_compact_g01_learned_direct_v1.py`, source SHA256
`84c76ba218410a9b6b2bfbd61c27b35e9cffc633318de52b9cbb948df081648c`.

Matched 200 honest rows, fixed160/40 net split, C64,40epochs, mean head only.
Held-out one-step final MSE6.39339e-9 versus host8.095e-9 (21.0% removal).
The completed training payload's actual mtime is September4,17:57:11 PDT.

Closed-loop development raw against stored1e9 post means, rank35/rank100:
2.2929837e-8 /2.3413942e-8. Full-bank learned reference:
2.2123083e-8 /2.1138533e-8. This modest raw loss buys a large compute saving.

Full independent4M noise-corrected14-net panel:

| Group | Raw MSE | Mean utilization | Adjusted |
| --- | ---: | ---: | ---: |
| Public8 | 2.6961976e-8 | .23010 | 6.2055707e-9 |
| Virgin6 | 2.3396607e-8 | .22818 | 5.3523087e-9 |

Versus the previous fully validated D5 full-bank card on identical panels,
adjusted decreases23.08% /25.60%. Versus the official-best329790 card's
identical local panels, decreases61.33% /58.88%. These are LOCAL scores,
not an official graded submission. The official best remains1.4369934e-8.

Native official-version35/100/35 gate passes: maximum mean deviations
1.2611e-6 /1.1714e-6 against independent NumPy; learned delta deviations
3.45e-8 /2.45e-8; repeated rank35 byte-identical. PeakRSS6,798,016,512bytes.
Native residual time under diagnostic copying is not the official gate.

Actual numpy-free local client/server: utilization.230770826, wall35.13s,
residual.174879970s, max childRSS6,624,051,200bytes. All resource limits pass.
Receipts `binary_compact_g01_learned_native_gate_v1.json` and
`binary_compact_g01_learned_direct_remote_v1{,_resources}.json`.

## Stronger cold pruning: bounded branch closed

`binary_asymmetric_v1.json` separates the gate threshold from birth pruning.
Birth floor stays.01. Development raw rank35/rank100:

- Gate.03:6.39236e-8 /6.33339e-8; exact next-layer covariance/third-moment
  diagonal shadows improve to5.78605e-8 /5.82215e-8, still far worse.
- Gate.05:1.43146e-7 /1.74671e-7; with shadows1.33226e-7 /1.16228e-7.

These results do not justify a port or matched retraining of this form.
They close the earlier confound that near-linear positive birth pruning
alone caused the failure. They do not assert universal impossibility.

## Two bounded cost follow-ups

1. Audited direct-write D6/4096 rectangular products with exact zero padding
   for active counts that are only multiples of32. All masks and learned
   parameters frozen. `candidate_binary_compact_g01_learned_d6_v1.py`:
   native rank35 raw2.2937990e-8, utilization.222519884 versus D5~.2307.
   Actual client/server resource check running at this entry. No full panel
   verdict yet for this exact version.
2. Scalar-only final read. Layers1..14 and their learned correction remain
   frozen. Refit only the final head on its own prefix's states, using scalar
   features plus the existing recurrent messages. This permits omitting the
   final full CP plane, off-diagonal covariance output, and star plane reads.
   Not a reopened iterative DAgger program: the older program's flat result
   was checked in T-116j/T-117e before this cost-motivated design.

`chain_compact_final_head_v1.py` runs three nice10 CPU row workers then one
MPS fixed40epoch final-only fit,160/40 same net split. Every final input is
captured before current correction; no truth feedback anywhere. Targets
are read only after all inputs and recurrent states have been frozen.
Live exec session24634; inspect its handle before acting. Reference and
FlopScope builders are versioned separately for independent parity checks.

## Completed scalar-final round and current status at 18:33 PDT

The200 on-policy final-only rows completed in356.6s. The original trainer
failed before epoch1 because its composed output bias inferred float64,
unsupported on MPS. Original script preserved; the versioned resume wrapper
adds explicit float32 only. It completed the same fixed40epoch fit and
wrote `payload_compact_final_head_v1.npz` (mtime18:16:04PDT). Held-out40 raw
MSE2.5778075e-8 versus original final head2.5609225e-8, a0.66% loss.
Clamped prefix and original-final predictions match the frozen reference
BYTE-for-BYTE. The initial unclamped comparison differed only at negative
outputs that the estimator clips; this was checked explicitly.

Independent scalar-final reference development raw2.3340556e-8 /2.3494745e-8.
It reproduces frozen full-matrix host means within1.041e-7 and its final
inputs within8.53e-6. FNP D6 rank35 raw2.3350245e-8, utilization.214039197.

`candidate_compact_scalar_final_d6_v1.py` full14-net independent4M panel:

| Group | Raw MSE | Mean utilization | Adjusted |
| --- | ---: | ---: | ---: |
| Public8 | 2.7222063e-8 | .21485 | 5.8500076e-9 |
| Virgin6 | 2.3517122e-8 | .21327 | 5.0310490e-9 |

Native35/100/35, INCLUDING both prefix and separate final-head parity,
passes: max mean1.5382e-6, final correction4.406e-8, final input.002009;
repeated rank35 byte-identical; peakRSS7,147,962,368bytes. Actual numpy-free
client/server utilization.214135089, wall70.03s under concurrent load,
residual.272378087, max childRSS6,383,304,704bytes. All resource limits pass.
Source SHA256`69e1ef2cf0828a5bd3567007b6981a97bfcca95302a760b95d0e0b6d362d62c1`.
This is the current best fully validated LOCAL result, not a graded score.

The previous compact D5 package was locally built and inspected: only
estimator.py and manifest.json. Official tiny plumbing validate passed.
No candidate has been uploaded since the permission rejection.

Early single-cube schedule, restoring current repeated entries into star:
first layer only raw3.39770e-8 /3.34568e-8; first two4.78951e-8 /4.69640e-8,
versus compact full two-cube2.93709e-8 /3.32566e-8. Not worth a port or
coadaptation at this cost/accuracy tradeoff. No global-family reopening.

## New raw lever: covariance-radial K4-to-K3 interaction

The derived quadratic-vertex source is
`d_i*g_j*g_k*Q_iijk/2`, with`Q=alpha*Sym(C,C)`. It separates into a changed
covariance-column cubic coefficient and a cheap vector-covariance tensor
`Sym(b,C)`. Small full-tensor algebra certificate passes. Earlier testing
used the weaker mixed-cubic host; the new two-cube composition explicitly
subtracts this tensor's repeated plane BEFORE fitting the binary diagonal
innovation, preventing double counting.

Natural-coefficient full-bank raw2.6223121e-8 /2.4716687e-8 versus full-bank
unlearned2.797178e-8 /2.945288e-8 (roughly6% /16% better).

Compact composition uses`Sym(b,B)`, with`B`the active covariance block,
cold diagonals excluded and b supported on active units. Its linear push
reuses the already-required active covariance product. The current repeated
plane is restored exactly by the compact star. Natural-coefficient compact
raw2.7708041e-8 /2.7820710e-8 versus2.9370859e-8 /3.3256605e-8.

FNP unlearned implementation`candidate_compact_radialfeed_unlearned_d6_v1.py`
matches: rank35 raw2.7711482e-8, utilization.220240519. The independent
reference deliberately computes its metric separately; FNP reuses the
same active covariance push, with no additional cubic-cost matrix product.

Current live handle91416: `chain_compact_radialfeed_training_v1.py`.
Three nice10 CPU workers generating200 honest rows, then ONE MPS fixed
160/40split,C64,40epoch matched closure. Expected
`payload_compact_radialfeed_v1.npz`; G22 model tag
`ASTRA_COMPACT_RADIALFEED_V1_C64_E40`. No verdict before coadapted closed-loop
development, then the required14-net gate. Last checked swap6474MiB stable.
