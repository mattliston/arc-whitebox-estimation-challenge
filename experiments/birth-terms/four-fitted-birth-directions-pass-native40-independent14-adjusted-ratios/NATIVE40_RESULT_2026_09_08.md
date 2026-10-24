# Fixed terminal birth-direction fit — native40 result

Phase 2 only. Judged 2026-09-08 23:08:24–23:08:29 UTC. This is the
existing 40-network development VALID split against existing 1e9-sample
truths, not the independent14 panel or an official submission.

The single preset six-epoch fit was not extended, refitted or selected by
validation loss. Terminal coefficients and all40 MPS predictions were
sealed before validation truth access. All three native acquisitions use
official-owned FNP weights, their own complete closed bills, initialization
once and a byte-identical repeat excluded from the score.

| Configuration | Mean raw MSE | Mean adjusted | Mean complete FLOPs |
|---|---:|---:|---:|
| Current409f bank | 2.246383679135106e-8 | 4.766355577729645e-9 | 466876867515.125 |
| Zero birth-direction action | 2.246383679135106e-8 | 4.781115190719337e-9 | 468321676751.725 |
| Frozen terminal action | 2.12989435468549e-8 | 4.533103638013092e-9 | 468321676751.725 |

Adjusted is the mean per-net raw MSE times max(0.1, that net's complete
FLOPs/2^41), not the product of the two means. No independent4M noise
subtraction is used in this development frame.

Terminal/bank raw ratio0.9481436205526269, adjusted ratio0.9510628328263212:
5.1856% raw and4.8937% adjusted reduction. Terminal/zero adjusted ratio
0.9481268401172048. There are38/40 raw and adjusted wins against BOTH
controls, passing the prospective24/40 and mean-adjusted-gain gates.
The two losses are ranks146 and150. No post-hoc tuning or reweighting.

All40 zero-action outputs reproduce the bank bytes. All53 qualified old
v1 overlaps reproduce their output bytes; new actual bills are179306496 F
higher per call because the official weight preparation is now paid.
The old failed/partial acquisitions are preserved, comparison-only and
never spliced into this score. Harness-only public log reset occurs after
closed bills are saved; no candidate-state or in-predict budget reset.

All native calls passed8GB/120s. Process peak RSS: bank7.511GB,
terminal7.314GB, zero7.556GB. Native residual timing is diagnostic only;
production0.4s residual qualification remains required before any bank.

Result SHA:884089500b2f7f31cc9f1fbe343c6396df3cf976daee63141915715f8064c29f
in terminal_score_v2/result.json. Pre-truth prediction/bill seal SHA:
c0a8f6e6e62503be7d6a7c465050a95e1a58dd517c75a76849d2ace5001faf51.
Terminal estimator SHA:
d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451.

Decision: admit the fixed independent public8+virgin6 comparison, with fresh
bank AND terminal official-input bills. Preserve this as a possible accuracy
component if both panel groups improve. No milestone, release or new bank
has yet been established; official bank remains#329926 at4.982231460183304e-9.
