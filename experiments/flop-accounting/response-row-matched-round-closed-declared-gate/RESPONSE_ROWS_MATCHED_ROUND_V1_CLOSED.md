# Response-row matched round: closed at the declared gate

Completed September5,2026 PDT, according to the actual clock/file mtimes.
All200 honest rows completed in557.6seconds. Their one-step profile grows
from4.8525e-9 atL10 to7.1922e-9 atL15, without a current-target leakage cliff.
One fixed160/40,C64,40-epoch MPS fit completed in446seconds; held-out final
one-step MSE5.77077179376e-9 versus parent5.70353224649e-9.

Full-final learned recursion against the same stored1e9 targets:

- Rank35:2.10823951686e-8 versus2.04444474536e-8, +3.1204%.
- Rank100:1.99700304672e-8 versus2.00406278849e-8, -0.3523%.

Rank35 exceeds the predeclared maximum3% raw worsening. The separate
unlearned rank35 metered cost saving5.85% does not establish a competitive
learned adjusted result. No threshold sweep, final-head refit, candidate
port,14-net panel, or submission. Preserve rows, model, payload and receipts.

Payload:`payload_response_rows_v1.npz`; model tag
`ASTRA_RESPONSE_ROWS_V1_C64_E40`; training receipt
`chain_response_rows_training_v1.json`; full-loop receipts
`response_rows_closedloop_r35_v1.json` and `response_rows_closedloop_r100_v1.json`.
