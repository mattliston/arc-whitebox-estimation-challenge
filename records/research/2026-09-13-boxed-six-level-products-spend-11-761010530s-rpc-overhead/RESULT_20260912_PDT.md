# Actual unchanged-BOX timing attribution

September12PDT/September13UTC. One cold canonical37, not a new estimator,
runtime gain or score. Candidate378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa.
All16 output bytes equal frozen BOX/c2. Complete407741701711F and49590
operations reconcile exactly to the cached closed-context summary. No new
array operations or RPC requests were added by retaining that summary.

Actual context51.040018208s. Native operation backend34.962000845s;
native operation overhead3.622005700s; signed remaining RPC-compute gap
0.470171730s; reported RPC overhead11.761010530s; participant residual
0.224829403s. These sum to wall exactly at recorded precision. Native
overhead is inside RPC compute, not a subset of reported RPC overhead.
No causal label is assigned to the signed gap.

| Native category | Backend seconds | Calls | Complete category FLOPs |
| --- | ---: | ---: | ---: |
| subtract |10.702534650|14712|13684907853|
| add |9.168836600|13737|16765395469|
| matmul |7.658592069|653|346059966563|
| stack |3.020096193|3590|7875370254|
| multiply |2.566753210|4966|7233291747|
| copyto |0.500006972|1807|1276720986|
| astype |0.365096527|660|1354044808|
| QR |0.177730168|10|4454350848|

Adds/subtracts account for19.871371251s of the34.962s native backend.
This supports investigating transform additions/layout and call structure;
it does NOT establish per-dispatch or per-shape shares, physical bandwidth
saturation, or a specific future saving. QR and small pointwise changes
alone cannot account for the6.04s gap to the prior45s gate. No hidden serial-D6
READ path is inferred; the complete40 source/ledger reconciliation found none.

Whole watch52.764063792s,4866834432B maximum combined sampled RSS,
0.354444542s maximum completed observation interval,0 added swap. The client,
server and observation groups retired; no error or cleanup failure. Prior
BOX runtime advancement and official330840 failure remain unchanged.
Official bank330390 remains4.53886349950395e-9.

Actual result/completion/finalization:

- ed1cd38a7bd0822e233295bd06ddce6b303ccfc07ef6f2be1fbc7a50be3b55fc
- b88faae4d1acd2a70e96c8bb0480bad08f2b125be3a88450ef98912f5929c5f2
- ff4bb51426a9a800513c7bd96f2adea08948606e64be208559dd8c722c6644a6

Next decision: source-check a concrete transform scheduling/layout mechanism
against prior closes and complete cost before selecting another runtime test.
The fixed current-head joint-moment diagnostic continues independently from
its positive target-blind capture; no new raw score is reported here.
