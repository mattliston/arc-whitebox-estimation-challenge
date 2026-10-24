# Fixed response-row exception: development gate passes

September6,2026 PDT. This is the ONE no-refit re-evaluation authorized in
RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md, not a reopening of other families.
Both fresh native acquisitions and the subsequent sealed score process
completed. The candidate is unchanged SHA256
5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172.

The same fixed ranks35/100, same stored1e9 truth means, same native
FlopScope0.12/Whest0.16 configuration, and actual served float32 outputs:

| Net | Bank raw | Response raw | Bank FLOPs | Response FLOPs | Bank adjusted | Response adjusted |
|---|---:|---:|---:|---:|---:|---:|
|35|2.0731001663431382e-8|2.1099028082828634e-8|490466018197|476670806147|4.623803688036139e-9|4.573526314361569e-9|
|100|2.0592545939233167e-8|1.997488978344223e-8|489374517341|473227916215|4.582701526412439e-9|4.298579128245664e-9|

Mean of the per-net raw*max(.1,F/2^41) products:
4.603252607224289e-9 bank versus4.436052721303617e-9 response,
ratio.9636778816660484 (3.63221% lower). Mean raw ratio.9939591407109105;
mean complete-estimator FLOPs489920267769 versus474949361181.
Rank35 raw worsens1.775%, but its adjusted score improves1.087%.
This is exactly why the new adjusted-objective gate differs from the
old raw-only closeout. That old receipt remains unchanged.

First-call initialization is included identically in the comparison;
the third35 call is excluded from the two-net mean and reports warm cost:
bank489082038735 FLOPs, response475286892481. Both repeats are
byte-identical. Response/reference output max errors on35/100 are
2.14281e-6/2.55457e-6; head-delta max errors1.54178e-8/3.46086e-8,
within the unchanged1e-5/1e-6 gates. All16 features/heads were captured.
Native peak RSS6.610321408GB response,6.757580800GB bank; finite proper
nonnegativefloat32 outputs. Source/input hashes and seals pass.

Independent agent audit recomputed all score scalars and verified text/
JSON source, bill, provenance and chronology without decoding arrays.
Outputs and complete bills were sealed before true1e9 target access.

Qualification: threadpool_limits(1) was requested, but both reported
threadpool inventories are empty. That does not attest effective BLAS
thread count on Accelerate. No explicit BLAS environment variables were
passed to these native commands. Their timings are NOT a production
one-thread or residual release pass. This does not alter the measured
raw/FLOP comparison in the matched native frame. The separate persistent
NumPy-free/OpenBLAS test explicitly pins one thread. The14-net panel will
also record explicit BLAS environment settings; no numerical rerun or
tuning of this fixed development test is warranted.

Score receipt: response_rows_adjusted_reevaluation_v1/score/result.json,
SHA256 a5e4a8aa64968218931d3b11aa80141580371ec17931f7a1532b7381ab5437d0.
Response acquisition resultSHA
b65e248340c4bc3dd37948254a150fe28e75be906d0d133215b30d45996f511e;
bank acquisition resultSHA
9edd2b28c5818ca7fa8e8b07eb70942190215ff97a5469caa4e07743c89dfb34.

Decision: advance unchanged source to the independent8+6 panel and retain
all numerical/resource/release checks. This is development evidence only,
not a validated component, milestone bank or official score. Bank#329926
remains4.982231460183304e-9. No submission or model fit occurred.
