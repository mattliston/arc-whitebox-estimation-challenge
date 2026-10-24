# Confirmed official Phase 2 bank: #330390

Observed 2026-09-09 01:12:40 UTC (2026-09-08 Pacific). Submitted 2026-09-09 00:45:35 UTC.

The authenticated AIcrowd status API reports **graded**, **Graded successfully**:

- Official adjusted score: **4.53886349950395e-9**.
- Official secondary/raw final-layer MSE: **2.120838555441651e-8**.
- Previous confirmed official bank #329926: adjusted **4.982231460183304e-9**, raw **2.2448352261505987e-8**.
- Adjusted improvement over that official bank: approximately **8.899%**.
- Local independent14 noise-corrected estimate for this exact source: **4.4837991190464774e-9** adjusted. Official result is approximately **1.228% higher**. These are different evaluation panels/truth conventions, not exact-parity targets.

Official status receipt: `official_status_20260909T011240461373Z.json`, SHA256 `cba14f242ff24ca413808b52f6a6e570d1b6d41eb83f91117cc2c06a99e733a9`.

Exact submitted candidate SHA256 `d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451`. Package, runtime and submission receipts remain preserved in this directory. Neither the new NODE pilot nor the capped4096 adaptation is included.

https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330390

The status API does not expose per-MLP completion/failure counts. The browser surface is unavailable and the public-card fetch did not return content. Do not invent a separate zero-failures count. The successful official aggregate closely supports the local performance estimate, but does not prove the absence of every possible implementation bug.

This satisfies the user's explicit confirmation-bank exception, not the <4e-9 milestone. Goal remains official adjusted <1e-9, with validated banks below4e-9,3e-9,2e-9. No forum post, promotion, root estimator change or commit.
