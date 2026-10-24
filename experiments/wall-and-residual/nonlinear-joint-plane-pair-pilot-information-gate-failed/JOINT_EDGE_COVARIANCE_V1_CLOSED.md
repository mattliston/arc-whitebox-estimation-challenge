# Nonlinear joint-plane pair pilot: information gate failed

Phase2, September5,2026. All eight new S-plane trajectories preserve
every old source-v2 feature, rho, sigma, covariance target, next weight
and response BYTE EXACTLY. No current target correction enters S.
NumPy/MPS predictor parity relative1.22e-7; symmetry and permutation
certificates exact; initial output is byte-exact zero.

Fixed six-train/two-held-out,40-epoch comparison, final epoch only:

| Input | Held-out rank | Covariance ratio | Next-mean ratio |
| --- | --- | --- | --- |
| matched zero-S control |400|.948588133|.998424679|
| matched zero-S control |500|.941617787|.984953255|
| joint S/S-transpose |400|.894476248|.994759491|
| joint S/S-transpose |500|.891117670|.952178435|

The .75/.75 and >=10% incremental next-value gates FAIL on both nets.
No more epochs, hidden width, pair features, large fit, rollout or panel
in this specified nonlinear edge family. Nominal covariance improvement
must not be reported as a raw/adjusted score gain. One MPS job44.73s,
peakRSS1.537GB; all workers complete. Swap5934.50MiB at18:30PDT.

Receipts: joint_edge_covariance_verdict_v1.json, both .pt pilot models,
and rows_joint_edge_covariance_v1/. Official read-only poll at18:33PDT
still reports329926 graded successfully, adjusted4.982231460183304e-9,
raw2.2448352261505987e-8. No new submission or forum write.
