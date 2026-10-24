# Joint-plane information probe: closed

All24 new trajectories preserve the frozen full-pair host, sigma, and all30
old features byte-for-byte. No current-layer correction enters the added
48 response-weighted joint-plane features. Twenty train nets, four fixed
held-out nets37/39/42/48, layers10..15, one ridge scale, no sweep.

Held-out residual MSE against stored1e9 means after the frozen teacher-forced
parent head: base4.80933777312e-9; old-feature control4.82406653499e-9;
augmented joint-plane probe4.80789159607e-9. New/control ratio.9966470,
only0.34% gain, below the preregistered5% information gate.

No200-row GNN fit, port, scalar-final refit, panel, or submission is warranted
by this fixed feature block. This is a bounded linear information-probe
verdict, not a universal impossibility theorem for nonlinear joint models.
Receipts: `joint_plane_probe_rows_v1.json`, `joint_plane_probe_verdict_v1.json`.
