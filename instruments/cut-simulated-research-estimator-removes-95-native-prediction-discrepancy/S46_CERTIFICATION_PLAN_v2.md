# S46 cut simulator certification — 2026-10-10T23:24Z

Status: assembled and statically checked; NOT numerically certified, NOT enabled for refits.

The old `torch_cutsim_v2.py` is a standalone research runner from October 2. It does not load the S46 checkpoint/request, has no pair-mean truncation, assumes old-tier factors form one block, and adds radial repair after b-fix. Its old output is not evidence for current parity.

Current composite: `torch_cutsim_s46_v2.py`; dependencies `torch_push_native_v2.py`, `torch_rowcuts_v6.py`. Record all hashes from `sources()`. Install after PL and before M; call composite `install_m(cd, original_M_install)` instead of plain M.install. `opts={"enabled": false}` leaves the original card untouched.

Implemented:

- Native index-sorted Nystrom row ordering and 384/320/384 join sizes, native core formula, L13 lazy factor blocks retained through checkpoint state.
- Local pair-mean r_u=256 truncation at L10-15 in push and read; full factors remain carried and serve tail repairs.
- S46 dead threshold 3.0902 (64-grid), old positive/dead tails 2.5, young dead tail 2.5 and native repair formulas.
- Diagonal repair before local cp/k21; omitted source correction before b-fix/births/star; no in-place mutation of saved tensors.
- L15 3.5/32-grid pushed rows and K4L loc!=0 projection.

Explicit residual scope: covariance dead sandwich cut, edge response/alive cut and tanh stand-in, float32/tree-kernel rounding. These are not claimed simulated. Retained U rows are safe under the asserted alive-subset invariant: dead push uses a looser threshold and 64-grid versus alive 32-grid, so every omitted U row is annihilated before commit. Native-law beta column-sum restoration versus torch law V=2*m*(1-g) is an additional explicit residual.

Minimum numerical experiment, in sequence:

1. S46 checkpoint `trainer/cert/refit/jmI_p2_noAS.pt`, head `jmI_p2_hu.json`, exact S46 NCAR/f4c/B6 request, `KEEP_ALL=True`: 651 with simulation disabled. It must equal current original torch request exactly.
2. 651 with simulation enabled. Compare all 16 layers with existing native `engineer/logs/quick_s46s_fresh_a_pred_651.npy`; inspect per-layer jumps, especially L7/10/13/15. Repeat 378 only if 651 is finite and materially closes the residual.
3. Compare default ru256 with ru100000 on one network if the truncation direction disagrees; do not train on an unexplained wrong direction.
4. Suggested pre-refit target: remove at least 80% of original final-layer native/torch residual power on both ranks and final RMS residual <=7e-6, roughly the unmodeled-cut/rounding scale. This is a diagnostic target, not a leaderboard gain claim. Report MSE separately and investigate any residual structured jump.
5. One checkpoint/backward network at current values, finite gradients; compare a small learned-edge directional derivative against centered finite differences and verify checkpoint recomputation values. No new refit before this passes.
6. Then trainer fits on fresh sizing/acceptance sets; untouched judge and native release pipeline remain unchanged. Simulator fidelity on 651/378 is not evidence of a gain.

Static checks already passed: import/compile without torch; every one of five private step edits matched exactly once on current covstep; M projection replacement matched exactly once; resulting code compiled. No native session or torch process was launched by engineer.

Audits: physicist independently confirmed diagonal/radial placement, final-row mask and K4L loc!=0 projection. Native-push subagent independently reviewed UVP lazy blocks and float64 QR/solve -> float32 convention. Composite v2 source identities: entry 3e583d95, push c3b128b9, rows 8c511a0b. All-off control must use enabled:false; turning thresholds off alone intentionally retains native-push/projection arithmetic.

Read-only reporting: `parity_report_s46_v1.py RANK SIM.json ORIGINAL.json [NATIVE.npy]` prints all 16 RMS residuals, cut-direction cosine and residual-power ratio. Default native reference is stored S46 list-a predictions.
