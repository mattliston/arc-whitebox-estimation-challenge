# Platform wall calibration: about 2.5 times local server CPU plus 0.22 ms per request

2026-09-23T06:49Z · tool · informational · measured

Public per-network telemetry showed that the platform was primarily compute-bound. A telemetry reader and conservative worst-network wall target were adopted for candidate gating.

Recorded measurement:

    - 2026-09-23 00:15 lead: PLATFORM WALL CALIBRATED from the public per-MLP telemetry on each submission page (new tool fable_fire_v1/read_eval_page.py <sid> -> eval_<sid>.json). Predict wall = server kernel + server overhead + residual. #331824 bank: mean 100.0 s, worst 110.2 s; kernel 87.5 s; overhead 12.2 s at ~58k requests. #331947 S2: mean 95.7 s, worst 105.6 s; kernel 85.5 s; overhead 9.9 s at ~45k requests. Overhead is ~0.22 ms per request, and platform kernel time is ~2.4x our local quiet single-thread server CPU. So the platform is COMPUTE-bound; request diets buy little wall. RULE: delta platform wall ~ 2.5 x delta(local quiet paired server CPU) + 0.22 ms x delta(requests), and keep the worst MLP <= ~112 s. sq6_p64 (+11 s local CPU vs S3) -> ~128 s = TIMEOUT. S3-HE (-1.7 s local vs S2) -> worst ~101 s.
    - 2026-09-22 23:49 lead: TIMESTAMP CORRECTION: my entries stamped 23:50, 23:58 and '2026-09-23 00:15' were written at about 23:25-23:45 on 09-22 (I misread the clock). Content unchanged.
    - 2026-09-22 23:52 lead: RIVALS' TELEMETRY (every board entry's submission page is public; fable_fire_v1/board_evals/eval_<sid>.json, read_eval_page.py). Platform wall mean/worst (kernel): us 95.7/105.6 (85.5); J2W 110.2/116.7 (99.8); suliman 92.9/108.2 (80.7); marius 98.6/111.3 (85.8); reds 76.8/79.0 (67.6); ben3 95.9/99.3 (87.2). Flops/MLP: us 3.97e11; J2W 2.86e11; suliman 3.01e11; marius 3.209e11 CONSTANT on every net (fixed-size card); reds 3.56e11; ben3 4.04e11. Per-net final MSE vs ours (50 public nets): J2W 0.901, suliman 0.894, marius 0.917, ben3 0.918, reds 0.947; corr(log MSE, ours) .74-.90, so the per-net difficulty is shared (same family). ALL-LAYERS MSE: us 7.22e-9, J2W 7.34e-9, suliman 6.38e-9, marius 6.70e-9, ben3 13.6e-9. So J2W's intermediate means are no better than ours and its ~10% gain is at the final layer; ben3 is 2x worse on intermediate layers yet 8% better at the final one. J2W buys its bill cut with compute time (kernel +17% vs ours) and runs 3.3 s from the cap.

## Programs

- [code/read_eval_page.py](code/read_eval_page.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L98](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L98)
- [TEAM_LOG_2026-09-22_to_09-26.md#L99](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L99)
- [TEAM_LOG_2026-09-22_to_09-26.md#L100](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L100)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-23-platform-wall-calibration-about-2-5-times-local-server) · [Copy provenance](../../PROVENANCE.json)

