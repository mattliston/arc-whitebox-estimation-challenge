# Padded depth-six candidate withheld for wall risk despite 0.50% adjusted improvement

2026-09-23T11:40Z · decision · rejected · measured

The candidate added about 4% server CPU and 6 ms residual against a recorded thirteen-second platform excursion. Submission was held until stronger margins could be demonstrated.

Recorded measurement:

    - 2026-09-23 04:40 [engineer] S7 BUILT (lead: S7 = S6b + --bindhold + push5p6 + --caps): fable_engineer_v1/candS7_mem.py (5bf2a07e) = build_stack.sh --gn=gn_knobs/v3.json --l15cut --join=v2 --refit=fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps --batch --bindhold, then build_e2mem.py mem,yright,arena (the S6b recipe reproduces porter's stack_S6b_mem a514c21b byte-for-byte). Vs S6b in-process: util -0.21/-0.38/-0.83/-0.52% (r35 cold, r100, r37, r39), MSE within +-0.09% (rounding), adj -0.17/-0.46/-0.85/-0.52%. Census r35 cold: round trips 55,353 -> 52,508 (free trips 8,535 -> 5,710). Queued with engineer2 for one gate after S6b.
    2026-09-23 05:01 [engineer2] S7 GATE (candS7_mem.py 5bf2a07e, reproduced byte-for-byte via build_stack.sh + build_e2mem.py mem,yright,arena; vs the fired S6b #332005 a514c21b). In-process r35 cold/r100/r37/r39: util -0.21/-0.37/-0.83/-0.53%, MSE +0.04/-0.09/-0.02/+0.01%, adjusted -0.17/-0.46/-0.85/-0.52% (mean -0.50%). Paired 20 distinct nets (concurrent, paired20.py g20S7 g20S6bF): server CPU +3.99% mean / +4.50% median (push5p6's cost, no longer offset since S6b already carries arena); residual +5.8 ms mean, cold net 1 181 vs 178 ms (+4 ms despite -2.8k round trips from --bindhold: the caps' LRU bookkeeping and push5p6 Python), warm +4..+13 ms growing as the caps fill; handles capped 63,042 after 20 (max 63,211; S6b 74,701); live peak identical 2,383 MB. Memory/handles PASS; the price is +4% CPU (~+3.4 s platform mean) and ~+10 ms platform residual for -0.5% adjusted. Decide on S6b's telemetry.
    - 2026-09-23 05:00 lead: S7 HELD (fable_engineer_v1/candS7_mem.py, 5bf2a07e = S6b + --bindhold + push5p6 + --caps): -0.50% adjusted vs S6b for +4.0% server CPU (~+3.4 s platform) and +6 ms residual (engineer2 paired gate). With a +13 s single-MLP platform excursion on record, it would put the worst MLP within reach of the 120 s cap. The private-eval designated file must never time out, so robustness outranks 0.5%. Revisit only if S6b's telemetry shows wide margins.

## Programs

- [candS7_mem.py](candS7_mem.py)
- [../../../experiments/late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/code/fable_porter_v1/build_stack.sh](../../../experiments/late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/code/fable_porter_v1/build_stack.sh)
- [../../../experiments/arithmetic-kernels/padded-depth-six-pushes-save-0-426-work-4/code/fable_engineer_v1/depthrule_fragment_push5p6.py](../../../experiments/arithmetic-kernels/padded-depth-six-pushes-save-0-426-work-4/code/fable_engineer_v1/depthrule_fragment_push5p6.py)
- [../../../experiments/arithmetic-kernels/five-level-layouts-previously-shallower-pushes-reduce-warm-work/code/fable_engineer2_v1/build_e2mem.py](../../../experiments/arithmetic-kernels/five-level-layouts-previously-shallower-pushes-reduce-warm-work/code/fable_engineer2_v1/build_e2mem.py)
- [../../../experiments/arithmetic-kernels/padded-depth-six-pushes-save-0-426-work-4/code/fable_engineer2_v1/paired20.py](../../../experiments/arithmetic-kernels/padded-depth-six-pushes-save-0-426-work-4/code/fable_engineer2_v1/paired20.py)
- [../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer2_v1/hold2.py](../../../experiments/arithmetic-kernels/estimator-body-request-reductions-bit-exact-predictions-0-58/code/fable_engineer2_v1/hold2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L195](../../TEAM_LOG_2026-09-22_to_09-26.md#L195)
- [TEAM_LOG_2026-09-22_to_09-26.md#L198](../../TEAM_LOG_2026-09-22_to_09-26.md#L198)
- [TEAM_LOG_2026-09-22_to_09-26.md#L199](../../TEAM_LOG_2026-09-22_to_09-26.md#L199)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [v3.json](../../../experiments/arithmetic-kernels/padded-depth-six-pushes-save-0-426-work-4/v3.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-padded-depth-six-candidate-withheld-wall-risk-despite-0) · [Copy provenance](../../../PROVENANCE.json)

