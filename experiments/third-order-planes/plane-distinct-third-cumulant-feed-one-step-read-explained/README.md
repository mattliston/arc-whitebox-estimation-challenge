# Plane-to-distinct third-cumulant feed: one-step read explained 16–30% of the residual

2026-09-04 · experiment · informational · measured

The engine's kernel loop adds (1/2) K21pre[u,w] E_conn[phi''_u phi'_w phi_s] to the distinct post-tensor entries; the card has no such channel (planes -> planes via the pair series, star pushed once). Exact next-layer per-unit reads on the card's own pre-plane (30-pattern certificate 3.6e-7): rms 16-27% (r35) / 18-30% (r100) of dk3 at every layer, slope 0.9-1.15 (r35 L2-11), corr 0.19-0.39, one-step energy 4-12%. ONE pattern carries it all: (0,1,0) = (1/2) K21pre[u,w] (-alpha_u pdf_u/sd_u) Phi(alpha_w) (sd_s Phi(alpha_s)) rho_us -> 2 GEMMs per read (deployable class); two-sided 1024^4 patterns are nil.

Recorded measurement:

    The engine's kernel loop adds (1/2) K21pre[u,w] E_conn[phi''_u phi'_w phi_s] to the distinct post-tensor entries; the card has no such channel (planes -> planes via the pair series, star pushed once). Exact next-layer per-unit reads on the card's own pre-plane (30-pattern certificate 3.6e-7): rms 16-27% (r35) / 18-30% (r100) of dk3 at every layer, slope 0.9-1.15 (r35 L2-11), corr 0.19-0.39, one-step energy 4-12%. ONE pattern carries it all: (0,1,0) = (1/2) K21pre[u,w] (-alpha_u pdf_u/sd_u) Phi(alpha_w) (sd_s Phi(alpha_s)) rho_us -> 2 GEMMs per read (deployable class); two-sided 1024^4 patterns are nil. Accumulated through the gate-only chain it reaches 55-60% of dk3 at depth but overshoots (slope 0.4-0.8): the card's real propagation of this content runs through the next planes/diagonal too. T-70 engine ablation (r35, per-unit k3 vs 32M truth): card 7.3/11.1% at L8/L14, engine with feed 4.4/7.3%, without 4.9/8.0% -> the feed is ~20% of the card->engine k3 gap in error points (its content is as large as the engine's whole residual, corr +0.5 with it). MC certificate of the kernel: ratio 0.86-1.02 (coefficient 1/2 right). IN-CARD VALUE (T-69b/c, deployed operator untouched, independent 4M): k3 read alone r35 -10.2% / r100 -1.7% (3 GEMMs/layer, +4% util); full one-step plane read r35 -12.4% / r100 -2.7% at the physical scale (9 GEMMs/layer, +12% util); engine's exact accumulated feed content injected: r35 -9.1%. The first derived correction that improves the full card with the operator in place, but not yet a fire: adjusted score flat or worse at these bills; r100-class nets gain little.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g29_bond_kernel_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engine_feed_ablation_table.py](code/engine_feed_ablation_table.py)

## Evidence

- [LANE_g3_tracker.md#L681](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L681)
- [PLANE_FEED_ACC_rank0035_onesided.json#L1](PLANE_FEED_ACC_rank0035_onesided.json#L1)
- [ENGINE_FEED_ABLATION_rank0035.json#L1](ENGINE_FEED_ABLATION_rank0035.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [PLANE_FEED_ACC_rank0035_onesided.json](PLANE_FEED_ACC_rank0035_onesided.json) · result
- [ENGINE_FEED_ABLATION_rank0035.json](ENGINE_FEED_ABLATION_rank0035.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-plane-distinct-third-cumulant-feed-one-step-read-explained) · [Copy provenance](../../../PROVENANCE.json)

