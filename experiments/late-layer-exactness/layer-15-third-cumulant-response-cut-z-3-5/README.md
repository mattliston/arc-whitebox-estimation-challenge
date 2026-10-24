# Layer-15 third-cumulant response cut |z| < 3.5: native adjusted error improves 0.66% on 40 networks

2026-09-23T07:05Z · experiment · adopted · measured

Restricting the final third-cumulant read to responsive rows reduced work with a small raw-error cost. Other late-layer architecture cuts failed.

Recorded measurement:

    - 2026-09-23 00:09 [physicist] LEADER-GAP ARCH SCREEN (fable_physicist_v1/leadergap/RESULT.md; torch S2, learned ON, no refit, r35/r100): nothing local reaches -25% bill or -10% raw. Births into the frame at L13/14 +79 to +258% raw; response/live/birth thresholds .03 at L10-14 +3 to +10% raw for -0.7 to -2.6% bill (adj +2 to +8%); no cheaper exact star product. Only free change: L15 k3 row cut |z|<3.5, raw +0.01/+0.41%, bill ~-1.0% -> adj ~-0.6/-0.95%. Telemetry: J2W and ben3 gain only at the final layer; our L15 anatomy puts that on the per-unit variance from the carried off-diagonal L14 covariance.
    - 2026-09-23 00:05 [physicist] (correction) the LEADER-GAP line above stamped 00:09 was written at 00:05.
    - 2026-09-23 00:33 [physicist] L15 ROW CUT NATIVE (fable_physicist_v1/leadergap/build_l15_rowcut.py <in> <out> [zc=3.5]; count-asserted, v7/v8 forms): S2+cut raw +0.009/+0.434% (torch +0.011/+0.442%), util -0.61/-0.85%, adjusted -0.60/-0.42%; S3-HEB+cut same MSE, adjusted -0.58/-0.41%. Smaller than estimated because depth-6 Winograd pushes need m==1024. INFO TEST (RESULT.md addendum): Gaussian-born 4-cycle/triangle-pendant/(2,1,1)/W o W terms explain <=7% of the K31/K22 read error at r35 L12 (net <1%); cross-net R2 of the one-sided top-k R at L10-14 <=.012 with every new family (bar .2) -> the learned covariance channel is dead on information; no final-layer-only v15 mechanism either.
    2026-09-23 01:10 porter: S4 fire file fable_fire_v1/candS4_S3HEB_gn.py (sha 082f2b50...; MSE == gated S3+GN, flops -0.38/-0.25%, validate OK). Knob-port r35 gap = rounding noise (f64 delta alone moves r35 0.13 pt). build_stack.sh: --k31row --bfix --l15cut --join (default rebuild byte-identical to S4); l15cut on S4 adj -0.62%/-0.61% (r35/r100).
    2026-09-23 02:05 porter: S5 = S4 + L15 cut: VALID40 vs S4 exact per net adj 0.99344 (-0.66%), 37/40 below S4; vs bank 0.91543 40/40. S4 VALID40 0.92147. join-v2 twin bit-identical; validate OK on both.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/leadergap/build_l15_rowcut.py](code/fable_physicist_v1/leadergap/build_l15_rowcut.py)
- [code/fable_fire_v1/candS4_S3HEB_gn.py](code/fable_fire_v1/candS4_S3HEB_gn.py)
- [code/fable_porter_v1/build_stack.sh](code/fable_porter_v1/build_stack.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L108](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L108)
- [TEAM_LOG_2026-09-22_to_09-26.md#L109](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L109)
- [TEAM_LOG_2026-09-22_to_09-26.md#L123](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L123)
- [TEAM_LOG_2026-09-22_to_09-26.md#L134](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L134)
- [TEAM_LOG_2026-09-22_to_09-26.md#L148](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L148)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../../../submissions/331992/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-layer-15-third-cumulant-response-cut-z-3-5) · [Copy provenance](../../../PROVENANCE.json)

