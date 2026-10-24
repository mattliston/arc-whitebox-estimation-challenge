# Cheaper square-root activation: +.294248% adjusted versus its exact pruned-edge parent

2026-10-20T21:46:58Z · experiment · rejected · measured

The cheaper activation saved -.066175% FLOPs but increased raw MSE +.360662%, worsening all four paired networks. Keeping one formerly pruned channel also failed its exact-parent comparison.

Recorded measurement:

    Fresh sqrt4 cheaper activation CLOSED: -.367023% vs337393 but +.294248% vs own EH14 parent, SE.062012,0/4 lower; bill-.066175% outweighed by raw+.360662%. No32promotion. New follow-up drops ONLY edgechannel14 and preserves9 (priorEH15drop9 hurt, EH14drop9+14 helped). Tiny actual0.12.3 full edge function/zero-output oracle PASS; GNN messagewidth16 stays. SCI5af0cb591ab9909ee4eff342cb07d262032ede08f1f17aca614f6848d64426c9, RELd044389f4cbdd98341acf126d90547cb12899d4a2c47677407f283a06af0567a. Tailbd7762c4. D/EDGE_DROP14_ONLY_BUILD_v1.json and CHECK. Aux fresh4 runningD/edge_drop14_fresh_v1, session47105, launch21:38:17Z; last95started21:44:46Z after headroom drain. Same92–95bank/EH14 paired controls, no additive claim; selectedfollow-up, not independentholdout.
    Allthreepriority32 COMPLETE versus best337393: EH14+FG -.278827996%SE.099651,24/32lower,held-.286301936%19/25; GK4+FG -.168919680%held-.167024599%; EH14+GK4+FG -.169095778%held-.185924579%. ExtraGK4hurtsownparent;do notstack. EH14onlyessentiallytiesfallbackmicroedge-.276832%. Both oldfreshscreenlarger gainsdidnottransfer. FullVS_BEST_337393_v1 receipts saved. Freshdrop14-only -.3595569%vsbankbut+.3018815%vsEH14,0/4lower;closed. sqrt4alreadyclosed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/edge_drop14_fresh_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_v1.py](code/run_v1.py)
- [code/analyze_v1.py](code/analyze_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6333](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6333)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6342](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6342)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EDGE_DROP14_ONLY_BUILD_v1.json](EDGE_DROP14_ONLY_BUILD_v1.json) · result
- [MANIFEST_v1.json](MANIFEST_v1.json) · result
- [RESULT_FRESH4_v1.json](RESULT_FRESH4_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-cheaper-square-root-activation-294248-adjusted-versus-exact-pruned) · [Copy provenance](../../../PROVENANCE.json)

