# Learned-edge channels 9 and 14 pruned: fresh gain exceeds full-32 gain; pruning channel 9 alone fails

2026-10-20T21:36:20Z · experiment · adopted · measured

Removing both weak channels gave a fresh incremental gain of -.306759% over its exact parent. Full-32 results reduced that gain, while removing only one channel and combining an extra fourth-order scalar did not transfer.

Recorded measurement:

    Channel pruning fresh4: EH14 -.659639% vs bank, SE.268027,3/4 lower; incremental to microedge default -.306759%,SE.078085,4/4 lower. EH15 -.223220% vs bank but +.131515% to own default, only1/4 lower: do not promote. D/edge_hidden_fresh_v1/RESULT_FRESH4_v1.json. Combined EH14+GK4 gain UNKNOWN; no adding standalone percentages.
    Allthreepriority32 COMPLETE versus best337393: EH14+FG -.278827996%SE.099651,24/32lower,held-.286301936%19/25; GK4+FG -.168919680%held-.167024599%; EH14+GK4+FG -.169095778%held-.185924579%. ExtraGK4hurtsownparent;do notstack. EH14onlyessentiallytiesfallbackmicroedge-.276832%. Both oldfreshscreenlarger gainsdidnottransfer. FullVS_BEST_337393_v1 receipts saved. Freshdrop14-only -.3595569%vsbankbut+.3018815%vsEH14,0/4lower;closed. sqrt4alreadyclosed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/edge_hidden_fresh_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_v1.py](code/run_v1.py)
- [code/analyze_v1.py](code/analyze_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6325](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6325)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6342](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6342)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EH14_FRESH4_VS_BANK_PRELIM_v1.json](EH14_FRESH4_VS_BANK_PRELIM_v1.json) · result
- [MANIFEST_v1.json](MANIFEST_v1.json) · result
- [RESULT_FRESH4_v1.json](RESULT_FRESH4_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337421: Layer-14 edge width reduction with selective flat-upload restoration](../../../submissions/337421/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-learned-edge-channels-9-14-pruned-fresh-gain-exceeds) · [Copy provenance](../../../PROVENANCE.json)

