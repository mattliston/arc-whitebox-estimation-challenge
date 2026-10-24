# Depth-seven column packing needs true product dimensions to price padding

2026-09-25T19:21Z · experiment · adopted · measured

The first depth-seven packing census counted already padded dimensions and overstated savings; it also excluded inner dimensions of 992. A corrected rule covers those products and saves arithmetic across four native test networks.

Recorded measurement:

    - 2026-09-25 12:21 [engineer2] CORRECTION to my 12:12 frontier (depth-5 padding steps 1 and 3): they were priced on S27, whose k-split v4 left whole-fit reads at depth 5 (k >= 4096 and nch >= 2 only). S28's v5w already moves reads with k >= 2048 whole (nch >= 1), so part of that -0.22% / -0.40% is already inside S28. The depth-7 colpack step is unaffected (young-tier pushes, not reads). Partition proposed to the engineer: k-split v5w/v6 own every read reaching _packed_read5 with k >= 2048 and v6's big-k wrapper; my p6 hook (_ipt_product_ab at depth 5, below every k-split wrapper) gets only what is left: pushes and squares with dims not multiples of 64, and reads with k < 2048. p6's real increment on S28, with lazy binds, comes from the parity run queued behind the machine gate (memory free ~30-40%). Estimated lazy-bind drag with S28's key list: the moves need 21-28 depth-6 keys outside the list (~0.31-0.34% if all bind on one net) and free 25-35 lazily bound depth-5 keys (~0.16-0.23%).
    - 2026-09-25 12:52 [engineer2] DEPTH LEVERS, IN-PROCESS PARITY ON S28 (a559b073; setup first; nets 35/100/37/39 in one process, so lazy binds count as a worker pays them; final-layer MSE vs the 32M truth; depth7/s28d7/):
      d7cp v1 (colpack at depth 7): bill -0.293/-0.257/-0.115/-0.242% (35/100/37/39), MSE ratios 0.9994/0.9966/0.9958/0.9997 (rounding level). Short of the -0.44% price because the S27 shape census had shown PADDED dims: layer 5 (every net) and layer 4 (net 37) are k = 992, which the native path pads to 1024 (_push6_padded); v1 required k = 1024 exactly. A call probe (probe_d7cp_calls.py) confirms per-call bills equal the bench to the Mflop (-126.3/-275.4/-424.6 MF), and the net totals are 65 MF above their sum: moving the colpack to depth 7 loses the transform reuse the layer-2/3 newborn product had on the same wp (the resident key now names depth 7). d7cp v2 (64210b64) adds k = 864/992 with the same padding the native path does; expected about -0.43% on every net. Its parity, client census, user-CPU pair and residual gate are running (run_d7cp2_gates_v4.sh).
      p6 v1 (depth-5 -> padded depth-6 rule): bill -0.090/-0.164/-0.206/-0.087% (mean -0.137%), MSE ratios 0.9972/1.0010/0.9992/1.0003. On S28, v5w already moves the whole-fit reads and the new depth-6 keys bind lazily under S28's list. Below the 0.2% bar, so NOT proposed for S29. It would need its keys in the prebind list to approach -0.25%.
    - 2026-09-25 12:46 [engineer2] d7cp v2 PARITY PASS on S28 (s28d7/cand2_T.py b1f9f6ac = s28_pre 27d8e622 + d7cp_fragment_v2.py 64210b64 + treuse 75aa9233; vs S28 a559b073; in process, setup first, 35/100/37/39): bill -0.464/-0.430/-0.432/-0.597% (mean -0.481%); final-layer MSE ratio vs truth 0.9980/0.9992/0.9945/1.0024 (rounding level); max |diff| 1.2e-5. Client census next, then the user-CPU pair and residual gate, which wait for the S28 fire chain and held-out runner to finish (lock order).
    - 2026-09-25 12:45 lead: S29 ITEM d7cp v2 (engineer2; depth7/d7cp_fragment_v2.py 64210b64, composed before treuse) passes parity on S28: bill -0.48% mean (-0.46/-0.43/-0.43/-0.60% on 35/100/37/39); final-MSE ratios 0.998/0.999/0.995/1.002 (rounding level). v2 also moves the k = 992 layers (L4-5) that the native path pads to 1024. Pending: client census, user-CPU pair, residual gate (after S28's chain). The depth-5 padding rule p6 measured -0.14% on S28 (v5w already takes the whole-fit reads): below the bar, not proposed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/depth7`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_d7cp_calls.py](code/probe_d7cp_calls.py)
- [code/run_d7cp2_gates_v4.sh](code/run_d7cp2_gates_v4.sh)
- [code/s28d7/cand2_T.py](code/s28d7/cand2_T.py)
- [code/d7cp_fragment_v2.py](code/d7cp_fragment_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1215](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1215)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1222](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1222)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1223](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1223)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1224](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1224)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1227](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1227)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1228](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1228)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-depth-seven-column-packing-needs-true-product-dimensions-price) · [Copy provenance](../../../PROVENANCE.json)

