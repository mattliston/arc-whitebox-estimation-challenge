# Accumulated tracker readout reached 0.346 alone and 0.811 combined with the correction chain

2026-09-01 · experiment · informational · measured

(2026-09-01, post-restart): r35 readout ; v-arm −0.7% → +2.2%

Corrected joint-replacement dv-hat file through the consumption:
only_v **+2.2%** (the old carrier's v-arm was −0.7% on r35: the
tracker turns the variance channel positive, +2.9 pts swing). The
all-arms total (0.5% vs 2.1% base) is dragged by the known-bad k3
arm interacting with the better v-supply ; the deployable form is
the v-arm alone. Board-terms: 2.94 -> ~2.87e-8.

Recorded measurement:

    ## T-11 (2026-09-01): ACCUMULATION DELIVERS — 0.346 at L12 (chain: 0.79)
    
    Full-depth assimilated tracker (L3-L11 supplies, per-layer
    two-channel betas, hump-aware LMIN=3) accumulated through the
    chain's D-transport: acc corr vs dv_true(L12) = **0.346** (from
    0.17 with 5 layers/global betas). The G3-channel betas are STABLE
    across nine layers (0.41-0.85 — consistent physical scale; the
    frozen-transfer object). 44% of the chain's level from closed forms
    + card assimilation. T-10's diagnosis confirmed: coverage + scaling
    were the gap. RUNNING: per-layer dv-hat emission -> joint-with-chain
    orthogonality -> stacked consumption (the first estimated-MSE with
    the tracker in the loop). Then: r100 frozen, and the T1 gate
    re-read (the accumulated form is the gate's real object).
    
    
    ## T-11b (2026-09-01): JOINT 0.811 vs chain 0.79 — information real; combination bug found
    
    Per-layer dv-hat emission + joint grade: joint_with_chain **0.811**
    (chain 0.79; cross-corr 0.211 — largely orthogonal; added
    delta-corr^2 = 0.034). BUT the stacked consumption read -5.8%: the
    dvhat file ADDED the running accumulated total at every layer
    (double-counting early contributions downstream). The 0.811 came
    from the two-beta REPLACEMENT form — the file must be built the
    same way (per-layer lstsq of [chain, DV_ACC] onto the target;
    in-sample betas, frozen-tested next). Corrected chain running:
    r35 rebuild -> consumption -> r100 FROZEN -> consumption.
    The negative readout also certifies the instrument's honesty (it
    punishes over-counting rather than flattering it).
    
    
    ## T-11c (2026-09-01, post-restart): r35 readout — v-arm −0.7% → +2.2%
    
    Corrected joint-replacement dv-hat file through the consumption:
    only_v **+2.2%** (the old carrier's v-arm was −0.7% on r35: the
    tracker turns the variance channel positive, +2.9 pts swing). The
    all-arms total (0.5% vs 2.1% base) is dragged by the known-bad k3
    arm interacting with the better v-supply — the deployable form is
    the v-arm alone. Board-terms: 2.94 -> ~2.87e-8. Exactly the
    corr^2-predicted 2-3% row; no surprise either direction.
    r100 FROZEN replication running (same construction, same lambda).
    Parallel: assimilated-state dump at L5 -> word-vs-exact T-Gram
    probe (the word shortcut is the largest approximation in the live
    G3 channel); Q4-sector channel derivation certificate; the
    supply-strategist teammate researching faster paths to perfect m+v
    (FASTER_PATH_MEMO.md).
    
    ## Q4 emission CERTIFIED (2026-09-01): the fourth exact resolvent formula
    
    E[grad^4 phi] = E[phi'''' d4] + 6 sym(E[phi''' d2] (x) M)
                  + 3 sym(E[phi''] M (x) M), with tilted moments
    E[d4 e^{itz}] = CF[m4 + 6sym(mmC) + 3sym(CC)], E[d2 e^{itz}] =
    CF[mm' + C], kernels (it)^2 / it / p(0). MC certificate (r=5,
    N=6e6, six index patterns): corr 0.99991; ratios 0.998/1.039/0.995/
    0.999/0.987 on the material entries (the (0,1,2,3) entry 1.12 is
    noise-dominated at 2e-3 magnitude). The Q4 channel is buildable
    exactly; its word/exact form choice waits on the T-Gram probe
    (word-vs-exact fidelity at depth), running.
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/build_gate1_supplies.py](code/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/public_g19_attribution_v1/consumption_betas.py](code/public_g19_attribution_v1/consumption_betas.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [LANE_g3_tracker.md#L247](LANE_g3_tracker.md#L247)
- [LANE_g3_tracker.md#L261](LANE_g3_tracker.md#L261)
- [LANE_g3_tracker.md#L275](LANE_g3_tracker.md#L275)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](LANE_g3_tracker.md) · record
- [FASTER_PATH_MEMO.md](FASTER_PATH_MEMO.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-accumulated-tracker-readout-reached-0-346-alone-0-811) · [Copy provenance](../../../PROVENANCE.json)

