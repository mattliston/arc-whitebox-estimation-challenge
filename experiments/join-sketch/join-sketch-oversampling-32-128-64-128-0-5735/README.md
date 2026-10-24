# Join-sketch oversampling at 32/128 and 64/128: +0.5735% / +0.8255% adjusted on 32 networks

2026-10-20T19:07:58Z · experiment · rejected · measured

The current-trajectory oversampling variants failed full-32 validation despite promising fresh-network screens. A terminal readout fit conditional on the larger sketch also failed its eight held-out networks.

Recorded measurement:

    REJECTED: NT32/q128 all32+.5735%,held25+.4038; NT64/q128+.8255%,held25+.6046. No NT stack. Terminal NT-conditioned fit24training frozen ridge10 LOO-.1503% raw vs NT; only8holdoutpending, would need>-1.16% conditional to clear firebar; do not confuse it with adjusted-v4a.
    COMPUTE: driver run_queue_astra_v6 PID77740/session97058 capacity5, native reservations reduce judge count. Gram/micro judges finished, pair/oldfullstack/k4lc_early_p running; next newstack. Coretrace native fresh7100076–79 session39006/run_coretrace_shards_v1.sh with reservedslotcoretrace, alpha0/.5/1, singleton shards. eval_fresh_arm_v8 adds actual L10 call/layer assertions, no numeric change. Native probes use existing guard supported9.5GB RSS backstop but unchanged7.9GBphysical footprint,15%hardfree,swaplimits; judges remain7.9GBRSS. Systemfree62%/swap42.98GB at19:20Z. Copyall census session98638 finished and reserve released. K31 roundingreuse firstseed+.163%, nextseed22 RSS7.909GB retirement withfoot3.4GB, no remaining processes; deprioritized after exact NP saving found. No native K31 currently running. NT head final held8+.30446% raw vs NT,1/8 lower: REJECT/noexport (HELDOUT_v1.json), reservation released. RTT seeds72–74 poor/partial;75unrun. Agents: bill_frontier narrow old-bank frontier + cheap edge approx(source ready-.0766%paper bill, unmeasured accuracy); joint_audit stack/binding/release and QR prior-closure check. Original release guards unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_coretrace_shards_v1.sh](code/run_coretrace_shards_v1.sh)
- [code/build_nt_current_v1.py](code/build_nt_current_v1.py)
- [code/candidate_ntc32_all_sci_v1.py](code/candidate_ntc32_all_sci_v1.py)
- [code/candidate_ntc64_q128_sci_v1.py](code/candidate_ntc64_q128_sci_v1.py)
- [code/candidate_rel_v4a_sci_v1.py](code/candidate_rel_v4a_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6237](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6237)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6248](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6248)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [HELDOUT_v1.json](HELDOUT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-join-sketch-oversampling-32-128-64-128-0-5735) · [Copy provenance](../../../PROVENANCE.json)

