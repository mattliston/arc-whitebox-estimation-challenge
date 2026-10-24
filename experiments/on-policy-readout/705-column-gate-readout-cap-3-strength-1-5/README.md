# 705-column gate readout with cap 3: strength 1.5 gives -0.176% pooled adjusted across 76 observations

2026-10-21T16:44:50Z · experiment · adopted · measured

The gate-feature correction at strength 1.5 combined with a three-read cap improved Mac, Linux and fresh paired panels. Strength 1.0 was also measured but gave a smaller pooled improvement.

Recorded measurement:

    - 2026-10-21T16:44:50Z [fable-lead] GATE STACK VM32 vs 49bd: extgate_s15_cap3 all32 -0.212% (SE 0.268, 19/32) held25 -0.330% raw -0.576% bill +0.366% requests +0.857%; extgate_s10_cap3 all32 -0.078% held25 -0.155% raw -0.440%. POOLED (Mac+VM+fresh, 76 nets): s15_cap3 -0.176% (~1.4067e-9; held25 pooled -0.36%), s10_cap3 -0.121%. Presented to Matt: [instruction removed]) vs cap4 gate (~-0.36% expected, slowest ~109 s); recommended cap3 first; awaiting his word.
    - 2026-10-21T16:28:03Z [fable-lead] STACK extgate_s15_cap3 (sci 2f12cc29; gate fit x1.5 + cap3) Mac32 vs 49bd: all32 -0.104% (SE 0.282, 18/32), held25 -0.393% (raw -0.758%, 17/25), seven +0.928% (raw +0.571%), raw all32 -0.467%, bill +0.364%. Mac projection 1.4046 -> expected hosted ~1.4080 (all32) / ~1.4037 (held25). VM32 + fresh12 pending; s1.0 Mac32 running.
    - 2026-10-21T16:36:49Z [fable-lead] STACK extgate_s10_cap3 Mac32 vs 49bd: all32: adjusted -0.1457% (SE 0.2221, lower 18/32); raw -0.5085%, bill +0.3642%; legacy -0.1307 pp | heldout25: adjusted -0.2340% (SE 0.2625, lower 14/25); raw -0.5990%, bill +0.3667%; legacy -0.2162 p
    - 2026-10-21T16:38:22Z [fable-lead] fresh12: extgate_s15_cap3 -0.272% (SE 0.376, 7/12, bill +0.243); extgate_s10_cap3 -0.170% (SE 0.270, 7/12). Pooled with Mac32 (-0.104/-0.146) ~ -0.15..-0.2% -> ~1.407e-9 for the cap3 gate stacks; held25-only projection ~1.4037. Launching extgate_s15_cap4 (VM32+fresh12+Mac32) as the lower-margin/higher-gain option.
    - 2026-10-21T16:45:47Z [fable-lead] FIRED #337565 (fifth of final ten; 5 remain) on Matt's word ('okay fire cap-3'): tag fable_extgate_s15_cap3_8b662ff4_v1, archive 2b5f0e143fc9f342 (1,500,189 B), SCI 2f12cc29 / REL 8b662ff4 = #337506 source + gate-readout refit x1.5 (705-col on-policy fit) + Smirnov read cap 3. Evidence pooled 76 nets -0.176% (Mac32 -0.104/VM32 -0.212/fresh12 -0.272; held25 pooled -0.36%), bill +0.37%, requests +0.86%, paired wall ~-10%, residual ~-3%; gates clear (setup 3.49 s). Expected hosted ~1.4067e-9. Matt: [instruction removed]. BEST unchanged until 100/100 clean.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_gate_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_gate_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_second_step_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_second_step_v1.py)
- [code/fable_oct21_final_v1/stacks/extgate_s10_cap3/candidate_extgate_s10_cap3_science_v1.py](code/fable_oct21_final_v1/stacks/extgate_s10_cap3/candidate_extgate_s10_cap3_science_v1.py)
- [code/fable_oct21_final_v1/stacks/extgate_s15_cap3/candidate_extgate_s15_cap3_science_v1.py](code/fable_oct21_final_v1/stacks/extgate_s15_cap3/candidate_extgate_s15_cap3_science_v1.py)
- [code/fable_oct21_final_v1/tails/sm40_cap3.py](code/fable_oct21_final_v1/tails/sm40_cap3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7480](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7480)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7474](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7474)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7477](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7477)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7479](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7487](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7487)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337565: 705-column gate readout at strength 1.5 and grouped-read cap 3](../../../submissions/337565/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-705-column-gate-readout-cap-3-strength-1-5) · [Copy provenance](../../../PROVENANCE.json)

