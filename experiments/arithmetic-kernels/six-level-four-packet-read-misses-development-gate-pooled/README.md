# Six-level four-packet read misses development gate: pooled adjusted ratio 0.9950934019424302

2026-09-19 · experiment · rejected · measured

Cold binding costs and shape eligibility reduced isolated product savings, while one development network became slightly worse.

Recorded measurement:

    # Read6 four-packet variant — closed at the fixed development gate
    and a pooled adjusted ratio<=0.99 were required. Actual pooled ratio is
    **0.9950934019424302**, and rank100 is slightly worse.
    | Rank | Weight5 raw | Read6 raw | Weight5 cold FLOPs | Read6 cold FLOPs | Adjusted ratio |
    |35|1.87125543957e-08|1.8548489395e-08|405547750335|404792021179|0.989385217466|
    |100|1.85468233405e-08|1.85923440386e-08|408575923244|407905682722|1.0008099093|
    savings. New cold view bindings are charged, and only shapes divisible by64
    can select D6. Changed FP32 arithmetic slightly changes final predictions;
    it is not an accuracy-preserving cache optimization like Weight5. No physical
    35 repeats byte-for-byte, and warm/cold100 outputs match exactly.
    Independent fullD6 arithmetic matches all three tested shapes byte-for-byte:
    1024x8192x1024,832x8192x768,64x1088x64. The reference separately allocates each
    original frame, keeping each allocation below the4GB array cap. Its largest
    complete reference workspace is4728649728 bytes, not one allocation.
    Packet product bills match the exact rational fullD6 formula; interval lifetimes
    backward stages. Interleaved repeat passes. Maximum FP64-direct relativeRMS
    is7.2189531068e-5, below the fixed3e-4 tolerance.
    The two large warm bill ratios toD5 are0.9361604986 and0.9514411279. Warm
    wall time is materially worse (about0.624 versus0.406s, and0.554 versus0.290s).
    The new arena is1568602112 bytes,398297088 above Weight5/Read5. Unit peak
    observed RSS5423808512 bytes. Full development peak observed RSS6937296896 bytes,
    8GB guard. Four full predictions incurred1591065723250 billed FLOPs;
    outputs and bills were sealed before opening the existing35/1001B target means.
    Weight5 remains qualified at local existing14 NC3.883207214854186e-9; its exact
    #331584 remains4.015535762022727e-9, so no official4e-9 milestone has yet been

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/read6_packet_v1/build.py](code/read6_packet_v1/build.py)
- [code/read6_packet_v1/candidate.py](code/read6_packet_v1/candidate.py)
- [code/read6_packet_v1/kernel.py](code/read6_packet_v1/kernel.py)
- [code/read6_packet_v1/score_dev.py](code/read6_packet_v1/score_dev.py)
- [code/read6_packet_v1/unit.py](code/read6_packet_v1/unit.py)
- [code/weight5_reuse_v1/candidate.py](code/weight5_reuse_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-six-level-four-packet-read-misses-development-gate-pooled) · [Copy provenance](../../../PROVENANCE.json)

