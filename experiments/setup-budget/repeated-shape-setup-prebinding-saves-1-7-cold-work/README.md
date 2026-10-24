# Repeated-shape setup prebinding saves 1.7% cold work; all five platform workers complete setup in about 2 seconds

2026-09-25T00:36Z · experiment · adopted · measured

Prebinding recurring frame shapes moves their construction into setup and reduces the cold prediction premium. A forced arena-loss test confirmed byte-identical lazy rebinding fallback.

Recorded measurement:

    - 2026-09-24 17:36 [engineer] TREE PREBINDING IN setup() (E-61 --prebind): bit-exact AND a cold-bill cut. A worker's first MLP binds ~219 tree cache entries (~22 view requests each); binding reshapes are billed at numel. Keys recur: 8 non-VALID nets (300..580) give 358 keys used by >= 2 (<= 121 per cache side) covering 62-76% of held-out 35/100/37/39. S16r3bp (all 358 bound in 1.73 s at load 10): client/server predictions byte-identical to S16 (r35/r100); bill r35 cold 359.50 -> 353.50 G (-1.67%), r100 second -0.39%; ~-3.3k requests on the cold MLP. v1 bug: least-frequent-first, so at load 14 its 2.0 s guard kept the 175 least useful (-0.47% cold bill). FIXED in v2b = fable_engineer_v1/candS16r3bp2b_mem.py (579056e9): most frequent first, LRU re-order with no requests, 3.0 s guard (platform setup limit 5 s). flopscope_server's ConnectionStore keeps setup()-minted handles across MLPs by design; the first predict still checks the arena with one free request. Identity + 12-net handle run vs S16r3b queued. HARNESS NOTE: census clients must call Estimator.setup() before the first net for prebind candidates (fable_engineer_v1/census_setup.py). Local residual timings are unusable now (load 10-14 from background refits; S16 ran 46-68 s/net).
    - 2026-09-24 19:20 [engineer] S17 #332224 PLATFORM CHECK of my flags (setup_telemetry.py, eval_332224 vs eval_332210): setup() prebinding SUCCESS on all 5 workers, 1.93-2.04 s wall (limit 5 s; the 3.0 s guard never fired), 12.0 GFlop each booked to setup_executions (off-budget). Cold MLPs bill +0.9% over the warm mean (S16 +2.6%) = the predicted -1.7% on first MLPs. Residual: cold .227-.241 s (S16 .253-.275), warm mean .240 (S16 .262), MAX .267 s (S16 .313): -46 ms on the max, cap margin now 133 ms. Roundtrip max 73.7 s, 0 failed.
    - 2026-09-24 20:03 [engineer] (1) PREBIND FALLBACK FORCED ONCE (lead's S17 condition; prebind_fallback_test.py, fired S17 file in process): setup bound 358; the arena was then replaced by an object whose indexing raises; the first predict logged 'arena lost after setup: lazy binding', completed, and its predictions are BYTE-IDENTICAL to S17 without prebind (bill = the ordinary cold bill, 219 entries rebound lazily). PASS. (2) COVARIANCE LATE ARM (layers 8-14 only, 8 VALID nets): r512 +393% (~5x), r256 +3,282%; 0/8 better at both. Kill confirmed at every layer range. (3) Starting the computed-but-unused audit of S18 with a gradient-liveness probe on the torch card.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS16r3bp2b_mem.py](code/candS16r3bp2b_mem.py)
- [code/census_setup.py](code/census_setup.py)
- [code/setup_telemetry.py](code/setup_telemetry.py)
- [code/prebind_fallback_test.py](code/prebind_fallback_test.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L518](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L518)
- [TEAM_LOG_2026-09-22_to_09-26.md#L583](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L583)
- [TEAM_LOG_2026-09-22_to_09-26.md#L597](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L597)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../../../submissions/332224/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-repeated-shape-setup-prebinding-saves-1-7-cold-work) · [Copy provenance](../../../PROVENANCE.json)

