# Joint-dial evaluation required deep-copying mutable settings, integer layer keys and live cache resets

2026-10-20T17:13:16Z · incident · informational · measured

The first joint-dial run failed after estimator mutation of cache dictionaries, and the next silently missed layer keys converted to strings. A subsequent audit also found cached third/fourth scalar settings were no-ops until the constant cache was rebuilt.

Recorded measurement:

    - [fable-lead] 2026-10-20T17:13:16Z JOINT FD relaunched 17:12:00Z after a harness crash (settings that pass cache dicts (_PLD_KNOBS, _K21N_CACHE, _E2_EDGE_STATE) get FILLED with arrays by the estimator, so json.dumps(setting) failed after the first predict): eval_fresh_arm_v4.py deep-copies every setting value before setattr (v3 logs kept as eval_joint_fd_{A,B,C}_crash1.log). 41 settings x 2 nets x 3 processes, ETA ~18:12Z; the lane driver restarts after it (kc_late_mm/mmm first).
    - [fable-lead] 2026-10-20T17:19:15Z JOINT FD relaunched AGAIN 17:18:37Z: the 17:12Z run was invalid (JSON settings carry STRING layer keys; _GN_KC.get(l) / _GSK_BY_LAYER.get(int(l)) / _PLD iso lookups silently missed: base U read 1.3914e-8 instead of 1.3242e-8 on 7100005 (the KC term zeroed) and every dial move was a no-op). eval_fresh_arm_v5.py converts digit keys back to int recursively before setattr. Logs of the invalid run kept as eval_joint_fd_{A,B,C}_crash2_strkeys.log. ETA ~18:20Z.
    - [fable-lead] 2026-10-20T17:25:28Z JOINT FD (valid run, 17:18Z): base U reproduced (7100005 1.3242e-8); the half early CMODE profile (B1) reads +0.38% / -0.18% / +0.55% on 7100005/07/09 (the per-net split again); kc_early moves U (works); BUT the gsk_early and gk4_early settings are NO-OPS in process (U identical to the base): _gq's terms are cached in _GSK_CONST at setup, so swapping _GSK/_GK4 per call does nothing; the queued gsk_early/gk4 judge arms would be no-ops too (their tails swap the global per layer after the constants were built). Fix: the wrapper must clear _GSK_CONST when it swaps the value (rebuild per layer). The FD run continues (kc/iso/k21n/edge/iso_late/k31 dials and the pairs without gsk are valid).
    - [fable-lead] 2026-10-20T17:26:18Z GSK/GK4 per-layer arms REBUILT with a cache-safe wrapper (forces _GQ33_CONST rebuild around a swapped _gq call): gsk_early_p/m, gsk_late_p, gk4_early_p/m, gk4_late_p/m, stack_early_a/b (the first versions kept as *_NOOP); candidate_joint_sci_v2.py = joint base with the fixed wrapper for the next FD round. To verify in process before lane time is spent: gsk_verify_settings_v1.json (base vs GSK 0.99 at L1-7) on one fresh net in the gap after the FD.
    AUDIT: lead FD glob included crash logs (fixed to exact A/B/C + settings validation); step units differ (lead x must /0.2 before builder); old agent setting composer drops iso and attenuates previous dials (use clean helper/builder). Builder now uses cache-safe joint_sci_v2; release wrapper extraction fixed. Fresh harness v5 warmed only first net: repeated no-op settings expose -0.31..-0.37% spurious bill on second nets; v6 warms every net and prints 12-digit U. Provisional half-profile6: corrected adjusted +0.028% (3/6 lower), not a win. Existing compare32 reports percentage-point deltas against95c; new compare_relative_v1 gives true relative vs rel_v4a AND legacy. kc_late_m true adjusted -0.0843%, held25 -0.0962% (legacy -0.0830/-0.0937); below fire bar. Require both conventions <=-0.35 plus heldout negative.

## Programs

- [eval_fresh_arm_v4.py](eval_fresh_arm_v4.py)
- [../../../experiments/covariance-structure/early-mean-direction-covariance-corrections-layers-2-9-all/code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v5.py](../../../experiments/covariance-structure/early-mean-direction-covariance-corrections-layers-2-9-all/code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v5.py)
- [candidate_joint_sci_v2.py](candidate_joint_sci_v2.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6208](../../TEAM_LOG_2026-10-01_to_10-22.md#L6208)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6209](../../TEAM_LOG_2026-10-01_to_10-22.md#L6209)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6210](../../TEAM_LOG_2026-10-01_to_10-22.md#L6210)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6211](../../TEAM_LOG_2026-10-01_to_10-22.md#L6211)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6218](../../TEAM_LOG_2026-10-01_to_10-22.md#L6218)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-joint-dial-evaluation-required-deep-copying-mutable-settings-integer) · [Copy provenance](../../../PROVENANCE.json)

