# Final-source audit finds inert residual-meter access and setup-only budget binding

2026-10-21T22:03:54Z · finding · informational · derived

The review found residual-meter access confined to a disabled routing branch. Setup-only budget binding, scalar sample-count normalization and garbage-collection control were documented separately from network-dependent arithmetic, and subsequent release removed the dead meter access.

Recorded measurement:

    - 2026-10-21T22:03:54Z [fable-lead] IDEA_SCOUT COMPLIANCE REVIEW of #337619's release file: NO live breach. (1) GRAY: retired residual governor (lines 22136-22250) reads flops._budget._active_context.residual_wall_time_s in _fast_rescue_live_residual; INERT (only caller tests 'layer in _FAST_RESCUE_FRAC' first; file ends with _FAST_RESCUE_FRAC = {}; hosted ops identical) -> disclose as dead code in the write-up; strip from future lineages (not touching today's measured bytes). (2) self-opened BudgetContext only in setup on NoBudgetContextError for network-independent binds (disclose). (3) setup wall guard _PB_WALL_S via time.monotonic = plumbing. (4) count metadata used in Python 1/count, sqrt(count-1) (sample-size normalization) = plumbing. (5) shadowed dead _rtt_basis defs with Python **0.5 (dead). (6) Strassen/padding/route selection = exact-arithmetic methods, honestly metered; XSG off. (7) imports/RNG/no-IO clean.
    - 2026-10-21T23:55:03Z [fable-lead] DESIGNATION DONE by Matt: [instruction removed]) + #337619 (1.4005566e-9, XSG off) selected for the Phase 2 private re-evaluation. Archives verified from disk: XSG off in all three fired files, 0 governor refs in #337641, imports = flopscope/whestbench + stdlib gc/math/time/zlib/base64; gc block = interpreter plumbing, disclosed in the write-up draft (A0/A9). J2W late fires at 23:49Z: #337650 15/100 projecting ~1.49e-9 (paired +6.6% MSE at our util), #337651 dead (95 exceptions), 3 queued.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7575](../../TEAM_LOG_2026-10-01_to_10-22.md#L7575)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7594](../../TEAM_LOG_2026-10-01_to_10-22.md#L7594)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-final-source-audit-finds-inert-residual-meter-access-setup) · [Copy provenance](../../../PROVENANCE.json)

