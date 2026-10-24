# False initial rescue flag did not disable residual-triggered routing; future builds clear its activation map

2026-10-20T20:32:34Z · incident · informational · measured

The initial false rescue flag was reset state rather than a permanent disable. All future source builds appended an empty activation map; a mocked all-layer audit proved no meter read or activation, while submitted sources remained unchanged.

Recorded measurement:

    IMPORTANT NEW SOURCE CORRECTION: _FAST_RESCUE_ON=False wasINITIAL/resetstate, NOT permanentdisable. Inherited _w5_port_begin canenablepaidBLASrouteatL12–14basedonresidual. HistoricallyauthorizedOct17, butnewerMattNOgovernorsinstructionhasnograndfatherexceptionfound. Lead disclosedthismisreadtoMatt; BOTHalready-uploaded#337393/#337396retaininheritedmechanism. Jointagent buildingpermanent `_FAST_RESCUE_FRAC={}` tailandnewimmutablecandidate/releasecompanions, provingidenticalno-rescue32paths; DONOTmutatefired/runningsources. CarrypermanentdisableintoALLfuturebuilds/releases. Noadditionalfiremerelytosubmitanequivalentcandidate; awaitverifiedimprovingphysics. Leadowns follow-through.
    PERMANENT NO-GOVERNOR FIX COMPLETE: A/tail_disable_fast_rescue_astra_v1.py be4293e4 appends final _FAST_RESCUE_FRAC={}; mocked all16 layers proves no meter read/activation. Current K/make_stack_v1.sh 1d5b352c pins this tail LAST in both new sources, refuses collisions, compiles before exclusive writes. Old fired/running files untouched. A/MAKE_STACK_NOGOV_CHANGE_v1.json and FAST_RESCUE_DISABLED_VARIANTS_v1.json. Future release MUST use disabled companion and original release_min gates. All32 bank/micro/v4a saved cumulative rescue counts0; final resetFalse alone is NOT proof of inactivity. A/FAST_RESCUE_CONTROL_AUDIT_v1.json. No score gain claimed for disable.

## Programs

- [tail_disable_fast_rescue_astra_v1.py](tail_disable_fast_rescue_astra_v1.py)
- [make_stack_v1.sh](make_stack_v1.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6281](../../TEAM_LOG_2026-10-01_to_10-22.md#L6281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6289](../../TEAM_LOG_2026-10-01_to_10-22.md#L6289)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FAST_RESCUE_CONTROL_AUDIT_v1.json](FAST_RESCUE_CONTROL_AUDIT_v1.json) · result
- [MAKE_STACK_NOGOV_CHANGE_v1.json](MAKE_STACK_NOGOV_CHANGE_v1.json) · result
- [FAST_RESCUE_DISABLED_VARIANTS_v1.json](FAST_RESCUE_DISABLED_VARIANTS_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-false-initial-rescue-flag-did-not-disable-residual-triggered) · [Copy provenance](../../../PROVENANCE.json)

