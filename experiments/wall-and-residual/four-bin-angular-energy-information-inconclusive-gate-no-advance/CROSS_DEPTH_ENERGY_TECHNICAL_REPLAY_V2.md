# Source-freeze incident and identical technical replay

Recorded September 6, 2026 PDT, before launching replay v2 or inspecting any
energy-bin accuracy results. Phase 2 only. Root owns both replays and judgment;
the two independent agents are now assigned separate, read-only idea research.

## Preserved failure

The v1 A worker qualified before sampling at 20:08:57.413767 UTC. A late agent
provenance-only edit to `acquire_cross_depth_energy_v1.py` landed at 20:09:44 UTC,
adding the judge source to its provenance tuple. This was after launch despite
the source-freeze instruction. No math, RNG, bins, counts or reductions changed.

The worker drew the fixed 8,000,000 samples, then correctly rejected the changed
source at 20:11:33.068564 UTC (156.362565 seconds). Its failure and progress files
remain preserved under `cross_depth_energy_r35_A_v1.*`. No NPZ or acquisition
receipt was produced; no new A/B accuracy result was inspected. The worker was
not made to pass by restoring its old source hash. It is disqualified.

## Frozen technical replay, not another statistical arm

The additive v2 acquisition wrapper pins the now-reviewed parent acquisition
SHA-256 `5cc845a25c3c2b8a58ee62fd611e828f89fafe2e05e27a1e9b2a2083850cf0a7`.
It changes only the artifact suffix, schema version and provenance list. The
parent's complete numerical program and fixed stream A/B sampling laws remain
unchanged. Both wrappers, parents, statistics module and this document are
hashed before sampling and rechecked before artifacts are written.

The additive v2 judge pins the independently reviewed v1 judge SHA-256
`f43a7d2c0c3e35b6324ae9aed684d18e03fcc629e7f936075cf26ccaa85e974f` and changes
only input/output identities, schema versions and provenance list. It accepts
only qualified `cross_depth_energy_r35_A_v2.npz` and `_B_v2.npz` with their
matching immutable receipts. It never loads the failed v1 artifacts as data.

Run A once, then B only if A qualifies. A reproduces the same original sample
sequence; it is not extra effective sample size or a new seed. B is the second
already-prespecified stream. Existing V1 sections 1–5 and 8 plus amendments
V2/V3/V4 govern every mathematical gate. No change to samples, batches, bins,
thresholds, layers, metrics, jackknife, comparator or decision criteria.

Launch at nice 10, three CPU threads, one MPS job, with MPS fallback disabled.
NumPy-only judgment follows successful qualification of both streams. No fit,
candidate, panel or submission is implied by this diagnostic. Acquisition and
judgment sources must not be edited while either job is active.
