# Hybrid numerical qualification PASS — not an accuracy result

2026-09-09 UTC / September8 Pacific. Phase2 only. All three fixed stages
and their independent external watchdogs completed and passed. Main checked
each watchdog before progressing and rechecked the frozen watchdog-body SHA.
The earlier all32-MPS CO attempt remains failed and unmodified.

## Measured numerical result

On TRAIN35 and100, using the exact existing null,+/-.025,+/-.05 FP32 P
payloads and original synthetic cotangents:

- Both disabled-adapter nulls exactly reproduce saved CPU all16 output bytes
  and all47 ordered selector records.
- All10 hybrid all16 outputs are byte-identical to their paired saved CPU
  outputs. All4 prescribed scalar secants therefore have0 relative discrepancy.
- Both CPU and both hybrid checkpointed null primals exactly reproduce their
  own forward outputs. All observed prevariance and ordered-selection replay
  records are unchanged; every variance remains in the prescribed domain.
- The complete45200-coordinate P gradients on both nets have0 measured L2
  discrepancy, including all11 parameter groups. Layer0 inactive rows are
  exactly zero; frozen normalizers acquire no gradients. This is equivalence
  on these fixed qualification cases, not a universal derivative theorem.

The hybrid actually executed30 offloaded products per ordinary forward and
60 including checkpoint replay during each VJP. Its logical explicit-helper
traffic was3.1571GB/3.1469GB forward and6.3143GB/6.2937GB with replay on35/100.
Those counts exclude automatic backward copies and allocator staging.

## Resources and throughput

| Stage | Child wall | Child peakRSS | Maximum sampled MPS driver |
| --- | ---: | ---: | ---: |
| Hybrid forward,12 calls including disabled controls |134.44s|6.4603GB|2.1815GB|
| CPU full-P gradients,2 nets |72.98s|9.2499GB|0|
| Hybrid full-P gradients,2 nets |71.65s|9.3362GB|2.2790GB|

CPU per-net forward/backward36.58s/34.14s; hybrid34.44s/34.68s. The principal
result is numerical fidelity, NOT substantial training acceleration. Extrapolating
960 such serial graphs is roughly9.2hours before validation/overhead, not the
old all32-MPS7200s fit budget. No longer fit or parallel fit is admitted here.
Every stage remained below900s/12GB host/16GB driver; supervisors observed no
termination and no swap growth. Final swap760.75MiB; no numerical workers left.
Sampled memory is not an absolute-peak/cgroup certificate.

## Receipt seals

- Forward d6edae6b5828e09939d5afb7be7a3bc619f8e384fd909cbf105b26331a2943fc
- Forward watchdog288ed547f544a9ec749129f66e88e0655dcca6ec5125fec60d1e7885487d8c1d
- CPU VJP acfe54dfaffc19ad1d1c2c56c2349750ceff0cf4a587b72638827c7767abea0e
- CPU watchdog4d880e1b185767feb3db7543c506bf25af832fe83cbdc209822289f8f444261f
- Hybrid VJP f821503ed45372d6b9a14b363914ab255d643b4ae2d90b3eacb389f61d6892ec
- Hybrid watchdogf2d319892277473a13cc9d1a21b4a51754682842ab391dc55b9b301cebc19afd

Sources are pinned in LAUNCH_REVIEW.md and every stage receipt. No target
arrays, optimizer, fit, native candidate or new score was produced.

## Next decision

This clears the specified CPU/hybrid numerical gate. It does not qualify
native d50 under changed P, or establish learnable raw-error reduction.
Next: a separately specified native bridge for these SAME sealed P payloads,
using literal-only source replacements and the qualified official-input,
complete-bill35/100/repeat35 loop. Retain all raw/native/head compatibility,
resource and release checks. Any subsequent fit needs its own prospective
affordable plan, complete adjusted-objective gate and fixed terminal selection.

Confirmed bank330390 remains4.53886349950395e-9 adjusted,2.120838555441651e-8
raw. Goal stays official strictly<1e-9, with banks below4e-9,3e-9,2e-9.
