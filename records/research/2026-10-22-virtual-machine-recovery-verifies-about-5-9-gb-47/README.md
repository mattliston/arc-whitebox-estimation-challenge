# Virtual-machine recovery verifies about 5.9 GB and 47,862 files against source checksums

2026-10-22T01:35:41Z · tool · informational · measured

The recovery retained run records, workspace changes, the full-90 capture set, environment and truth files. A checksum comparison found no remaining differences after topping up records written during transfer; large regenerable raw stores and tensor caches were deliberately not copied.

Recorded measurement:

    - 2026-10-22T01:35:41Z [fable-lead] VM PULL (Matt: stop the VM without losing work; approved tiers 1+2+4a): vm_work/ = arc-runs records 1.50 GB (33,777 files), workspace delta 253 MB (3,844 VM-only files), full90 capture set 4.03 GB, arc-envs 207 MB (flopscope 0.12.3 client/server), truth tar 48 MB; ~5.9 GB / 47,862 files; all rsync stages exit 0; checksum verification + SHA256SUMS running. NOT pulled: ~46 GB raw captures/stores + 7.9 GB workspace tensors (regenerable).
    - 2026-10-22T01:37:01Z [fable-lead] VM PULL VERIFIED: checksum dry-run vs the VM = 0 differences in all tiers (the 28 flagged files were J2W board scans written during the pull; topped up). SHA256SUMS written. J2W #337666 GRADED 1.423e-9 (not a threat); #337675/682/683 queued. J2W watcher moved to the Mac (.venv python, scratchpad j2w_local) so the VM can be stopped.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7601](../../TEAM_LOG_2026-10-01_to_10-22.md#L7601)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7602](../../TEAM_LOG_2026-10-01_to_10-22.md#L7602)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-22-virtual-machine-recovery-verifies-about-5-9-gb-47) · [Copy provenance](../../../PROVENANCE.json)

