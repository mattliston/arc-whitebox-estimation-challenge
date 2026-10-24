# vm_work — work pulled back from the GCP VM (arc-worker, c4-standard-48) before shutdown

Pulled 2026-10-22T01:35:41Z by the lead session with rsync over SSH (archive mode, timestamps preserved, read-only on the VM; nothing on the VM was deleted or modified). Matt's instruction: pull the relevant work into one directory for a later organizing pass; no big data files beyond the one capture set the released coefficients were fitted on.

## Layout (VM paths preserved)

| directory | VM source | what it is | files | size |
|---|---|---|---|---|
| `arc-runs/` | `/home/arcworker/arc-runs/` minus `*.gz *.bin *.npz *.npy` | every run's manifests, COMPLETE/SCORE records, controller and per-rank logs, board scans (`board/`), the 73 root scripts (queue runner, paired scorer, polls), 114 candidate estimator files | 33,777 | 1.50 GB |
| `arc-runs/onpolicy_first128_full90_remaining_v1/` | same path, complete | the on-policy capture set (90 training networks) that `full90_BLOCKS_COEFFICIENTS_v1.json` (the readout of #337619 / #337641) was fitted on | 1,392 blobs added | 4.03 GB |
| `arc-workspace-delta/pure_closure_writeup/` | `/home/arcworker/arc-workspace/pure_closure_writeup/` | ONLY the files that did not exist on the Mac (VM-generated scores, plans, logs, candidates) plus 630 tiny files whose content differed; the rest of the workspace was a byte-identical mirror of the Mac tree and was not pulled. File list: `_pull_logs/workspace_delta_file_list.txt` | 3,844 | 253 MB |
| `arc-envs/` | `/home/arcworker/arc-envs/` | the client / judge / server virtualenvs (flopscope-client 0.12.3, flopscope + flopscope-server 0.12.3, numpy 2.2.6, Python 3.10.12) and `PIP_FREEZE_*.txt` | 8,837 | 207 MB |
| `home/` | `/home/arcworker/arc_truth_response90_payload_v1.tar` | truth-response payload tarball | 1 | 48 MB |
| `_pull_logs/` | — | rsync statistics per stage, timeline, checksum verification, VM system info | | |

Total: about 5.9 GB, 47,862 files. `SHA256SUMS.txt` covers every file.

## Deliberately NOT pulled (still on the VM until it is stopped)

| what | size | why |
|---|---|---|
| all other raw captures and stores in `arc-runs` (`*.gz` on-policy captures for folds 0-4 and head90, iter2, full-state, reconstruct, fresh-net stores; `*.bin`, `*.npz`, `*.npy`) | ~46 GB | big data files, regenerable from the deployed estimator + seeds + the capture scripts (pulled) at hours of 48-core compute |
| VM-only `*.npy` / `*.npz` tensors under the workspace (exported on-policy feature tensors/anchors, VM score stores) | 7.9 GB (877 files) | same; list in `_pull_logs/workspace_delta_file_list.txt`'s companion `ws_absent` listing (tier 3 in the plan) |
| the rest of `arc-workspace` | ~3.7 GB | byte-identical to files already on the Mac |

## Verification

See `_pull_logs/VERIFY_CHECKSUMS.txt` (rsync checksum dry-run of every pulled tier against the VM; a listed file name = a difference) and `_pull_logs/PULL_TIMELINE.txt` (per-stage rsync exit codes and byte counts; the tier1c exit 127 was the first pip-freeze attempt, redone successfully into `arc-envs/PIP_FREEZE_*.txt`).
