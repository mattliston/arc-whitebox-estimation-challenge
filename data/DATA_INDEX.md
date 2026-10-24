# Data index

The repository holds code, records and results. The large datasets behind them are not in git. This index lists
each one with its origin in the working repository, its size as measured on 2026-10-22, what produced it, what it
reproduces, and its publication status. No dataset upload is included in this repository build.

| dataset | origin (working repository) | size | what produced it | what it reproduces | published at |
|---|---|---|---|---|---|
| On-policy capture set, 90 training networks (`onpolicy_first128_full90_remaining_v1`) | `vm_work/arc-runs/onpolicy_first128_full90_remaining_v1/` | 4.03 GB, 1,392 blobs | the parent estimator of submissions #337619 and #337641 run on 90 fresh networks with the capture hook (`release/common/readout/first128_onpolicy_hook_v2.py`, `client_capture_v1.py`) | the exact design matrices behind `release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json`, the readout coefficients of the two designated submissions | pending |
| VM run records (`arc-runs` without tensors) | `vm_work/arc-runs/` | 1.50 GB, 33,777 files | every VM run's manifests, COMPLETE and SCORE records, controller and per-rank logs, board scans, 73 root scripts, 114 candidate estimator files | the paired-store scores recorded on 2026-10-14 through 2026-10-21 (VM32) and the queue runner's provenance | pending |
| On-policy feature tensors and anchors | `.../opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/` (`*.npy`, `*.npz`) | 1.04 GB (directory total) | export of the captured features per layer (`export_first128_onpolicy_tensors_v1.py`) | the ridge fits of 2026-10-20 and 10-21 (gate design, hidden x gate, first 128 encoder features x hidden) | pending |
| Join-sketch paired score stores | `.../opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/` | 5.9 GB (directory total) | paired scoring of the join-sketch column counts on the 32-rank store | every paired measurement behind the kc squeeze and the 32-rank law | pending |
| Release archives of October | `.../opus_reexam_20261001_v1/release/runs/*/submission.tar.gz` | 260 archives, 0.8 GB (directory total with payloads) | the release gate of every October candidate (`whest package`) | byte-exact artifacts of every October submission and of the gated candidates that were never fired | the fired ones are in `submissions/<id>/submission.tar.gz` where under 25 MB; the rest pending |
| VM environments | `vm_work/arc-envs/` | 207 MB, 8,837 files | the client, judge and server virtualenvs (flopscope-client 0.12.3, flopscope and flopscope-server 0.12.3, numpy 2.2.6, Python 3.10.12) with `PIP_FREEZE_*.txt` | the exact judging environment of the local gates | `PIP_FREEZE_*.txt` only; the environments are rebuilt from them |
| Truth-response payload | `vm_work/home/arc_truth_response90_payload_v1.tar` | 48 MB | Monte Carlo truth of the 90 training networks | the targets of the on-policy fits | pending |
| Data gate (2026-08-23 program) | `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/` | 72 GB | truth and feature generation for the learned residual observer | the August observer and gate experiments | not planned (regenerable from the public generator and the scripts) |
| Learned residual observer design data | `.../p2_width1024_depth16/learned_residual_observer_design/` | 51 GB | training sets of the August learned observer | the August observer experiments | not planned (regenerable) |

Not pulled from the VM before it was stopped (see the [VM pull manifest](../records/VM_PULL_MANIFEST.md)): about
46 GB of further on-policy captures (folds 0 to 4, head90, iteration 2, full-state, reconstruction and fresh-network
stores) and 7.9 GB of VM-only tensors. The historical pull manifest describes them as regenerable from the deployed estimator, seeds and capture
scripts, with an estimate of hours on 48 cores. This release has not rerun that reconstruction; the anatomy
documents additional data and environment dependencies.

Sizes are `du` totals of the directories named; the tensor share is most of each total.
