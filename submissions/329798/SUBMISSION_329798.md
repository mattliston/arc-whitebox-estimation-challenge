# Phase 2 checkpoint submission 329798

Accepted at 2026-09-04T23:11:57.901Z (server timestamp).
Candidate: `candidate_radial_gnn_splitread_v1.py`.
Archive: `radial_gnn_splitread_v1.tar.gz`; only estimator.py and manifest.json.

Independent4M noise-corrected public8 adjusted1.1548055732603588e-8,
raw3.388356351160948e-8; virgin6 adjusted1.0288461900146e-8,
raw3.018774417983994e-8. Utilization .3408 on each local panel entry.
Adjusted gains vs recovered329790 candidate:28.04% public,20.95% virgin.

Standard validation and official-version local35/100/35 resource/parity gate
passed. See `radial_splitread_native_parity_v1.json` and
`RESEARCH_RECEIPTS_2026_09_04.md`. Cold residual-time margin is small:
.394s with diagnostic copies; clean hot residual .330s. PeakRSS7.22GB.
Rank35 repeat output byte-identical; head correction difference from NumPy
at most1.35e-8 on the two checked nets.

Official grade: adjusted1.747652393096604e-8, raw3.17094401225404e-8.
This is WORSE adjusted than329790 despite comparable raw accuracy. The
implied utilization is about.551, not the local.341. A local/remote billing
or dispatch mismatch is under investigation; the panel's cost did not
transfer. Do not treat the projected adjusted gains as official gains.
No promotion or commit. The approximately4e-9 goal remains unmet.
