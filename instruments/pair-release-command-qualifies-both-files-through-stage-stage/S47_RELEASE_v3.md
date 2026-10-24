# S47 release checklist, version 3 (release lane, 2026-10-11 03:55 UTC)

Supersedes S47_RELEASE_v2.md (kept). Target: fire-ready lines within 60 minutes of the engineer's two built files.
Dropped from v2 because they do not protect the score: stage D / worker checks and every designation step (Matt: no time on
designation), the "no other native session" coordination (request counts, setup log, bill and outputs do not depend on load),
the separate no-reviewed stage A pass (one pass lists every new signature against the last fired source), and blocking on this
machine's swap / paging gates (release_gate_v3.py turns those into warnings).

1. Engineer posts SAFE and TWIN files (built, parity done, ten networks judged). The lead decides the physics.
2. Release, one command (from O/release):
   `bash s47_pair.sh <engineer safe .py> <engineer twin .py> s47 ['reviewed sig;sig']`
   - SAFE = safe + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero; TWIN = engineer twin as built (TWIN_FRAGS='...' adds fragments).
   - stage A of both against the last fired S46 sources: any new einsum signature not in the reviewed list stops the script and
     is written to TEAM_LOG for the lead's review (a call must not pass the same operand twice); rerun with tag s47b and the list.
   - per file: stage C, 1.3x slow-setup session, designation_judge.py (twin: --primary), release_gate_v3.py; the script writes
     ONE fire line per file (archive path, archive and candidate sha256, judge numbers) or a DO NOT FIRE line with the errors.
   - watch: tail -F O/release/runs/s47pair_s47.log ; DRY=1 = builds + stage A only (seconds).
3. Lead fires from the fire lines (O/fire/fire_once.py). Release never uploads and never reads key.txt.

Limits that block a fire (designation_judge.py + gate): safe requests first <= 43,800 / warm <= 42,700; twin P(loss) <= 20%
(loss_risk.py, 100 networks); binding <= 18,600 requests and >= 90% of keys bound; projected setup wall <= 4.5 s (<= 4.6 s at 1.3x;
5 s fails the whole submission); projected worst-network wall <= 100 s; slow-setup requests <= +2%; every stage A gate; exact
hashes and archive members. Twin risk pricing: python -B twin_risk.py (S44-S46 graded basis). Control 10-11 03:53Z: DRY run on the
S46 engineer files rebuilt the fired 70f7857f / f89b8951 and passed stage A.
