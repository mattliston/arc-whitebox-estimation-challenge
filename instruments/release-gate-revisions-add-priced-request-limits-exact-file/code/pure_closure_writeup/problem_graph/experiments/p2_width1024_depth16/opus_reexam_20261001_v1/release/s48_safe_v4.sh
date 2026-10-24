#!/bin/bash
# TAG opus-lead S48 SAFE v4 (2026-10-11 19:26 UTC) = s48_safe_v3.sh, but through s47_safe_v5.sh with FRAGS=none and the run tag s48t.
# Why: the S48 sources are built on the fired S47e and already carry Rl3_bind128 + Rlvfs_limit_off + Rprices_zero. v3 (tag s48s)
# appended them a second time; the doubled l3_bind128 broke the setup binding (stage C FAIL: setup 592 requests, first network 49,873
# requests). Here the variant is the engineer's file byte for byte. Same override (FIRST_PRICED=1: first-network count priced,
# P(loss) per 100 <= 10%; warm cap 42,700 and every other gate stay).
#   bash O/release/s48_safe_v4.sh <engineer S48 file .py>      watch: tail -F O/release/runs/s47safe_s48t.log
set -u
cd "$(dirname "$0")"
[ $# -ge 1 ] && [ -f "$1" ] || { echo "usage: s48_safe_v4.sh <engineer S48 file .py>"; exit 2; }
FT=s48s_$(date -u +%Y%m%d)
[ -e ../fire/$FT ] && FT=${FT}b
echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) S48 GATES RE-RUN by the lead (s48_safe_v4.sh, tag s48t): the s48s run appended the release fragments a second time to a file that already carries them (doubled l3_bind128 broke the setup binding: stage C FAIL, setup 592 requests, first network 49,873); s48t uses the engineer's file byte for byte (FRAGS=none). Same first-network PRICED override (P(loss) per 100 <= 10%); warm cap 42,700 and every other gate stay. The s48s run's FAIL line is void." >> ../TEAM_LOG.md
FRAGS=none FIRST_PRICED=1 exec bash s47_safe_v5.sh "$1" s48t "$FT" "S48 safe: S47e + MS literals + one-direction star + read column cut a>=-2.0 (engineer $(shasum -a 256 "$1" | cut -c1-8); carries Rl3_bind128 + Rlvfs_limit_off + Rprices_zero)"
