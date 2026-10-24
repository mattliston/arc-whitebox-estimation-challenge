#!/bin/bash
# TAG opus-release S48 SAFE v2 (2026-10-11 18:20 UTC; s47_safe_v4.sh with the lead's override FIRST_PRICED=1, logged in TEAM_LOG at the
# start: the first-network count is a priced risk, not a 43,800 cap; v1 = s48_safe.sh; lead: S48 = S47e + MS literals + r_u 320 at L10-14 + one-direction star + read
# column cut, built by the engineer in engineer/a62_modules/s48/). One command once the engineer hands over the file:
#   bash O/release/s48_safe.sh <engineer S48 file .py>
# = s47_safe_v3.sh with tag s48s: SAFE = file + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero; stage A against the fired S47e #335349
# source (a new einsum signature stops it with a TEAM_LOG line for the lead's review); stage C; 1.3x slow-setup session; designation
# judge; release_gate_v3; ONE TEAM_LOG line with counts, P(loss) (loss_model_v3 basis + wall) and the exact fire_once.py command, or the
# gate's errors. The lead fires only after the clean judge on the sealed seeds. watch: tail -F O/release/runs/s47safe_s48s.log
set -u
cd "$(dirname "$0")"
[ $# -ge 1 ] && [ -f "$1" ] || { echo "usage: s48_safe.sh <engineer S48 file .py>"; exit 2; }
FT=s48s_$(date -u +%Y%m%d)
[ -e ../fire/$FT ] && FT=${FT}b
echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) S48 GATE OVERRIDE LOGGED (lead 18:0xZ): for this file the first-network request count is a PRICED risk (P(loss) per 100 networks, loss_model_v3 with first networks priced separately, must be <= 10%), not the 43,800 cap; the warm cap 42,700 and every other gate stay (designation_judge --first-priced, release_gate_v4)." >> ../TEAM_LOG.md
FIRST_PRICED=1 exec bash s47_safe_v4.sh "$1" s48s "$FT" "S48 safe: S47e + MS literals + r_u 320 at L10-14 + one-direction star + read column cut (engineer $(shasum -a 256 "$1" | cut -c1-8)) + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero"
