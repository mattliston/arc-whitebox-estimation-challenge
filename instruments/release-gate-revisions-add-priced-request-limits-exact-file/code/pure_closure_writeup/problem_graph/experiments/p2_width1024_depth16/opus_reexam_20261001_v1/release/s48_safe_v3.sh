#!/bin/bash
# TAG opus-lead S48 SAFE v3 (2026-10-11 18:50 UTC) = s48_safe_v2.sh with the fire description corrected: r_u 320 is OUT of S48 (lead 17:34Z),
# so the description names only what the file holds: MS literals + one-direction star + read column cut a >= -2.0. Same override
# (FIRST_PRICED=1: the first-network count is a priced risk, P(loss) per 100 <= 10%; the warm cap 42,700 and every other gate stay).
#   bash O/release/s48_safe_v3.sh <engineer S48 file .py>      watch: tail -F O/release/runs/s47safe_s48s.log
set -u
cd "$(dirname "$0")"
[ $# -ge 1 ] && [ -f "$1" ] || { echo "usage: s48_safe_v3.sh <engineer S48 file .py>"; exit 2; }
FT=s48s_$(date -u +%Y%m%d)
[ -e ../fire/$FT ] && FT=${FT}b
echo "[release] $(date -u +%Y-%m-%dT%H:%M:%SZ) S48 GATE OVERRIDE LOGGED (lead 18:0xZ; run by the lead via s48_safe_v3.sh): for this file the first-network request count is a PRICED risk (P(loss) per 100 networks, loss_model_v3 with first networks priced separately, must be <= 10%), not the 43,800 cap; the warm cap 42,700 and every other gate stay (designation_judge --first-priced, release_gate_v4)." >> ../TEAM_LOG.md
FIRST_PRICED=1 exec bash s47_safe_v4.sh "$1" s48s "$FT" "S48 safe: S47e + MS literals + one-direction star + read column cut a>=-2.0 (engineer $(shasum -a 256 "$1" | cut -c1-8)) + Rl3_bind128 + Rlvfs_limit_off + Rprices_zero"
