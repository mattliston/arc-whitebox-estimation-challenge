#!/bin/bash
# TAG opus-physicist T121b trigger (lead 04:2xZ: S50's first check NOT ACCEPTED; RESERVE CHECK on A38 7100424-455 pooled with 520-551, accept iff pooled < -SE;
# GO in advance): start S50 stage 2 exactly as armed in t121_trigger.sh the moment the trainer's next 'S50 ... CHECK' line (after TEAM_LOG line     1317) carries a
# clear ACCEPTED verdict ('-> ACCEPTED', '= ACCEPTED' or ': ACCEPTED'); NOT ACCEPTED or anything unclear: start nothing, log one line (only lines that carry a verdict word count: setup lines are skipped). Stop: STOP_T121B.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
P=$O/physicist
while true; do
  [ -f $P/STOP_T121B ] && exit 0
  V=$(tail -n +1318 $O/TEAM_LOG.md | grep -E "^\[trainer\].*S50.*CHECK.*ACCEPTED" | head -1)
  [ -n "$V" ] && break
  sleep 15
done
if echo "$V" | grep -q "NOT ACCEPTED"; then
  echo "[physicist] $(date -u +%Y-%m-%dT%H:%M:%SZ) S50 STAGE 2 NOT STARTED: the trainer's reserve check line says NOT ACCEPTED (lead's conditional GO: do nothing)." >> $O/TEAM_LOG.md; exit 0
fi
if ! echo "$V" | grep -qE "(->|=|:) *ACCEPTED"; then
  echo "[physicist] $(date -u +%Y-%m-%dT%H:%M:%SZ) S50 STAGE 2 NOT STARTED: the trainer's S50 check line carries no clear ACCEPTED verdict; waiting for the lead." >> $O/TEAM_LOG.md; exit 0
fi
MAN=$P/t121_S50_stage2/manifest.json
nohup bash $P/t112_chain_v13.sh $MAN s50a $(seq 7100552 7100567) > /dev/null 2>&1 &
nohup bash $P/t112_chain_v13.sh $MAN s50b $(seq 7100568 7100583) > /dev/null 2>&1 &
echo "[physicist] $(date -u +%Y-%m-%dT%H:%M:%SZ) S50 STAGE 2 STARTED (trainer's reserve check ACCEPTED, pooled 64; lead's GO in advance 04:2xZ): judge v13 (worker_generated_v11 / otr_m_v5, sha-pinned) on SEALED A40 7100552-583, base = the trainer's a_RS50C_ab (MS point) vs candidate a_RS50C_pt (S50 frozen point), each seed's pair in one process (3 threads) under the release memory guard, 2 chains of 16 (physicist/t121_S50_stage2; logs physicist/t112_chain_s50a.log / s50b.log); table: physicist/t112_table.py physicist/t121_S50_stage2 trainer/run_RAD_S50_gen/results a_RS50C_ab a_RS50C_pt." >> $O/TEAM_LOG.md
