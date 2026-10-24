#!/bin/bash
# TAG analyst A16 odds waiter v10 (lead 2026-10-12 23:0xZ; replaces v9: the pair is now the first stack twin s51k1tw #335633 primary at the release's
# s51k1tw verdict 4.41% per 100 + S51 safe #335618 fallback at 2.5%; single-thread BLAS). v9 text: S51 twin #335621 at 4.92% + #335618 at 2.5%. v8 text: S50 twin #335505 primary at 4.3% + S50 safe #335508 at 2.49%. v7 text: the pair was twin v2 #335483 primary at 4.2% + S48 #335411 safe at 3.0%. v6 text: v5 (FIXED pair #335451 primary at the release's 3.97% per 100 + #335411 safe
# at 3.0%, rivals' risks from a16_loss_risk.json) on a16_pair_mc_v3.py, plus P(we finish ahead of Luna) at x1.000 / x0.995 / x0.99.
# After the FIRST scan seen and after every scan whose top five (names or scores) differ from the previous one: wait until no a11_rescan runs
# (it rebuilds a16_loss_risk.json after its log line), run the odds at x1.000 / 0.995 / 0.99 / 0.985 / 0.98 / 0.975 / 0.970 of #335451, once with
# the release's risks and once (x1.000 only) with a16's own risks for our two files, and append ONE TEAM_LOG line (date -u).
# Reads only; writes a16_pair_mc_v3.json (the script's own output) and the TEAM_LOG line. Checks every 5 min. Stop: touch O/analyst/A16_ODDS_STOP.
# usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nohup bash a16_odds_waiter_v10.sh >> a16_odds_waiter_v10.out 2>&1 &
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst
O=$A/..; PY=/Users/matt/anaconda3/bin/python; first=1; FACTORS="1.00 0.995 0.99 0.985 0.98 0.975 0.97"
SID=335618; LSR=0.025; PID=335633; LPR=0.0441
top5() { echo "$1" | sed -n 's/.*top six \([^|]*\).*/\1/p' | cut -d, -f1-5; }
last=$(tail -n 1 $A/a11_rescan.log)
while true; do
  [ -e $A/A16_ODDS_STOP ] && { echo "$(date -u +%FT%TZ) A16_ODDS_STOP present: waiter ends"; exit 0; }
  sleep 300
  now=$(tail -n 1 $A/a11_rescan.log)
  [ "$now" = "$last" ] && continue
  while pgrep -f "a11_rescan" > /dev/null; do sleep 30; done
  if [ $first -eq 1 ] || [ "$(top5 "$now")" != "$(top5 "$last")" ]; then
    first=0; echo "$(date -u +%FT%TZ) reporting after the scan: $(top5 "$now")"
    read lsa lpa <<< "$($PY -c "
import json; d = json.load(open('$A/a16_loss_risk.json')); E = {x['id']: x['p_any'] for x in d['entries_risk']['mliston']}; print(E.get($SID, $LSR), E.get($PID, $LPR))")"
    ( cd $A && $PY a16_pair_mc_v3.py $SID $lsa $PID $lpa 1.00 > $A/a16_odds_last_own.txt 2>&1 && cp $A/a16_pair_mc_v3.json $A/a16_pair_mc_v3_own_last.json ) \
    && ( cd $A && $PY a16_pair_mc_v3.py $SID $LSR $PID $LPR $FACTORS > $A/a16_odds_last.txt 2>&1 ) && $PY - <<EOF >> $O/TEAM_LOG.md
import json, subprocess
d = json.load(open('$A/a16_pair_mc_v3.json')); o = json.load(open('$A/a16_pair_mc_v3_own_last.json'))['rows'][0]
t = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(); base = d['rows'][0]['score']
riv = {k: v[0][1] for k, v in d['pairs'].items()}
stand = ', '.join(f"{k} {riv[k]:.4e} ({(riv[k] / base - 1) * 100:+.2f}% from us)" for k in ('Luna', 'oliver_chen') if k in riv)
cells = '; '.join(f"x{r['factor']:.3f} {r['independent']['p_podium'] * 100:.1f}% ({r['J2W together']['p_podium'] * 100:.1f}%)" for r in d['rows'])
ahead = ', '.join(f"x{r['factor']:.3f} {r['independent']['p_ahead_luna'] * 100:.1f}%" for r in d['rows'][:3])
p1 = max(r[c]['p_first'] for r in d['rows'] for c in ('independent', 'J2W together'))
print(f"[analyst] {t} PAIR ODDS (automatic, a16_odds_waiter_v10.sh, after the scan: $(top5 "$now" | cut -c1-160)): pair #$PID primary ({float('$LPR') * 100:.2f}% per 100, release) + #$SID safe ({float('$LSR') * 100:.1f}%); our #$PID {base:.4e} vs {stand}. P(we finish ahead of Luna): {ahead}. Podium by our score x of #$PID: {cells}; P(1st) at most {p1 * 100:.1f}%. Brackets = J2W's two files fail together. x1.000 with a16's own risks for our files ({float('$lpa') * 100:.1f}% / {float('$lsa') * 100:.1f}%): podium {o['independent']['p_podium'] * 100:.1f}%, ahead of Luna {o['independent']['p_ahead_luna'] * 100:.1f}%.")
EOF
  fi
  last=$now
done
