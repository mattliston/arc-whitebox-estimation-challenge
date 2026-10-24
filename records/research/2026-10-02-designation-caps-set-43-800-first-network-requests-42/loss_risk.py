"""TAG opus-release: probability that a fresh 100-network run of a file loses a network on the 0.4 s residual limit, from the
analyst's excursion sample (O/analyst/a19_page_list.json 'excess': for each of 729 leader-sized public runs, the run's
worst / mean residual divided by its team's usual value of that ratio; 'local' = the median over the team's 10 neighbouring
runs, 'week' = over the team's whole record) and our own graded pages.

A fresh run's worst network = (mean residual of the file) x (our usual worst / mean) x (excursion of that run). The file loses
a network when the excursion exceeds  headroom = 0.4 / (mean x usual).  P = share of the 729 excursions above the headroom.
  mean  = requests per network (stage C's count: mean of network 35, first after the setup, and network 100, warm) x the
          platform's price per request in the bank's graded run (bank.json: residual_mean / the requests of the bank's stored
          stage C legs). The price is that run's hour: S40's 7.13 microseconds, S41's 7.44 (3-4% slow);
  usual = our worst / mean: 'last' (default since 18:40Z) = the newest graded file's own ratio: the ratio is a property of the
          file (how much a first network costs more than the others: S39 1.149, S40 1.174, S41 1.111 after the diet);
          'last3' = median over the last three graded files, 'record' = median over all our leader-sized runs, or usual=X.
Also printed: the analyst's own convention (headroom of the worst network AS GRADED, for graded files) and the request count
at which each reading reaches a target probability.
usage: loss_risk.py [usual=last|last3|record|X] [target=0.011] name=requests [name=requests ...]
       a graded file may be given as name=<eval json>: its measured mean and worst are used."""
import json
from pathlib import Path
import statistics as st
import sys

R = Path(__file__).resolve().parent; O = R.parent
LIMIT = 0.400
PL = json.loads((O/'analyst'/'a19_page_list.json').read_text())
E_WEEK = sorted(v[0] for v in PL['excess'].values()); E_LOCAL = sorted(v[1] for v in PL['excess'].values()); N = len(E_LOCAL)
surv = lambda e, t: sum(1 for x in e if x > t)/len(e)


def page(path):
    z = json.loads(Path(path).read_text()); res = (z.get('evaluation') or z)['results']
    r = [m['telemetry']['residual_wall_time_s'] for m in res['per_mlp'] if (m.get('telemetry') or {}).get('residual_wall_time_s') is not None]
    return st.mean(r), max(r)


OURS = {}                                                  # our leader-sized runs: worst / mean residual
for line in (O/'analyst'/'scan_merged.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r.get('http') == 200 and (r.get('team') or r['who']) == 'mliston' and len(r.get('flops') or []) == 100 and r.get('resid') and r['wall'][0] >= 40:
        OURS[r['id']] = r['resid'][1]/r['resid'][0]
for f in sorted((O/'board_evals').glob('eval_3335*.json')):          # graded after the analyst's last scan
    sid = int(f.stem.split('_')[1])
    if sid not in OURS and sid > max(OURS):
        m, w = page(f); OURS[sid] = w/m
ids = sorted(OURS)
USUAL = dict(last=OURS[ids[-1]], last3=st.median(OURS[i] for i in ids[-3:]), record=st.median(OURS.values()))
bank = json.loads((R/'calibration'/'bank.json').read_text())
legs = Path(sorted((R/'calibration').glob('bank_stage_c_paced_*.txt'))[-1].read_text().strip())      # the bank's stored stage C legs
counts = [json.loads(x)['requests'] for leg in sorted(legs.glob('P*/rows.jsonl')) for x in leg.read_text().splitlines()]
PRICE = bank['platform']['residual_mean']/st.mean(counts)      # of the bank's own graded run: S40 7.13 microseconds, S41 7.44 (a slow hour)

args = [a.split('=', 1) for a in sys.argv[1:]]
usual_arg = next((v for k, v in args if k == 'usual'), 'last'); target = float(next((v for k, v in args if k == 'target'), 0.011))
usual = USUAL[usual_arg] if usual_arg in USUAL else float(usual_arg)
print(f"excursion sample: {N} runs. Our worst / mean residual: last three graded files {USUAL['last3']:.3f} "
      f"({', '.join(f'#{i} {OURS[i]:.3f}' for i in ids[-3:])}), whole record ({len(ids)} runs) {USUAL['record']:.3f}. "
      f"Price of a request in the bank's run ({bank['name']}): {PRICE*1e6:.3f} microseconds. usual = {usual:.3f}")
print(f"{'file':34s} {'requests':>8s} {'mean s':>7s} {'usual worst s':>13s} {'headroom':>8s} {'P local':>14s} {'P week':>14s}   as graded: worst s, headroom, P local")
for name, value in args:
    if name in ('usual', 'target'):
        continue
    graded = None
    if value.endswith('.json'):
        mean, worst = page(value); graded = worst; requests = mean/PRICE
    else:
        requests = float(value); mean = requests*PRICE
    t = LIMIT/(mean*usual)
    line = (f"{name:34s} {requests:8.0f} {mean:7.4f} {mean*usual:13.4f} {t:8.3f} {surv(E_LOCAL, t)*100:6.2f}% ({sum(x > t for x in E_LOCAL):2d}) "
            f"{surv(E_WEEK, t)*100:6.2f}% ({sum(x > t for x in E_WEEK):2d})")
    if graded:
        tg = LIMIT/graded; line += f"   {graded:.4f}, {tg:.3f}, {surv(E_LOCAL, tg)*100:.2f}% ({sum(x > tg for x in E_LOCAL)})"
    print(line)
# the request count at which the local reading reaches the target: the smallest headroom with survival <= target
t_need = next(x for x in E_LOCAL if surv(E_LOCAL, x) <= target)
print(f"target {target*100:.1f}% (local reading): headroom {t_need:.3f} -> usual worst network {LIMIT/t_need:.4f} s -> mean residual "
      f"{LIMIT/t_need/usual:.4f} s -> {LIMIT/t_need/usual/PRICE:,.0f} requests per network at usual = {usual:.3f}; "
      + '; '.join(f"at usual {u:.2f}: {LIMIT/t_need/u/PRICE:,.0f}" for u in (1.10, 1.12, 1.15, 1.175)))
