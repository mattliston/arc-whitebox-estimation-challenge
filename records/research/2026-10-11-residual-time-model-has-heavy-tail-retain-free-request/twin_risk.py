"""TAG opus-release (2026-10-11, lead's task (b)): loss risk of a TWIN that carries extra requests over the safe file, and the pair's
expected gain. No session.

Model (loss_risk.py's, re-priced on our six graded S44-S46 runs): a fresh run's worst network = mean residual x our usual worst / mean x
the run's excursion; the file loses a network when that exceeds 0.4 s. mean residual = requests per network (mean of stage C's first
network after the setup and warm network) x the platform's price of a request; usual = our graded runs' worst / mean; excursions = the
analyst's 729 leader-sized public runs (O/analyst/a19_page_list.json 'excess'; 'local' = against the team's 10 neighbouring runs,
'week' = against the team's whole record). A run of N networks: P_N = 1 - (1 - P_100)^(N / 100).
The pair: only the better designated file is ranked, so a twin that loses a network falls back to the safe file: expected gain =
(1 - P_twin) x the twin's bill cut (score = error x bill share; the twin's error equals the safe file's).
usage: twin_risk.py [base=<safe requests>] [price=<microseconds>] [usual=<worst/mean>] [N=100,300]"""
import json
from pathlib import Path
import statistics as st
import sys

R = Path(__file__).resolve().parent; O = R.parent
LIMIT = 0.400
GRADED = {'335147': ('S44 safe', 'c_s44_safe_pz'), '335148': ('S44 twin', 'c_s44_primary'), '335213': ('S45 safe', 'c_s45_safe_pz'),
          '335215': ('S45 twin', 'c_s45_primary'), '335259': ('S46 safe', 'c_s46_safe_pz'), '335261': ('S46 twin', 'c_s46_eprimary')}
args = dict(x.split('=', 1) for x in sys.argv[1:])
PL = json.loads((O/'analyst'/'a19_page_list.json').read_text())
E = {'local': sorted(v[1] for v in PL['excess'].values()), 'week': sorted(v[0] for v in PL['excess'].values())}
survive = lambda e, t: sum(1 for x in e if x > t)/len(e)

prices, ratios = [], []
print('graded runs (residual on the platform; requests = stage C first / warm):')
for sid, (name, c) in GRADED.items():
    z = json.loads((O/'board_evals'/f'eval_{sid}.json').read_text()); res = (z.get('evaluation') or z)['results']
    r = [m['telemetry']['residual_wall_time_s'] for m in res['per_mlp'] if m.get('telemetry')]
    req = [json.loads(x)['requests'] for x in (R/'runs'/c/'C0'/'rows.jsonl').read_text().splitlines()][:2]
    price = st.mean(r)/(sum(req)/2); prices.append(price); ratios.append(max(r)/st.mean(r))
    print(f"  {name:9s} #{sid}: {req[0]:,} / {req[1]:,} requests, residual mean {st.mean(r):.4f} s, worst {max(r):.4f} s "
          f"(x{max(r)/st.mean(r):.3f}), {1e6*price:.3f} us a request, lost {sum(1 for m in res['per_mlp'] if m.get('error_code') not in (None, 'SUCCESS'))}")
price = float(args.get('price', 1e6*st.median(prices)))*1e-6
usual = float(args.get('usual', st.median(ratios)))
base = float(args.get('base', 42970))
Ns = [int(x) for x in args.get('N', '100,300').split(',')]
print(f"price {1e6*price:.3f} us a request (median of the six; slowest {1e6*max(prices):.3f}), usual worst / mean {usual:.3f} "
      f"(median of the six; range {min(ratios):.3f}-{max(ratios):.3f}), base = the safe file's {base:,.0f} requests")


def p_loss(requests, reading):
    head = LIMIT/(requests*price*usual)
    return survive(E[reading], head)


print(f"{'extra':>7s} {'requests':>9s} {'worst s':>8s} " + ' '.join(f"{'P' + str(n) + ' ' + rd:>11s}" for n in Ns for rd in ('local', 'week'))
      + '   expected pair gain at -0.43 / -0.60 / -0.70% bill (N = 100, local .. week)')
for extra in (0, 500, 1000, 1500, 2000, 2500, 3000):
    q = base + extra
    p = {rd: p_loss(q, rd) for rd in ('local', 'week')}
    cells = ' '.join(f"{100*(1 - (1 - p[rd])**(n/100)):10.2f}%" for n in Ns for rd in ('local', 'week'))
    gains = ' / '.join(f"{c*(1 - p['local']):.3f}..{c*(1 - p['week']):.3f}%" for c in (0.43, 0.60, 0.70))
    print(f"{extra:+7,d} {q:9,.0f} {q*price*usual:8.4f} {cells}   {gains}")
