"""TAG opus-release (2026-10-12 03:15 UTC, lead: the MOST aggressive twin for P(podium) alone, pair = twin + #335411 S48 safe). No session.
Twin options = the fired twin #335451 (s48tx: 44,476 / 42,846 requests first / warm, graded score 0.99858 x S48's) + stackable dials, each with
its measured or estimated bill change and requests; P(loss) = loss_model_v3 first-priced at 7.35 us a network-100 request (6 first networks
per 100); P(podium) = pair_mc_twin.py on S48 #335411's own page (rivals' pairs from the analyst's current scan), safe = S48 at P(loss) 3.03% /
8.81% (100 / 300 networks), our position x = 1.00 / 0.98 / 0.97 scaling both files (S50 or the pair-split bet landing).
usage: max_twin_table.py"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.stats import genpareto
R = Path(__file__).resolve().parent
x = np.array(json.loads((R/'runs'/'net_factors_20261011b.json').read_text())['ours'])
e = x[x > 1.10] - 1.10; c, _, s = genpareto.fit(e, floc=0); S = lambda t: len(e)/len(x)*genpareto.sf(t - 1.10, c, 0, s)
def P(first, warm, N):
    mw = warm*7.35e-6
    return 1 - (1 - S(0.4/(mw*first/warm)))**(0.06*N)*(1 - S(0.4/mw))**(0.94*N)
BASE = (44476, 42846, 0.99858)
DIALS = {'XSG': (-0.077, 465, 465), 'L3off': (-0.06, 2200, 1800), 'HZ7': (-0.02, 281, 7),
         'tilesU': (-0.12, 6537, 0), 'tilesB': (-0.12, 0, 0)}
COMBOS = [(), ('XSG',), ('XSG', 'HZ7'), ('XSG', 'L3off'), ('XSG', 'L3off', 'HZ7'), ('XSG', 'tilesB'), ('XSG', 'L3off', 'tilesB'),
          ('XSG', 'L3off', 'HZ7', 'tilesB'), ('XSG', 'tilesU')]
def mc(f1, l1, f2, l2):
    out = subprocess.run([sys.executable, '-B', str(R/'pair_mc_twin.py'), '335411', f'{f1:.6f}', f'{l1:.4f}', f'{f2:.6f}', f'{l2:.4f}', 'x'],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return float(out.split('P(podium) ')[1].split('%')[0])
rows = []
for combo in COMBOS:
    db = sum(DIALS[k][0] for k in combo); first = BASE[0] + sum(DIALS[k][1] for k in combo); warm = BASE[1] + sum(DIALS[k][2] for k in combo)
    f1 = BASE[2]*(1 + db/100)
    l100, l300 = P(first, warm, 100), P(first, warm, 300)
    pod = {}
    for X in (1.00, 0.98, 0.97):
        pod[X] = (mc(f1*X, l100, X, 0.0303), mc(f1*X, l300, X, 0.0881))
    rows.append(dict(combo='fired twin' + ''.join(' + ' + k for k in combo), bill_vs_fired=db, f1=f1, first=first, warm=warm, l100=l100, l300=l300, pod=pod))
    r = rows[-1]
    print(f"{r['combo']:42s} bill {db:+.3f}% (score x{f1:.5f} of S48)  {first:,} / {warm:,}  P(loss) {100*l100:.1f}% / {100*l300:.1f}%  "
          + '  '.join(f"x{X:.2f}: {pod[X][0]:.1f}% / {pod[X][1]:.1f}%" for X in pod), flush=True)
(R/'runs'/'max_twin_table.json').write_text(json.dumps(rows, indent=1))
