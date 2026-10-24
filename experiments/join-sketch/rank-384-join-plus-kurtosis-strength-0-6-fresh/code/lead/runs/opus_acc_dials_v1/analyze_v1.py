"""[opus-acc] paired log-changes vs base: final MSE, mean-of-L12-15 MSE, bill; per seed and mean."""
import json, math, sys
from pathlib import Path
from collections import defaultdict
R = Path(__file__).resolve().parent/'res'; base = sys.argv[1] if len(sys.argv) > 1 else 'base'
rows = defaultdict(dict)
for p in R.glob('*.json'):
    if p.name.endswith('_guard.json'): continue
    d = json.loads(p.read_text()); tag = p.stem.rsplit('_', 1)[0]; rows[tag][d['seed']] = d
B = rows.get(base, {})
print(f"{'arm':<14}{'n':>3}{'dMSE15%':>10}{'per-seed':>34}{'dL12-14%':>10}{'dbill%':>9}  pred==base")
for tag in sorted(rows):
    if tag == base: continue
    seeds = [s for s in rows[tag] if s in B]
    if not seeds: continue
    dm = [100*math.log(rows[tag][s]['mse_final']/B[s]['mse_final']) for s in seeds]
    dl = [100*math.log(sum(rows[tag][s]['mse_layers'][12:15])/sum(B[s]['mse_layers'][12:15])) for s in seeds]
    db = [100*math.log(rows[tag][s]['flops']/B[s]['flops']) for s in seeds]
    same = sum(rows[tag][s]['pred_sha'] == B[s]['pred_sha'] for s in seeds)
    print(f"{tag:<14}{len(seeds):>3}{sum(dm)/len(dm):>10.3f}{' '.join(f'{x:+.3f}' for x in dm):>34}{sum(dl)/len(dl):>10.3f}{sum(db)/len(db):>9.3f}  {same}")
