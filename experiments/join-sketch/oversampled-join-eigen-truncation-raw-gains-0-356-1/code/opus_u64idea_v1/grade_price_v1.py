"""[opus-ntsched] first/warm deterministic bill + charged ops vs the S2tr base (opus_pushsm_v1/warm/base_7103256.json); also the target's
plain MSE vs the base's (single network, NOT a screen). usage: grade_price_v1.py TAG..."""
import json, math, sys
from pathlib import Path
D = Path(__file__).resolve().parent
base = json.loads((D.parent/'opus_pushsm_v1/warm/base_7103256.json').read_text())
bw = base['rows'][0]
for tag in sys.argv[1:]:
    p = D/'price'/f'{tag}_7103256.json'
    if not p.exists(): print(f'{tag:6s} missing'); continue
    d = json.loads(p.read_text()); w = d['rows'][0]
    f1 = 100 * (d['first']['flops'] / base['first']['flops'] - 1); f2 = 100 * (w['flops'] / bw['flops'] - 1)
    o1 = d['first']['ops'] - base['first']['ops']; o2 = w['ops'] - bw['ops']
    m = 100 * math.log(w['mse_final'] / bw['mse_final']) if 'mse_final' in w and 'mse_final' in bw else float('nan')
    print(f"{tag:6s} first bill {f1:+.3f}% ops {o1:+d} | warm bill {f2:+.3f}% ops {o2:+d} | warm-net plain MSE {m:+.2f}% (1 net) | wall {w.get('wall', 0):.1f}s vs {bw.get('wall', 0):.1f}s")
