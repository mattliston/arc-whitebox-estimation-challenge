"""(v2, 2026-10-13 17:5xZ: ids parsed by a regex, so arm names may contain '_r'; v1 mis-parsed them and returned n = 0.)
(22:1xZ: + the noise-free equivalent, raw x 1.36 = radlab's 16M-truth factor for the final layer, lead's request.)
TAG scout-card (2026-10-12): one TEAM_LOG-ready line for one arm against a reference arm over a seed range (pure python; the arithmetic of
k_table_v4.py): pooled final-layer change, mean log with se, networks lower, all-layer, layers 11-15, per network.
usage: python k_line_v1.py <run dirs comma-separated> <arm> <reference arm or base> <first seed> <last seed>"""
import json
import re
import math
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[1]
BASE = O / 'trainer' / 'run_RAD_S51_gen' / 'results'
runs, arm, ref, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
res = {}
for one in runs.split(','):
    for f in sorted((Path(one) / 'results').glob('sc_*.json')):
        r = json.loads(f.read_text())
        m = re.match(r'^[^_]+_[^_]+_r(\d+)_(.+)$', r['id'])      # v2: sc_<tag>_r<seed>_<arm>; an arm may contain '_r' (v1's split did not allow it)
        assert m, r['id']
        seed, a = m.group(1), m.group(2)
        if lo <= int(seed) <= hi:
            res.setdefault(a, {}).setdefault(int(seed), r)
ref_of = (lambda s: json.loads((BASE / f'a_RS51_base_r{s}.json').read_text())) if ref == 'base' else (lambda s: res.get(ref, {}).get(s))
pairs = [(r, ref_of(s)) for s, r in sorted(res.get(arm, {}).items()) if ref_of(s) is not None]
n = len(pairs)
if n < 2:
    print(f'{arm} vs {ref}: n = {n}'); sys.exit(0)
logs = [math.log(r['mse'] / b['mse']) for r, b in pairs]
m = sum(logs) / n; se = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1) / n)
pooled = sum(r['mse'] for r, _ in pairs) / sum(b['mse'] for _, b in pairs) - 1
alll = sum(sum(r['mse_layers']) for r, _ in pairs) / sum(sum(b['mse_layers']) for _, b in pairs) - 1
lay = [sum(r['mse_layers'][l] for r, _ in pairs) / sum(b['mse_layers'][l] for _, b in pairs) - 1 for l in range(16)]
print(f'{arm} vs {ref}, n = {n} ({lo}-{hi}): final-layer MSE pooled {100 * pooled:+.3f}%, mean log {100 * m:+.3f}% (se {100 * se:.3f}, '
      f'{abs(m / se) if se else 0:.1f} SE), lower on {sum(x < 0 for x in logs)} of {n} [noise-free x1.36: pooled {136 * pooled:+.3f}%, mean log {136 * m:+.3f}%], all-layer {100 * alll:+.3f}%; '
      f'L11-15 ' + ' / '.join(f'{100 * x:+.2f}' for x in lay[11:]) + '%; per network ' + ' '.join(f'{100 * x:+.2f}' for x in logs) + '%')
