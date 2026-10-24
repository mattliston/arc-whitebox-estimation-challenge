"""TAG scout-card K table v4 (2026-10-12): = k_table_v3 with a choice of REFERENCE: 'base' (the trainer's a_RS51_base eval of the seed, S50 point)
or any arm name found in the run dirs (e.g. 'gk': arms measured on top of gk against gk's own result on the same seed). Pooled change per layer,
final mean log with its standard error, networks lower, all-layer change.
usage: python k_table_v4.py <run dir>[,<run dir>...] [reference arm, default base] [first seed] [last seed]"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

O = Path(__file__).resolve().parents[1]
BASE = O / 'trainer' / 'run_RAD_S51_gen' / 'results'


def main(runs, ref='base', lo=0, hi=10 ** 9):
    res = defaultdict(dict)
    for one in runs.split(','):
        for f in sorted((Path(one) / 'results').glob('sc_*.json')):
            r = json.loads(f.read_text())
            seed, arm = r['id'].split('_r')[1].split('_', 1)
            if lo <= int(seed) <= hi:
                res[arm].setdefault(int(seed), r)
        for f in sorted((Path(one) / 'results').glob('sc_*.error')):
            print('ERROR', f.name, f.read_text().strip().splitlines()[-1][:200])
    def reference(seed):
        if ref == 'base':
            return json.loads((BASE / f'a_RS51_base_r{seed}.json').read_text())
        return res[ref].get(seed)
    for arm, d in sorted(res.items()):
        if arm == ref:
            continue
        pairs = [(r, reference(s)) for s, r in sorted(d.items()) if reference(s) is not None]
        if not pairs:
            continue
        n = len(pairs)
        logs = [math.log(r['mse'] / b['mse']) for r, b in pairs]
        mean = sum(logs) / n
        se = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1) / n) if n > 1 else float('nan')
        pooled = sum(r['mse'] for r, _ in pairs) / sum(b['mse'] for _, b in pairs) - 1
        alll = sum(sum(r['mse_layers']) for r, _ in pairs) / sum(sum(b['mse_layers']) for _, b in pairs) - 1
        layer = [sum(r['mse_layers'][l] for r, _ in pairs) / sum(b['mse_layers'][l] for _, b in pairs) - 1 for l in range(16)]
        print(f'{arm:10s} vs {ref:6s} n={n:3d}  final pooled {100 * pooled:+.3f}%  mean log {100 * mean:+.3f}% (se {100 * se:.3f})  '
              f'lower {sum(x < 0 for x in logs)}/{n}  all-layer {100 * alll:+.3f}%')
        print('         per layer 1-15 (%): ' + ' '.join(f'{100 * x:+.2f}' for x in layer[1:]))
        print('         per network (%): ' + ' '.join(f'{100 * x:+.2f}' for x in logs))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else 'base', *(int(x) for x in a[2:4]))
