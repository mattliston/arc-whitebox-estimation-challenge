"""TAG opus-reexam E27 summary: pooled change of each arm against the first arm of the same log (held-out ranks). usage: e27_summary.py <dir> [base arm]"""
import json, glob, math, sys
d0 = sys.argv[1]; basearm = sys.argv[2] if len(sys.argv) > 2 else 'S37'
rows = {}
for f in sorted(glob.glob(d0 + '/rank*.log')):
    for line in open(f):
        if line.startswith('{'):
            x = json.loads(line); rows.setdefault(x['arm'], {})[x['rank']] = (x['final_x1e9'], x['per_layer_x1e9'])
base = rows.get(basearm, {})
print('ranks', sorted(base))
for arm in rows:
    if arm == basearm: continue
    common = [r for r in rows[arm] if r in base]
    if not common: continue
    d = [100 * (rows[arm][r][0] / base[r][0] - 1) for r in common]; m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
    pooled = 100 * (sum(rows[arm][r][0] for r in common) / sum(base[r][0] for r in common) - 1)
    lay = [100 * (sum(rows[arm][r][1][l] for r in common) / sum(base[r][1][l] for r in common) - 1) for l in (3, 6, 9, 12, 14, 15)]
    print(f"{arm:8s} n={len(d)} pooled {pooled:+.2f}%  mean {m:+.2f}% se {sd/math.sqrt(len(d)):.2f}  better {sum(x < 0 for x in d)}/{len(d)}  layers 3/6/9/12/14/15 {[round(x, 2) for x in lay]}")
