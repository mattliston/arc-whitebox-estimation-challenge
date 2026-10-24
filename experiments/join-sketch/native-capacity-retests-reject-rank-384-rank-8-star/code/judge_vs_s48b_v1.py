"""TAG opus_engineer a62 (2026-10-12 00:2xZ): native arms against S48b on 7100136-143 (S48b's runs: gate_rdz_ee_ref_s48b.json for 136-137,
gate_rdz_rz_ref6.json for 138-143): pooled final-layer MSE change, pooled bill change, score = both (logs), the se of the per-network mean log
score, operations, per-layer MSE; the lead's pre-declared rule: an arm goes to a 32-network fresh judge only if its 8-network score < -0.5%.
usage: judge_vs_s48b_v1.py <arm tag> [<arm tag> ...]   (reads gate_rdz_<tag>.json)"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ref = {}
for f in ('gate_rdz_ee_ref_s48b.json', 'gate_rdz_rz_ref6.json'):
    for x in json.load(open(HERE/f))['rows']:
        ref[x['seed']] = x
mean = lambda v: sum(v) / len(v)
se = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else float('nan')
for tag in sys.argv[1:]:
    f = HERE/f'gate_rdz_{tag}.json'
    if not f.exists():
        print(tag, 'no result yet')
        continue
    var = {x['seed']: x for x in json.load(open(f))['rows']}
    seeds = sorted(set(var) & set(ref))
    lm = [math.log(var[s]['mse_final'] / ref[s]['mse_final']) for s in seeds]
    lb = [math.log(var[s]['flops'] / ref[s]['flops']) for s in seeds]
    ls = [m + b for m, b in zip(lm, lb)]
    pm = math.log(sum(var[s]['mse_final'] for s in seeds) / sum(ref[s]['mse_final'] for s in seeds))
    pb = math.log(sum(var[s]['flops'] for s in seeds) / sum(ref[s]['flops'] for s in seeds))
    layers = [math.log(sum(var[s]['mse_layers'][l] for s in seeds) / sum(ref[s]['mse_layers'][l] for s in seeds)) for l in range(16)]
    score = 100 * (pm + pb)
    print('%s: n %d | final-layer MSE %+.3f%% pooled (mean log %+.3f%%, se %.3f, lower on %d) | bill %+.3f%% | SCORE %+.3f%% pooled (mean log %+.3f%%, se %.3f) | ops %+.0f'
          % (tag, len(seeds), 100 * pm, 100 * mean(lm), 100 * se(lm), sum(x < 0 for x in lm), 100 * pb, score, 100 * mean(ls), 100 * se(ls),
             mean([var[s]['ops'] - ref[s]['ops'] for s in seeds])))
    print('   per layer MSE:', ' '.join('L%d %+.2f' % (l, 100 * layers[l]) for l in (4, 7, 8, 10, 11, 12, 13, 14, 15)))
    print('   per network score:', ' '.join('%+.2f' % (100 * x) for x in ls))
    print('   RULE: score %+.3f%% %s -0.5%% -> %s' % (score, '<' if score < -0.5 else '>=', '32-NETWORK FRESH JUDGE' if score < -0.5 else 'stays closed'))
