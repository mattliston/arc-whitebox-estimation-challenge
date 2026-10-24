"""TAG opus_engineer a62 (2026-10-11 23:2xZ): the one-sided read row cut test (chain_rdz3_v2.sh) against S48b on 7100136-143:
pooled final-layer MSE change per arm (se of the mean log ratio) and each tail's bill share. The share prices a REAL restricted read:
the per-layer bill of the products whose size is the read's row count (young square read, old-tier inner read and the old columns'
reconstruct on the read rows: census of the fired S47e on 651 / 378, module_census_v1 keys classified by table_v1.py) times the tail's
fraction of the response rows at that layer (the arms' own counters rt / rp / rm, summed over the 8 networks).
Pre-declared (lead): (A) = the a > +2 tail proceeds to a build only if its MSE change < its bill share - 0.5%.
usage: judge_rdz3_v1.py"""
import collections
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
se = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1) / len(v))

# per-layer bill of the row-sized read products, % of the mean bill (S47e census, 651 / 378)
src = (HERE/'table_v1.py').read_text().split("\ntot = {n['rank']")[0]
argv, sys.argv = sys.argv, ['x', str(HERE/'census_s47e_651_378.json')]
ns = {'__file__': str(HERE/'table_v1.py')}
exec(compile(src, 'table_v1.py', 'exec'), ns)
sys.argv = argv
d = json.load(open(HERE/'census_s47e_651_378.json'))
mean_bill = sum(n['flops'] for n in d['nets']) / len(d['nets'])
rowsized = collections.Counter()
for rank, layer, line, chain, op, fl, n in d['rows']:
    names = [c[0] for c in chain]
    for m, frac in ns['classify'](layer, line, chain):
        if m in ('BASE bank read: young square', 'BASE bank read: old tier') or (m == 'BASE bank push: old tier' and '_slr_reconstruct' in names):
            rowsized[layer] += frac * fl / len(d['nets'])
share_l = {l: 100 * v / mean_bill for l, v in rowsized.items()}

for arm, tail in (('rdz3_plus', 'rp'), ('rdz3_minus', 'rm')):
    rows = {x['seed']: x for x in json.load(open(HERE/f'gate_rdz_{arm}.json'))['rows']}
    seeds = sorted(set(rows) & set(ref))
    lm = [math.log(rows[s]['mse_final'] / ref[s]['mse_final']) for s in seeds]
    pm = math.log(sum(rows[s]['mse_final'] for s in seeds) / sum(ref[s]['mse_final'] for s in seeds))
    layers = [math.log(sum(rows[s]['mse_layers'][l] for s in seeds) / sum(ref[s]['mse_layers'][l] for s in seeds)) for l in range(16)]
    cnt = collections.Counter()
    for s in seeds:
        for k, v in rows[s]['rdz'].items():
            cnt[k] += v
    frac = {l: cnt.get(f'{tail}{l}', 0) / cnt[f'rt{l}'] for l in range(1, 16) if cnt.get(f'rt{l}')}
    share = sum(frac.get(l, 0) * share_l.get(l, 0) for l in range(1, 16))
    print('%s (%s tail): final-layer MSE %+.3f%% pooled (mean log %+.3f%%, se %.3f, lower on %d of %d); bill share of a real restricted read %.2f%%'
          % (arm, '+' if tail == 'rp' else '-', 100 * pm, 100 * mean(lm), 100 * se(lm), sum(x < 0 for x in lm), len(seeds), share))
    print('   tail fraction of the response rows by layer:', ' '.join('L%d %.3f' % (l, frac[l]) for l in sorted(frac)))
    print('   per layer MSE:', ' '.join('L%d %+.2f' % (l, 100 * layers[l]) for l in (2, 4, 6, 7, 8, 10, 12, 13, 14, 15)))
    print('   per network MSE:', ' '.join('%+.2f' % (100 * x) for x in lm))
    if tail == 'rp':
        print('   RULE (A): MSE change %+.3f%% %s bill share - 0.5%% = %.3f%% -> %s' % (
            100 * pm, '<' if 100 * pm < share - 0.5 else '>=', share - 0.5, 'BUILD' if 100 * pm < share - 0.5 else 'NO BUILD'))
print('row-sized read bill by layer (%% of bill):', ' '.join('L%d %.2f' % (l, share_l[l]) for l in sorted(share_l)))
