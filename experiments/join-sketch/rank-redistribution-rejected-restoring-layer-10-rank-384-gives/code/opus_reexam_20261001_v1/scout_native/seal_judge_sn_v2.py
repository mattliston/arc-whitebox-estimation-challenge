"""TAG scout-native seal_judge_sn_v2 (2026-10-13, lead 07:40Z: s53k1 fires only if its seal mean <= -0.10%, no extension) = v1 + an
optional pre-registered threshold: with 'fire_at_or_below_pct' in the PREREG, FIRE iff 100 x mean <= that value, else NO FIRE (no
extension); without it, v1's rule. v1's text follows.
TAG scout-native seal_judge_sn_v1 (2026-10-13): judge a pre-registered native seal from run_cert_v2 outputs (PB_ALL=1). Per network:
score = log(final MSE stack / base) + log(flops stack / flops base). Prints each network, then one line
'MEAN <x%> SE <se%> LOWER <k> OF <n> -> FIRE | EXTEND | NO FIRE' by the pre-registration's rule (8 networks: FIRE iff mean < 0; 0 <= mean < +1 se
-> EXTEND; else NO FIRE; with the extension (16): FIRE iff mean < 0). Only the pre-registered seeds count.
usage: seal_judge_sn_v1.py <PREREG.json> <base json[,json]> <stack json[,json]>"""
import json
import math
import sys

pre = json.load(open(sys.argv[1]))
rows = lambda ps: {r['seed']: r for p in ps.split(',') for r in json.load(open(p))['rows']}
base, stack = rows(sys.argv[2]), rows(sys.argv[3])
seeds = [s for s in pre['seeds'] + pre.get('extension_seeds', []) if s in base and s in stack]
d = []
for s in seeds:
    b, a = stack[s], base[s]
    lm, lb = math.log(b['mse_final'] / a['mse_final']), math.log(b['flops'] / a['flops'])
    d.append(lm + lb)
    print('%d  MSE %+.4f%%  bill %+.4f%%  score %+.4f%%  requests %+d  pred %s' % (s, 100 * (math.exp(lm) - 1), 100 * (math.exp(lb) - 1), 100 * (lm + lb), b['ops'] - a['ops'],
                                                                                    'same' if a['pred_sha'] == b['pred_sha'] else 'changed'))
n = len(d)
m = sum(d) / n
se = (sum((x - m) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else float('nan')
extended = n > len(pre['seeds'])
if pre.get('fire_at_or_below_pct') is not None:
    verdict = 'FIRE' if 100 * m <= pre['fire_at_or_below_pct'] else 'NO FIRE'
elif m < 0:
    verdict = 'FIRE'
elif not extended and n == len(pre['seeds']) and m < se:
    verdict = 'EXTEND'
else:
    verdict = 'NO FIRE'
print('MEAN %+.4f%% SE %.4f%% LOWER %d OF %d -> %s' % (100 * m, 100 * se, sum(x < 0 for x in d), n, verdict))
