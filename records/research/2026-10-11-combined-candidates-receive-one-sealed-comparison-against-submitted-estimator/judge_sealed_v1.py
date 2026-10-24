"""TAG opus_engineer a62 sealed judge (lead's pre-registered fire rule 17:20:17Z): S48 vs S47e on the SEALED A37 7100328-359, paired by
network. Score change per network = (MSE_S48 / MSE_S47e) x (bill_S48 / bill_S47e) - 1 (final-layer MSE against the analyst's fresh
truths; bill = the network's flops; the 0.1 floor of the bill share is not reached: about 0.115). Pooled = (sum MSE_S48 / sum MSE_S47e)
x (mean bill ratio) - 1; se = the standard error of the per-network log score ratios. Also prints MSE and bill separately, the number
of networks better, and per-layer pooled MSE changes. Reads the logs while the arms run (partial verdicts are marked).
usage: judge_sealed_v1.py <S48 arm name> [<S47e arm name>]   (arm name = gate_rdz_<name>.log in this folder)"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
arm, ref = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'sealed_s47e2')
load = lambda name: {r['seed']: r for r in (json.loads(l) for l in open(HERE/f'gate_rdz_{name}.log') if l.startswith('{"seed"'))}
a, b = load(arm), load(ref)
seeds = sorted(s for s in a if s in b and 7100328 <= s <= 7100359)
n = len(seeds)
done = n == 32 and (HERE/f'gate_rdz_{arm}.json').exists() and (HERE/f'gate_rdz_{ref}.json').exists()
mse = sum(a[s]['mse_final'] for s in seeds) / sum(b[s]['mse_final'] for s in seeds) - 1
bill = sum(a[s]['flops'] / b[s]['flops'] for s in seeds) / n - 1
logs = [math.log(a[s]['mse_final'] / b[s]['mse_final'] * a[s]['flops'] / b[s]['flops']) for s in seeds]
m = sum(logs) / n
se = (sum((x - m) ** 2 for x in logs) / (n - 1) / n) ** 0.5 if n > 1 else float('nan')
score = (1 + mse) * (1 + bill) - 1
print('%s %s vs %s on %d sealed networks: SCORE %+.3f%% pooled (mean log %+.3f%%, se %.3f, better on %d of %d) | MSE %+.3f%% | bill %+.3f%% | ops %+.0f'
      % ('FINAL' if done else 'PARTIAL', arm, ref, n, 100 * score, 100 * m, 100 * se, sum(x < 0 for x in logs), n, 100 * mse, 100 * bill,
         sum(a[s]['ops'] - b[s]['ops'] for s in seeds) / max(n, 1)))
print('per layer MSE', ' '.join('L%d %+.2f' % (l, 100 * (sum(a[s]['mse_layers'][l] for s in seeds) / sum(b[s]['mse_layers'][l] for s in seeds) - 1))
                               for l in (4, 7, 10, 13, 14, 15)))
if done:
    print('VERDICT under the pre-registered rule (fire if the pooled score change is negative and the release gate passes):',
          'NEGATIVE -> fire if the release gate passes' if score < 0 else 'NOT NEGATIVE -> do not fire')
