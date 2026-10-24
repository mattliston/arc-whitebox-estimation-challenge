"""TAG opus_engineer a62 (file written 2026-10-12T17:40Z; lead 17:3xZ decision 3, radlab's rule): the stack SEAL. Commands:
  alloc <tag> <n>: the next n networks of the analyst's block (seal_block_v1.json) that no earlier seal took (seal_alloc_v1.jsonl, append-only);
     prints them as a comma list and records them under <tag>;
  prereg <tag> <base file> <stack file> <seeds> <bill ratio>: writes seal_<tag>_PREREG.json (refuses to overwrite) BEFORE any seal run;
  judge <tag> <base gate json[,json]> <stack gate json[,json]>: score change per network = log(final MSE stack / base) + log(bill ratio), the bill ratio
     from the final twins' count sessions on 35 / 100 (emulated setup clock, so load cannot move it; gate_rdz flops are load-dependent through
     the prebind wall guard and are not used). Prints 'MEAN <x%> SE <se%> LOWER <k> OF <n>' and the pre-registered decision:
     FIRE (mean < 0), EXTEND (0 <= mean < +1 se, first 8 only), NO FIRE (otherwise; after the extension: FIRE iff mean < 0)."""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
cmd, tag = sys.argv[1], sys.argv[2]
if cmd == 'alloc':
    block = json.load(open(HERE/'seal_block_v1.json'))['seeds']
    alloc = HERE/'seal_alloc_v1.jsonl'
    used = {s for l in (alloc.read_text().splitlines() if alloc.exists() else []) for s in json.loads(l)['seeds']}
    take = [s for s in block if s not in used][:int(sys.argv[3])]
    assert len(take) == int(sys.argv[3]), 'the seal block is used up: ask the analyst for a new one'
    with open(alloc, 'a') as f:
        f.write(json.dumps(dict(tag=tag, seeds=take, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')
    print(','.join(map(str, take)))
elif cmd == 'prereg':
    out = HERE/('seal_%s_PREREG.json' % tag)
    assert not out.exists(), out
    out.write_text(json.dumps(dict(tag=tag, base=sys.argv[3], stack=sys.argv[4], seeds=[int(s) for s in sys.argv[5].split(',')], bill_ratio=float(sys.argv[6]),
                                   score='per network log(final MSE stack / base) + log(bill ratio of the final twins, counts 35 / 100)',
                                   rule='8 networks: FIRE iff mean < 0; 0 <= mean < +1 se -> 8 more, then FIRE iff the 16-network mean < 0; else NO FIRE',
                                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1) + '\n')
    print('pre-registered', out)
elif cmd == 'judge':
    pre = json.load(open(HERE/('seal_%s_PREREG.json' % tag)))
    rows = lambda ps: {r['seed']: r for p in ps.split(',') for r in json.load(open(p))['rows']}   # comma lists: the first 8 + the extension
    base, stack = rows(sys.argv[3]), rows(sys.argv[4])
    d = [math.log(stack[s]['mse_final'] / base[s]['mse_final']) + math.log(pre['bill_ratio']) for s in sorted(stack) if s in base]
    n = len(d); m = sum(d) / n
    se = (sum((x - m) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else float('nan')
    decision = 'FIRE' if m < 0 else ('EXTEND' if n <= 8 and m < se else 'NO FIRE')
    print('MEAN %+.3f%% SE %.3f%% LOWER %d OF %d -> %s' % (100 * m, 100 * se, sum(x < 0 for x in d), n, decision))
