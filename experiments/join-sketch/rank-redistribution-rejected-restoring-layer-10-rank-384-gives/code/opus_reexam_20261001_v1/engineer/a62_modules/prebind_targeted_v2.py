"""TAG opus_engineer a62 v2 (2026-10-12 00:5xZ, lead: the S48 TWIN failed only 'requests under a setup 1.3x slower' (+2.20%, limit +2%) because
prebind list v2 was fit to S48b's keys): a SLOW-AWARE targeted list = v1's swaps (every listed key no fit network asks for, in place, for
the most valuable key the fit networks ask for that is not listed) followed by a STABLE REORDER by usage (keys used by more fit networks
first; ties keep the list order), so the keys a slow worker's setup binds before its wall guard are the ones the networks use. Values are
the same (bit-identical predictions: only which tree keys setup binds, and in which order).
The setup's binding phase is cut by its wall guard after a REQUEST budget (the emulated pace): the twin's own stage C traces give the
budgets (normal pace 17,812 - 573 = 17,239 binding requests, 1.3x slow 13,430 - 573 = 12,857) and each listed key's binding cost (its
increment in the normal trace; keys never bound there: the median of their (kind, L) group). The script simulates worker sessions on
the fit (and hold-out) censuses: setup binds the list in order until the budget, then each network binds lazily the used keys still
unbound, at their binding cost in REQUESTS. Prints old vs new, normal vs slow.
usage: prebind_targeted_v2.py <candidate.py> <normal stage C manifest.json> <out fragment.py> <fit quick json> [...] [-- <holdout json> ...]"""
import collections
import json
import os
import statistics
import sys
import types
from pathlib import Path

args = sys.argv[1:]
candidate, manifest, out = Path(args[0]), Path(args[1]), Path(args[2])
rest = args[3:]
fit_paths = rest[:rest.index('--')] if '--' in rest else rest
hold_paths = rest[rest.index('--') + 1:] if '--' in rest else []
assert not out.exists()
BUDGET = dict(normal=int(os.environ.get('BUDGET_NORMAL', 17812 - 573)), slow=int(os.environ.get('BUDGET_SLOW', 13430 - 573)))


def nets(paths):
    return [[(row['rank'], {tuple(k[:4]): k[4] for k in row['key_uses']}) for row in json.load(open(p))['rows']] for p in paths]


stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_targeted_v2'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
old = [tuple(k) for k in mod._PB_SINGLE]

setup = json.load(open(manifest))['setup']
trace, keys_bound = setup['binding_trace'], int(setup['keys_bound'])
inc = [b - a for a, b in zip(trace, trace[1:])][:keys_bound]   # the trace ends with phase markers (0 increments): only the bound keys
cost = {}
for k, c in zip(old[:keys_bound], inc):
    cost.setdefault(k, c)
group = collections.defaultdict(list)
for k, c in cost.items():
    group[(k[0], k[1])].append(c)
overall = statistics.median(inc)
print('binding cost per key from the trace: %d keys, median %.0f, mean %.1f requests' % (len(inc), overall, statistics.mean(inc)))


def kcost(k):
    if k in cost:
        return cost[k]
    g = group.get((k[0], k[1]))
    return statistics.median(g) if g else overall


fit_sessions, hold_sessions = nets(fit_paths), nets(hold_paths)
fit = [keys for s in fit_sessions for _r, keys in s]
count = collections.Counter()
for keys in fit:
    count.update(k for k in keys if k[0] != 3)
share = {k: (count[k] - 0.7) / len(fit) for k in count if count[k] >= 2}
value = {k: kcost(k) * (1 - (1 - share[k]) ** 12) for k in share}     # requests saved over a 12-network-ish worker life
old_set = set(old)
dead = [i for i, k in enumerate(old) if count[k] == 0]
wanted = sorted((k for k in value if k not in old_set), key=lambda k: (-value[k], k))
swapped = list(old)
swaps = []
for i in dead:
    if not wanted:
        break
    k = wanted.pop(0)
    swaps.append((i, swapped[i], k))
    swapped[i] = k
order = sorted(range(len(swapped)), key=lambda i: (-count[swapped[i]], i))   # stable: most-used first, ties in list order
new = [swapped[i] for i in order]


def bound_prefix(listed, budget):
    used, n = 0, 0
    for k in listed:
        c = kcost(k)
        if used + c > budget:
            break
        used += c
        n += 1
    return n


def sessions(sess, listed, budget):
    n = bound_prefix(listed, budget)
    firsts, others = [], []
    for s in sess:
        bound = set(listed[:n])
        for j, (_rank, keys) in enumerate(s):
            lazy = [k for k in keys if k[0] != 3 and k not in bound]
            req = sum(kcost(k) for k in lazy)
            bound.update(lazy)
            (firsts if j == 0 else others).append(req)
    return n, firsts, others


print('old list %d keys; dead for this candidate (no fit network asks): %d; wanted (>= 2 fit networks, not listed): %d; swaps: %d; '
      'fit networks %d' % (len(old), len(dead), sum(1 for k in value if k not in old_set), len(swaps), len(fit)))
for label, listed in (('OLD', old), ('SWAPS ONLY', swapped), ('NEW (swaps + reorder)', new)):
    for pace in ('normal', 'slow'):
        for name, sess in (('fit', fit_sessions), ('hold', hold_sessions)):
            if not sess:
                continue
            n, f, o = sessions(sess, listed, BUDGET[pace])
            print('%-22s %-6s %-4s keys bound %3d | lazy-bind requests: first network mean %6.0f (%s) | later networks mean %5.1f'
                  % (label, pace, name, n, statistics.mean(f), ' '.join('%d' % x for x in f), statistics.mean(o) if o else 0.0))
text = ('\n# BEGIN OPUS-ENGINEER a62 SLOW-AWARE TARGETED PREBIND LIST (a62_modules/prebind_targeted_v2.py; bit-identical: only which tree keys\n'
        '# setup binds and in which order): the candidate\'s own list with %d keys that no fit network asks for swapped for keys they ask for,\n'
        '# then ordered by how many fit networks use each key (most first, ties in list order), so a slow worker\'s setup binds the used keys\n'
        '# first (fit networks %s).\n_PB_SINGLE = (\n' % (len(swaps), ', '.join(str(r) for s in fit_sessions for r, _k in s)))
for i in range(0, len(new), 6):
    text += '    ' + ' '.join('(%d, %d, %d, %d),' % k for k in new[i:i + 6]) + '\n'
text += ')\n# END OPUS-ENGINEER a62 SLOW-AWARE TARGETED PREBIND LIST\n'
compile(text, str(out), 'exec')
out.write_text(text)
print('written', out)
