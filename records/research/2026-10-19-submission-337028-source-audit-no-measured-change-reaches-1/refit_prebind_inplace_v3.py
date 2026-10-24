"""TAG fable-lead 2026-10-19 v3 (blend: uses x (1 + billed elements/3e7)): IN-PLACE per-file prebind refit for the 95c lineage (Astra's o18 setup repair asserts on the list).

Same algorithm as engineer/a62_modules/prebind_targeted_v2.py (swap every listed key no fit network asks for, in place, for the most
valuable used-but-unlisted key; then a STABLE reorder by fit usage), with two constraints the 95c tail needs:
  (1) the list is REPLACED between its own BEGIN/END markers (the o18 setup code reads _PB_SINGLE[:128] / [:192] / [:280] at import
      time and asserts on them; an appended redefinition would be invisible to those import-time captures), same key count, same
      line count (6 keys per line, same 4-line header);
  (2) keys the tail pins into the first 128 (_O18_SETUP_SKIP_PREFIX) stay inside the first 128.
Values are untouched: bit-identical predictions; only which tree keys setup binds first.
usage: refit_prebind_inplace_v3.py <candidate.py> <setup probe json (binding_trace, keys_bound)> <out candidate.py> <fit quick json> [...] [-- <holdout json> ...]
env BUDGET_NORMAL / BUDGET_SLOW: binding-request budgets for the simulation (default = the probe's first-128 binding requests)."""
import collections
import json
import os
import re
import statistics
import sys
import types
from pathlib import Path

args = sys.argv[1:]
candidate, probe, out = Path(args[0]), Path(args[1]), Path(args[2])
rest = args[3:]
fit_paths = rest[:rest.index('--')] if '--' in rest else rest
hold_paths = rest[rest.index('--') + 1:] if '--' in rest else []
assert not out.exists(), out

text = candidate.read_text()
lines = text.split('\n')
BEGIN = '# BEGIN OPUS-ENGINEER a62 SLOW-AWARE TARGETED PREBIND LIST'
END = '# END OPUS-ENGINEER a62 SLOW-AWARE TARGETED PREBIND LIST'
begins = [i for i, l in enumerate(lines) if l.startswith(BEGIN)]
b = begins[-1]
e = next(i for i in range(b, len(lines)) if lines[i].startswith(END))
block = lines[b:e + 1]
assert block[4] == '_PB_SINGLE = (' and block[-2] == ')', (block[4], block[-2])
old = [tuple(int(x) for x in m) for m in re.findall(r'\((\d+), (\d+), (\d+), (\d+)\)', '\n'.join(block[5:-2]))]
print('in-place block lines %d-%d (%d lines), %d keys' % (b + 1, e + 1, len(block), len(old)))

# pinned keys: the tail's first-128 assertions
pinned = set()
for m in re.finditer(r'_O18_SETUP_SKIP_PREFIX = frozenset\(\((.*?)\)\)\n', text):
    pinned |= {tuple(int(x) for x in k) for k in re.findall(r'\((\d+), (\d+), (\d+), (\d+)\)', m.group(1))}
print('pinned into the first 128:', sorted(pinned))
assert pinned.issubset(set(old[:128]))

# binding cost per key from the probe trace (prebind_targeted_v2's rule)
setup = json.load(open(probe))
trace, keys_bound = setup['binding_trace'], int(setup['keys_bound'])
inc = [y - x for x, y in zip(trace, trace[1:])][:keys_bound]
cost = {}
for k, c in zip(old[:keys_bound], inc):
    cost.setdefault(k, c)
group = collections.defaultdict(list)
for k, c in cost.items():
    group[(k[0], k[1])].append(c)
overall = statistics.median(inc)
first128 = sum(inc[:128])
BUDGET = dict(normal=int(os.environ.get('BUDGET_NORMAL', first128)), slow=int(os.environ.get('BUDGET_SLOW', first128)))
print('binding cost per key from the trace: %d keys, median %.0f, mean %.1f requests; first-128 total %d; budgets %s'
      % (len(inc), overall, statistics.mean(inc), first128, BUDGET))


def kcost(k):
    if k in cost:
        return cost[k]
    g = group.get((k[0], k[1]))
    return statistics.median(g) if g else overall


def nets(paths):
    return [[(row['rank'], {tuple(k[:4]): k[4] for k in row['key_uses']}) for row in json.load(open(p))['rows']] for p in paths]


fit_sessions, hold_sessions = nets(fit_paths), nets(hold_paths)
fit = [keys for s in fit_sessions for _r, keys in s]
count = collections.Counter()
for keys in fit:
    count.update(k for k in keys if k[0] != 3)
share = {k: (count[k] - 0.7) / len(fit) for k in count if count[k] >= 2}
value = {k: kcost(k) * (1 - (1 - share[k]) ** 12) for k in share}
old_set = set(old)
dead = [i for i, k in enumerate(old) if count[k] == 0 and k not in pinned]
wanted = sorted((k for k in value if k not in old_set), key=lambda k: (-value[k], k))
swapped = list(old)
swaps = []
for i in dead:
    if not wanted:
        break
    k = wanted.pop(0)
    swaps.append((i, swapped[i], k))
    swapped[i] = k
def _elements(k):   # v2: a lazy bind is billed at the tree's element count 7^L * (rows/2^L) * (cols/2^L)
    return (7 ** k[1]) * (k[2] // (1 << k[1])) * (k[3] // (1 << k[1]))
def _billed(k):   # v3: box views v2 serve depth-6 A/B trees as free slices; only C-side and other depths bill a reshape
    return 0 if (k[1] == 6 and k[0] in (0, 1) and not (k[0] == 1 and k[2] in (256, 384))) else _elements(k)
order = sorted(range(len(swapped)), key=lambda i: (-(count[swapped[i]] * (1.0 + _billed(swapped[i]) / 3e7)), i))   # v3: uses x (1 + billed elements / 3e7): request coverage kept, bill-heavy keys first
new = [swapped[i] for i in order]
# constraint (2): pinned keys back inside the first 128 (swap with the last unpinned key of the prefix, lowest usage first)
for k in sorted(pinned):
    if k in new[:128]:
        continue
    j = new.index(k)
    i = max(i for i in range(128) if new[i] not in pinned)
    new[i], new[j] = new[j], new[i]
    print('pinned %s moved %d -> %d (displaced %s, used by %d fit nets)' % (k, j, i, new[j], count[new[j]]))
assert pinned.issubset(set(new[:128])) and len(new) == len(old) and set(new) == set(swapped)
dead128_old = sum(1 for k in old[:128] if count[k] == 0)
dead128_new = sum(1 for k in new[:128] if count[k] == 0)
used_unlisted = sum(1 for k in count if k not in old_set and count[k] >= 1)
print('old list %d keys; dead (no fit network asks): %d (%d inside the first 128); used keys unlisted: %d (>=2 nets: %d); swaps: %d; '
      'fit networks %d; dead inside the first 128 after: %d' % (len(old), len(dead), dead128_old, used_unlisted,
                                                                  sum(1 for k in value if k not in old_set), len(swaps), len(fit), dead128_new))


def bound_prefix(listed, budget):
    used, n = 0, 0
    for k in listed:
        c = kcost(k)
        if used + c > budget:
            break
        used += c
        n += 1
    return min(n, 128)   # the 95c setup binds at most the first 128 keys


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


for label, listed in (('OLD', old), ('SWAPS ONLY', swapped), ('NEW (swaps + reorder)', new)):
    for pace in ('normal', 'slow'):
        for name, sess in (('fit', fit_sessions), ('hold', hold_sessions)):
            if not sess:
                continue
            n, f, o = sessions(sess, listed, BUDGET[pace])
            print('%-22s %-6s %-4s keys bound %3d | lazy-bind requests: first network mean %6.0f (%s) | later networks mean %5.1f'
                  % (label, pace, name, n, statistics.mean(f), ' '.join('%d' % x for x in f), statistics.mean(o) if o else 0.0))

header = [BEGIN + ' (a62_modules/prebind_targeted_v2.py algorithm, in place; bit-identical: only which tree keys',
          '# setup binds and in which order): the candidate\'s own list with %d keys that no fit network asks for swapped for keys they ask for,' % len(swaps),
          '# then ordered by how many fit networks use each key (most first, ties in list order; refit_prebind_inplace_v3 2026-10-19,',
          '# pinned first-128 keys kept; fit networks %s).' % ', '.join(str(r) for s in fit_sessions for r, _k in s),
          '_PB_SINGLE = (']
body = ['    ' + ' '.join('(%d, %d, %d, %d),' % k for k in new[i:i + 6]) for i in range(0, len(new), 6)]
newblock = header + body + [')', END]
assert len(newblock) == len(block), (len(newblock), len(block))
lines[b:e + 1] = newblock
result = '\n'.join(lines)
compile(result, str(out), 'exec')
out.write_text(result)
assert len(result.split('\n')) == len(text.split('\n'))
print('written', out, 'lines unchanged', len(lines))
