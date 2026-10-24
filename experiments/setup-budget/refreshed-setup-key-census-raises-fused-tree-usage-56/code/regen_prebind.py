"""TAG opus_engineer a13: a prebind key list for a candidate, from tree-key usage censuses (quick.py --keys).
usage: regen_prebind.py <candidate.py> <out fragment.py> <length> <fit quick json> [<fit quick json> ...] [-- <holdout quick json> ...]
The candidate's own list (_PB_SINGLE, the keys Estimator.setup binds whole and fuses, most valuable first, cut by the wall guard) was
made from censuses of the September kernels. A kernel change moves shapes between depths, so some listed keys are never asked for
again and new recurring keys are bound lazily inside predict (reshape billed at the tree's element count, and no fused level ops).
New list: the 128 keys with the most uses over the fit networks first (setup fuses the level ops of the first 128 listed keys
as it binds them, and a fused key saves requests at every use), then every other key by expected saved bind flops in a worker
session of 12 networks = tree elements x (1 - (1 - f)^12), f = the share of fit networks that ask for the key, shrunk for rare
keys (count - 0.7; a key seen in one fit network only counts 0.3 and only if it was on the candidate's own list). Cut to <length>.
Prints, for the fit sessions and the holdout sessions, the lazy-bind flops a worker session would pay with the old and the new
list (all listed keys bound, and only the first 480: the platform's setup guard cuts the tail).
The fragment only re-assigns _PB_SINGLE: predictions and every arithmetic op are unchanged."""
import collections
import json
import sys
import types
from pathlib import Path

args = sys.argv[1:]
candidate, out, length = Path(args[0]), Path(args[1]), int(args[2])
rest = args[3:]
fit_paths = rest[:rest.index('--')] if '--' in rest else rest
hold_paths = rest[rest.index('--') + 1:] if '--' in rest else []
assert not out.exists()


def nets(paths):
    # one list of (rank, {key: uses}, bill, measured lazy-bind flops) per file = per worker session, in session order
    res = []
    for path in paths:
        res.append([(row['rank'], {tuple(k[:4]): k[4] for k in row['key_uses']}, row['flops'], row.get('binds', {}).get('flops'))
                    for row in json.load(open(path))['rows']])
    return res


def numel(key):
    kind, L, rows, cols = key
    return 7 ** L * (rows >> L) * (cols >> L)


stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_regen'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
old = list(mod._PB_SINGLE)
fit_sessions, hold_sessions = nets(fit_paths), nets(hold_paths)
fit = [(rank, keys) for session in fit_sessions for rank, keys, _b, _m in session]
hold = [(rank, keys) for session in hold_sessions for rank, keys, _b, _m in session]
count = collections.Counter()
acc = collections.Counter()      # (M, N) of the k-split's second leaf buffer: recorded as kind 3 at its first use in a process only,
for _rank, keys in fit:          # so every read-shaped output key (rows, cols <= 1024) counts as a possible buffer too
    count.update(k for k in keys if k[0] != 3)
    acc.update({(k[2], k[3]) for k in keys if (k[0] == 3) or (k[0] == 2 and k[1] == 6 and k[2] <= 1024 and k[3] <= 1024
                                                               and 2 * (k[2] // 64) * (k[3] // 64) <= 1024)})
fit = [(rank, {k: v for k, v in keys.items() if k[0] != 3}) for rank, keys in fit]
hold = [(rank, {k: v for k, v in keys.items() if k[0] != 3}) for rank, keys in hold]
uses = collections.Counter()
for _rank, keys in fit:
    for k, v in keys.items():
        uses[k] += v
old_set = set(old)
share = {k: (count[k] - 0.7 if count[k] >= 2 else (0.3 if k in old_set else 0.0)) / len(fit) for k in count}
value = {k: numel(k) * (1 - (1 - share[k]) ** 12) for k in count}
head = sorted((k for k in count if count[k] >= 2), key=lambda k: (-uses[k], -numel(k), k))[:128]
rest = sorted((k for k in count if value[k] > 0 and k not in set(head)), key=lambda k: (-value[k], k))
new = (head + rest)[:length]


def session_binds(name, sessions, listed, bound_count=None):
    # lazy tree-view binds of a worker session if setup binds the first bound_count listed keys and nothing is evicted:
    # every key a network asks for that is neither bound in setup nor bound by an earlier network bills its tree elements once
    for session in sessions:
        bound = set(listed if bound_count is None else listed[:bound_count])
        total = bill = measured = 0
        per = []
        for rank, keys, flops, meas in session:
            new = [k for k in keys if k[0] != 3 and k not in bound]
            cost = sum(numel(k) for k in new)
            bound.update(new)
            total += cost; bill += flops; measured += meas or 0
            per.append(f'{100*cost/flops:.2f}')
        print(f'  {name} session {session[0][0]}..{session[-1][0]} ({len(session)} networks): simulated binds {total/1e9:6.2f} G = {100*total/bill:.3f}% of the bill'
              + (f' (measured with the candidate\'s own list {100*measured/bill:.3f}%)' if measured else '') + '; per network % ' + ' '.join(per))


print(f'old list {len(old)} keys ({sum(numel(k) for k in old)/1e9:.1f} G elements); used by a fit network: {sum(1 for k in old if k in count)}; '
      f'new list {len(new)} keys ({sum(numel(k) for k in new)/1e9:.1f} G elements), of which {sum(1 for k in new if k not in set(old))} are new')
def head_uses(name, sessions, listed):
    # share of a session's tree uses that fall on the first 128 listed keys (fused at bind time) and on the first 480 (bound)
    head, body = set(listed[:128]), set(listed[:480])
    for session in sessions:
        total = on_head = on_body = 0
        for _rank, keys, _flops, _meas in session:
            for k, v in keys.items():
                if k[0] != 3:
                    total += v; on_head += v * (k in head); on_body += v * (k in body)
        print(f'  {name} session {session[0][0]}..{session[-1][0]}: tree uses per network {total/len(session):.0f}; on the first 128 listed keys {100*on_head/total:.1f}%, on the first 480 {100*on_body/total:.1f}%')


for name, listed in (('OLD list, all bound', old), ('NEW list, all bound', new), ('OLD list, first 480 bound', old[:480]), ('NEW list, first 480 bound', new[:480])):
    print(name)
    session_binds('fit ', fit_sessions, listed)
    session_binds('hold', hold_sessions, listed)
for name, listed in (('OLD list', old), ('NEW list', new)):
    print(name)
    head_uses('fit ', fit_sessions, listed)
    head_uses('hold', hold_sessions, listed)
text = ('# BEGIN OPUS-ENGINEER PREBIND LIST (a13, bit-identical: only which tree keys setup binds and fuses, in which order): regenerated by\n'
        '# a13_prebind/regen_prebind.py from tree-key censuses of this candidate over ' + str(len(fit)) + ' networks (' + ', '.join(str(r) for r, _k in fit) + ').\n'
        '_PB_SINGLE = (\n')
for i in range(0, len(new), 6):
    text += '    ' + ' '.join('(%d, %d, %d, %d),' % k for k in new[i:i + 6]) + '\n'
text += ')\n'
old_acc = list(mod._KS_ACC_MN)
new_acc = sorted(acc, key=lambda mn: (-acc[mn], -mn[0] * mn[1], mn))[:max(len(old_acc), 40)]
print(f'leaf buffers: old list {len(old_acc)}, new list {len(new_acc)}, of which {sum(1 for x in new_acc if x not in set(old_acc))} new')
text += '_KS_ACC_MN = (' + ', '.join('(%d, %d)' % mn for mn in new_acc) + ')\n'
text += '_PB_EAGER = 128                            # listed keys whose level ops are fused as they are bound (the builder points _ipt_side at this)\n# END OPUS-ENGINEER PREBIND LIST\n'
compile(text, str(out), 'exec')
out.write_text(text)
print('written', out)
