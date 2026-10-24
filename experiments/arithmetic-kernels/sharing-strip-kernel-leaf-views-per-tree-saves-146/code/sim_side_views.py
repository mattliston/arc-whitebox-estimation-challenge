"""TAG opus_engineer hzs: what the strip kernels' leaf views cost in index requests over a worker-like session, from a key census
(quick.py --hzkeys: every call of _hz_views in order), under three cache designs:
  v7      one entry per product key (M, q, N, row strip, column strip), cap 96 entries, oldest entry dropped (the file's form);
  sides   one entry per TREE: left operand (M, q), right operand (q, N), output tree (M, N), leaf-sum buffer (M, N), each with the
          strip flags; a product key is assembled from four of them (no request when all four exist);
  both of them with the first <prebound> keys of a frequency list built at setup (fit on other networks).
usage: sim_side_views.py <quick json with hz_keys> [<second json: the list is fitted on it>]"""
import collections
import json
import sys

ROW_FULL, COL_FULL = (0, 2, 3, 5), (0, 2, 5, 6)


def slices(chunks):
    out, rest = 0, list(chunks)
    while rest:
        run = [rest[0]]
        if len(rest) > 1:
            step = rest[1] - rest[0]
            for c in rest[1:]:
                if c == run[-1] + step:
                    run.append(c)
                else:
                    break
        out += 1
        rest = rest[len(run):]
    return out


def counts(hzr, hzc):
    """-> (matmul groups, fill views) of one output tree"""
    groups = collections.defaultdict(list)
    for c in range(7):
        groups[((not hzr) or c in ROW_FULL, (not hzc) or c in COL_FULL)].append(c)
    mats = sum(slices(ch) for ch in groups.values())
    fills = (slices([c for c in range(7) if c not in ROW_FULL]) if hzr else 0) + (slices([c for c in range(7) if c not in COL_FULL]) if hzc else 0)
    return mats, fills


def load(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get('rows', d)
    return [(r['rank'], [tuple(k[:5]) for k in r['hz_keys']]) for r in rows]


def simulate(session, design, prebound=()):
    per_net, cache, sides = [], collections.OrderedDict(), set()
    for key in prebound:
        cost(key, design, cache, sides)
    for rank, keys in session:
        total = 0
        for key in keys:
            total += cost(key, design, cache, sides)
        per_net.append(total)
    return per_net


def cost(key, design, cache, sides):
    M, q, N, hzr, hzc = key
    if key in cache:
        return 0
    mats, fills = counts(hzr, hzc)
    if design == 'v7':
        if len(cache) >= 96:
            cache.popitem(last=False)
        cache[key] = True
        return 4 * mats + 2 * fills            # A, B once (shared), the output tree's and the buffer's views and fills
    cache[key] = True
    out = 0
    for side, n in ((('a', M, q, hzr, hzc), mats), (('b', q, N, hzr, hzc), mats), (('c', M, N, hzr, hzc), mats + fills), (('s', M, N, hzr, hzc), mats + fills)):
        if side not in sides:
            sides.add(side); out += n
    return out


session = load(sys.argv[1])
fit = load(sys.argv[2]) if len(sys.argv) > 2 else None
uses = collections.Counter(k for _r, keys in session for k in keys)
print(f'{len(session)} networks; strip products per network {[len(k) for _r, k in session]}; distinct product keys over the session {len(uses)}; '
      f'distinct trees: left {len({(k[0], k[1], k[3], k[4]) for k in uses})}, right {len({(k[1], k[2], k[3], k[4]) for k in uses})}, output {len({(k[0], k[2], k[3], k[4]) for k in uses})}')
for design in ('v7', 'sides'):
    per = simulate(session, design)
    print(f'{design:6s} index requests for leaf views per network: {per}  mean {sum(per) / len(per):.0f}, first {per[0]}, the others {sum(per[1:]) / max(len(per) - 1, 1):.0f}')
if fit is not None:
    order = [k for k, _n in collections.Counter(k for _r, keys in fit for k in keys).most_common()]
    for n in (16, 32, 64):
        for design in ('v7', 'sides'):
            setup = sum(simulate([(0, order[:n])], design))
            per = simulate(session, design, prebound=order[:n])
            print(f'{design:6s} with the {n} most used keys of the other session bound at setup ({setup} requests there): per network {per}  mean {sum(per) / len(per):.0f}, first {per[0]}')
