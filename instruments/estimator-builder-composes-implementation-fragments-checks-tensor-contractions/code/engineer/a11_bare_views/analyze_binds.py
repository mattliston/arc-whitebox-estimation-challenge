"""TAG opus_engineer a11: what the lazily bound tree views of a quick.py --binds run are, and what a bare-view family list covers.
usage: analyze_binds.py <quick json with bind_list> [fit ranks] [test ranks]
Decodes every bind (arena start, shape) into (side, depth, leaf rows, leaf cols): side A/B/C by the depth's region offsets,
'acc' = the k-split's second leaf buffer behind the C tree, 'r7' = the depth-7 read's private region, '?' = anything else.
Prints the per-network bind bill, the distribution by (side, depth), and for the FAMILIES below: keys, setup flops, setup requests,
and the share of each network's bind flops they cover."""
import collections
import json
import sys

REGION = {2: (1 << 20, 1 << 22, 1 << 22), 3: (5619712, 22478848, 22478848), 4: (78675968, 78675968, 78675968),
          5: (137682944, 137682944, 68841472), 6: (90354432, 120472576, 120472576), 7: (7 ** 7 * 64, 7 ** 7 * 128, 7 ** 7 * 128)}
R7 = (7 ** 7 * 128,) * 3


def decode(start, shape):
    L = sum(1 for x in shape[:-2] if x == 7)
    if len(shape) != L + 2 or L not in REGION:
        return ('?', L, tuple(shape[L:]))
    r, c = shape[-2:]
    reg = REGION[L]
    offs = {0: 'A', reg[0]: 'B', reg[0] + reg[1]: 'C', reg[0] + reg[1] + 7 ** L * r * c: 'acc'}
    if start in offs:
        return (offs[start], L, r, c)
    if L == 7 and start in (R7[0], 2 * R7[0], 2 * R7[0] + 7 ** 7 * r * c):
        return ('r7', L, r, c)
    return ('?', L, r, c)


# (side, depth, leaf-row range, leaf-col range), most valuable first
FAMILIES = [
    ('C', 6, range(8, 17), range(1, 65)), ('B', 6, range(8, 17), range(1, 65)),            # push packets and their tails
    ('A', 6, range(8, 17), range(17, 97)), ('B', 6, range(17, 97), range(8, 17)),           # read chunks
    ('C', 6, range(1, 17), range(1, 17)), ('acc', 6, range(1, 17), range(1, 17)), ('A', 6, range(1, 17), range(1, 17)), ('B', 6, range(1, 17), range(1, 17)),
    ('B', 6, range(1, 8), range(1, 65)), ('A', 6, range(8, 17), range(1, 8)),               # low-rank tier
]


def family_keys(families):
    cap = {(s, L): REGION[L][{'A': 0, 'B': 1, 'C': 2, 'acc': 2}[s]] // 7 ** L // (2 if s == 'acc' else 1) for s, L, _r, _c in families}
    keys = []
    seen = set()
    for s, L, rs, cs in families:
        for c in cs:
            for r in rs:
                if r * c <= cap[(s, L)] and (s, L, r, c) not in seen:
                    seen.add((s, L, r, c)); keys.append((s, L, r, c))
    return keys, seen


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    rows = d['rows']
    tot = collections.Counter(); cnt = collections.Counter()
    per = []
    for row in rows:
        binds = [(decode(b[0], b[1]), b[2]) for b in row['bind_list']]
        per.append((row['rank'], row['flops'], binds))
        for k, re in binds:
            n = 7 ** k[1]
            for x in k[2:] if k[0] != '?' else k[2]:
                n *= x
            tot[k[:2]] += n; cnt[k[:2]] += 1
    print('binds by (side, depth), all networks: Mflop, count')
    for k, v in tot.most_common():
        print(f'   {k}: {v/1e6:9.1f} MF  {cnt[k]:4d}')
    keys, seen = family_keys(FAMILIES)
    setup = sum(7 ** L * r * c for _s, L, r, c in keys)
    print(f'families: {len(keys)} keys, setup flops {setup/1e9:.1f} GF, setup requests {2*len(keys)}')
    for rank, bill, binds in per:
        total = cov = 0
        miss = collections.Counter()
        for k, re in binds:
            n = 7 ** k[1]
            for x in (k[2:] if k[0] != '?' else k[2]):
                n *= x
            total += n
            if k in seen:
                cov += n
            else:
                miss[k] += n
        top = ', '.join(f'{k}:{v/1e6:.0f}' for k, v in miss.most_common(6))
        print(f'  rank {rank:4d}: binds {len(binds):3d} {total/1e6:7.1f} MF = {100*total/bill:.3f}% of bill; covered {100*cov/max(total,1):5.1f}%; left {100*(total-cov)/bill:.3f}% of bill; missed: {top}')
