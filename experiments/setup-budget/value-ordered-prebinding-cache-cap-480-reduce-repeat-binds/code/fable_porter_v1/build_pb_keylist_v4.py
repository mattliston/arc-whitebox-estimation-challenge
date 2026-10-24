#!/usr/bin/env python3
"""TAG fable-porter (PT-67: versioned copy of build_pb_keylist_v2.py with engineer2's S28 rule (lead 11:5x): candidates = single keys used by >= 1
of the 8 selection nets; value = freq x bind_bill / setup_ms (freq = nets/8; bind_bill = 1.076 x 7^L x (rows/2^L) x (cols/2^L) flops; setup_ms = 8 per
key); order = descending value, ties (-freq, -calls, key); per-side limit 480 (cap-480 fragment); total list length capped at --max-total=N (default
500 = 4.0 s at 8 ms/key, so every intended key fits the 4.0 s guard on the slowest platform worker). The old >= 2-of-8 rule is still self-tested on
S24's list (census parsing). Original: TAG fable-porter (PT-55b: versioned copy of build_pb_keylist.py; the ONLY change: optional --exclude=<keys json> removes keys (e.g. the k-split
phase-3 chunk keys from ks_phase3_keys.py) from the re-derived single list after the rule, before the cap). Original: TAG fable-porter (PT-53b, lead 02:4x "R320 with a prebind key list MATCHED to rank 320, captured the way S23's was"): rebuilds the LIVE (last)
_PB_SINGLE table of a candidate from an ipt_key_census json by engineer2's d6x rule, the one that produced S23/S24's live list
(fable_engineer2_v1/d6x/pb_single_d6x.json from ipt_key_census_S18d6x_sel8test4.json): single keys used by >= 2 of the 8 selection nets
(ranks 300..580), ordered by (-nets using it, -total calls, key string), at most 160 per cache side (the cache cap). Self-test first: the rule
must reproduce S24's live list from engineer2's census exactly (set and order), else it refuses. The batched table (_PB_BATCHED, v2b) is KEPT, as
engineer2's d6x re-derivation kept it (02:47 revision; the census rule's batched result is only reported). Prints held-out coverage (35/100/37/39) of the old and the new list on the new census.
usage: build_pb_keylist.py <census.json> <in.py> <out.py>"""
import sys, json, re, collections
from pathlib import Path
H = Path(__file__).resolve().parent; P2 = H.parent
census, src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MAXT = int(next((a.split('=', 1)[1] for a in sys.argv[4:] if a.startswith('--max-total=')), '500'))
EXCL = {tuple(k) for k in json.load(open(next(a.split('=', 1)[1] for a in sys.argv[4:] if a.startswith('--exclude='))))['keys']} if any(a.startswith('--exclude=') for a in sys.argv[4:]) else set()
assert not out_path.exists(), f'refusing to overwrite {out_path}'
def rule_v4(d, prefix, excl=frozenset()):
    sel = [n for n in d['nets'] if 300 <= n['rank'] <= 580]; assert [n['rank'] for n in sel] == list(range(300, 581, 40)), [n['rank'] for n in sel]
    nets, calls = collections.Counter(), collections.Counter()
    for n in sel:
        for k, c in n['used'].items():
            if k[0] == prefix: nets[k] += 1; calls[k] += c
    def parse(k): p = k.split('|'); return (int(p[0][1:]),) + tuple(int(x) for x in p[1:])
    def value(k):
        t = parse(k); L, rows, cols = t[1], t[-2], t[-1]
        return (nets[k] / 8.0) * 1.076 * 7 ** L * (rows / 2 ** L) * (cols / 2 ** L) / 8.0
    keys = sorted(nets, key=lambda k: (-value(k), -nets[k], -calls[k], k))
    out, per = [], collections.Counter()
    for k in keys:
        t = parse(k)
        if t in excl: continue
        if per[t[0]] < 480 and len(out) < MAXT: out.append(t); per[t[0]] += 1
    return out, {parse(k): value(k) for k in keys}
def rule(d, prefix, excl=frozenset()):
    sel = [n for n in d['nets'] if 300 <= n['rank'] <= 580]; assert [n['rank'] for n in sel] == list(range(300, 581, 40)), [n['rank'] for n in sel]
    nets, calls = collections.Counter(), collections.Counter()
    for n in sel:
        for k, c in n['used'].items():
            if k[0] == prefix: nets[k] += 1; calls[k] += c
    keys = sorted([k for k in nets if nets[k] >= 2], key=lambda k: (-nets[k], -calls[k], k))
    out, per = [], collections.Counter()
    for k in keys:
        p = k.split('|'); t = (int(p[0][1:]),) + tuple(int(x) for x in p[1:])
        if t in excl: continue
        if per[t[0]] < 160: out.append(t); per[t[0]] += 1
    return out
def table(src, name, last=True):
    i = (src.rindex if last else src.index)(f'\n{name} = (') + 1; j0 = src.index('(', i); depth = 0
    for j in range(j0, len(src)):
        if src[j] == '(': depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0: break
    return i, j + 1, list(eval(re.sub(r'#[^\n]*', '', src[j0:j + 1])))
def fmt(name, comment, keys):
    rows = [', '.join(repr(t) for t in keys[k:k + 8]) + ',' for k in range(0, len(keys), 8)]
    return f'{name} = (    # {comment}\n' + ''.join(f'    {r}\n' for r in rows) + ')'
# self-test: engineer2's census -> S24's live table
s24 = (P2 / 'fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py').read_text()
e2 = json.load(open(P2 / 'fable_engineer2_v1/d6x/ipt_key_census_S18d6x_sel8test4.json'))
assert rule(e2, 's') == table(s24, '_PB_SINGLE')[2], 'self-test failed: the rule does not reproduce S24 live _PB_SINGLE'
print('self-test PASS: the rule reproduces S24 live _PB_SINGLE (359 keys) from engineer2 d6x census')
d = json.load(open(census)); src = src_path.read_text()
i, j, old = table(src, '_PB_SINGLE'); new, VAL = rule_v4(d, 's', EXCL)
print(f'rule v4: {len(new)} keys (max-total {MAXT}); value covered {sum(VAL[k] for k in new) / sum(VAL.values()):.1%} of all candidates ({len(VAL)}); old-list value {sum(VAL.get(k, 0) for k in old) / sum(VAL.values()):.1%}')
ib, jb, oldb = table(src, '_PB_BATCHED', last=True); newb = [tuple(t) for t in rule(d, 'p')]   # reported only; the table is kept (as engineer2's d6x re-derivation kept it)
out = src[:i] + fmt('_PB_SINGLE', '(kind, L, rows, cols), most frequent first; PT-53b: re-derived for the R320 structure (census ' + census.name + ')', new) + src[j:]
out_path.write_text(out)
print(f'single: old {len(old)} keys {dict(collections.Counter(k[0] for k in old))} -> new {len(new)} keys {dict(collections.Counter(k[0] for k in new))}; '
      f'kept {len(set(old) & set(new))}, dropped {len(set(old) - set(new))}, added {len(set(new) - set(old))}; batched table KEPT {oldb} (census rule would give {newb})')
for n in d['nets']:
    if n['rank'] in (35, 100, 37, 39):
        ks = {tuple(int(x) for x in [k.split('|')[0][1:]] + k.split('|')[1:]) for k in n['used'] if k[0] == 's'}
        print(f"  held-out r{n['rank']}: {len(ks)} single keys; covered by old list {len(ks & set(old)) / len(ks):.0%}, by new list {len(ks & set(new)) / len(ks):.0%}")
