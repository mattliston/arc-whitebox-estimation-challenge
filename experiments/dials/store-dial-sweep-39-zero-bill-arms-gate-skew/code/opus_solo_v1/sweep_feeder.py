"""[opus-solo] self-refilling store sweep: keeps queue/todo at >= MIN arms from a generated family (per-layer knob x layer group x direction),
records every finished arm's store verdict in SWEEP_RESULTS.md, and when an arm qualifies (store xnet <= -0.15%, >= 20/32 lower, n >= 30)
builds the running stack (all qualified fragments on the kinkoracle base) as cand_decomp_stk<N>_v1.py and queues it (verification).
usage: sweep_feeder.py [--once]   (loop: every 60 s)"""
import hashlib, json, math, re, statistics as st, sys, time
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
K = O/'lead/runs/opus_kinkoracle_v1'; S = O/'lead/runs/opus_solo_v1'; Q = S/'queue'
MIN_TODO = 6
BASE = (K/'cand_s2tr_kinkoracle_v2.py').read_text()
OTT = '''_L2H_OTT_TABLE = %s
_L2H_OTTW_PARENT = _shared_old384_push


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    global _OTT_Z, _OTT_NEG_Z
    saved = (_OTT_Z, _OTT_NEG_Z)
    z = _L2H_OTT_TABLE.get(layer)
    if z is not None:
        _OTT_Z, _OTT_NEG_Z = z
    try:
        return _L2H_OTTW_PARENT(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _OTT_Z, _OTT_NEG_Z = saved
'''
DC = '''_L2H_DEADCUT_TABLE = %s
_L2H_DEADCUT_BASE = _DEADCUT_Z
_L2H_DCP_PARENT = _deadcut_push


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    global _DEADCUT_Z
    saved = _DEADCUT_Z
    _DEADCUT_Z = _L2H_DEADCUT_TABLE.get(layer, _L2H_DEADCUT_BASE)
    try:
        return _L2H_DCP_PARENT(layer, w32, p, previous, newborn, lam, state, a)
    finally:
        _DEADCUT_Z = saved
'''
LCR = '''_L2H_LCR_TABLE = %s
_L2H_LCR_INNER = _UBC_PARENT_STATE


def _l2h_lcr_state(state):
    global _LCR_RU
    saved = _LCR_RU
    _LCR_RU = _L2H_LCR_TABLE.get(_UVP_CUR[0], saved)
    try:
        return _L2H_LCR_INNER(state)
    finally:
        _LCR_RU = saved


_UBC_PARENT_STATE = _l2h_lcr_state
'''
GROUPS = {'l1215': range(12, 16), 'l0811': range(8, 12), 'l1315': range(13, 16), 'l1011': range(10, 12)}
def family():
    arms = {}
    for g, L in GROUPS.items():
        Ls = list(L)
        for f, tag in ((0.9, 'm10'), (1.1, 'p10'), (0.8, 'm20'), (1.2, 'p20')):
            arms[f'ssb_{tag}_{g}'] = "_SS_BETA = tuple(((_b * %s) if (_i + 1) in %s else _b) for _i, _b in enumerate(_SS_BETA))\n" % (f, Ls)
            arms[f'pe34_{tag}_{g}'] = "_PE34_BY_LAYER = {_l: ((_a * %s if _a != 1.0 else 1.0, _b * %s if _b != 1.0 else 1.0) if _l in %s else (_a, _b)) for _l, (_a, _b) in _PE34_BY_LAYER.items()}\n" % (f, f, Ls)
            arms[f'k4l34_{tag}_{g}'] = "_K4L34_BY_LAYER = {_l: (tuple(%s * _x for _x in _v) if _l in %s else _v) for _l, _v in _K4L34_BY_LAYER.items()}\n" % (f, Ls)
            arms[f'gne_{tag}_{g}'] = "_GN_EDGE = {_l: (%s * _v if _l in %s else _v) for _l, _v in _GN_EDGE.items()}\n" % (f, Ls)
            arms[f'np_{tag}_{g}'] = "_NP_STRENGTH = tuple(((%s * _v) if _i in %s else _v) for _i, _v in enumerate(_NP_STRENGTH))\n" % (f, Ls)
        for z, tag in ((2.75, 'z275'), (2.25, 'z225')):
            arms[f'ott_{tag}_{g}'] = OTT % ('{l: (%s, %s) for l in %s}' % (z, z, Ls))
        for z, tag in ((3.25, 'z325'), (3.75, 'z375'), (4.5, 'z45')):
            arms[f'dc_{tag}_{g}'] = DC % ('{l: %s for l in %s}' % (z, Ls))
    for ru, tag in ((288, 'r288'), (320, 'r320'), (224, 'r224')):
        for g in ('l1011', 'l1215'):
            arms[f'lcr_{tag}_{g}'] = LCR % ('{l: %s for l in %s}' % (ru, list(GROUPS[g])))
    return arms
def store(tag):
    ctl = {int(p.stem.split('_')[-1]): json.loads(p.read_text()) for p in K.glob('res_ctl_*.json')}
    a = {int(p.stem.split('_')[-1]): json.loads(p.read_text()) for p in K.glob(f'res_d_{tag}_*.json')}
    rs = sorted(set(a) & set(ctl))
    if len(rs) < 3: return None
    x = [100 * math.log(a[r]['U'] / ctl[r]['U']) for r in rs]; f = [100 * math.log(a[r]['flops'] / ctl[r]['flops']) for r in rs]; t = [p + q for p, q in zip(x, f)]
    return dict(n=len(rs), x=st.mean(x), xse=st.stdev(x) / math.sqrt(len(rs)), bill=st.mean(f), t=st.mean(t), tse=st.stdev(t) / math.sqrt(len(rs)), lower=sum(v < 0 for v in x), tlower=sum(v < 0 for v in t))
def qualifies(m): return m is not None and m['n'] >= 30 and m['t'] <= -0.15 and m['tlower'] >= 20
def frag_of(tag):
    src = (K/f'cand_decomp_{tag}_v1.py').read_text()
    m = re.search(r'(# BEGIN \[opus-solo\] [^\n]*\b' + re.escape(tag) + r'\b[^\n]*\n.*?# END \[opus-solo\] [^\n]*\n)', src, re.S)
    return m.group(1) if m else None
def write_arm(tag, frag, kind='SWEEP ARM'):
    p = K/f'cand_decomp_{tag}_v1.py'
    if p.exists(): return False
    text = BASE + '\n\n# BEGIN [opus-solo] %s %s (judged on the 32 store ranks)\n%s# END [opus-solo] %s %s\n' % (kind, tag, frag, kind, tag)
    compile(text, tag, 'exec'); p.write_text(text); (Q/'todo'/tag).write_text(hashlib.sha256(text.encode()).hexdigest()[:8]); return True
def tick():
    done = sorted(p.name for p in (Q/'done').iterdir()) if (Q/'done').exists() else []
    results = {}; lines = []
    seen = set()
    for tag in done + [p.name.rsplit('.', 1)[0] for p in (Q/'running').iterdir()]:
        if tag in seen: continue
        seen.add(tag); m = store(tag); results[tag] = m
        if m: lines.append('| %s | %d | %+.3f (%.3f) | %+.3f | %+.3f (%.3f) | %d/%d | %s |' % (tag, m['n'], m['x'], m['xse'], m['bill'], m['t'], m['tse'], m['tlower'], m['n'], 'QUALIFIES' if qualifies(m) else ''))
    (S/'SWEEP_RESULTS.md').write_text('# store sweep (32 store ranks vs S2tr control; xnet = xMSE + bill, %)\n\n| arm | n | xMSE (se) | bill | xnet (se) | xnet lower | verdict |\n|---|---|---|---|---|---|---|\n' + '\n'.join(sorted(lines)) + '\n')
    winners = sorted(t for t in done if qualifies(results.get(t)) and not t.startswith('stk'))
    stk_path = S/'STACK_current.json'; cur = json.loads(stk_path.read_text()) if stk_path.exists() else dict(parts=[], n=0)
    if winners and winners != cur['parts']:
        n = cur['n'] + 1; frags = [frag_of(t) for t in winners]
        if all(frags):
            write_arm(f'stk{n}', '\n\n'.join(frags), 'STACK'); cur = dict(parts=winners, n=n); stk_path.write_text(json.dumps(cur))
            (S/'SWEEP_RESULTS.md').open('a').write(f'\nSTACK stk{n} queued = {winners}\n')
    todo = [p.name for p in (Q/'todo').iterdir()]
    if len(todo) < MIN_TODO:
        for tag, frag in family().items():
            if tag in todo or (K/f'cand_decomp_{tag}_v1.py').exists(): continue
            if write_arm(tag, frag): todo.append(tag)
            if len(todo) >= MIN_TODO + 4: break
    return len(todo), winners
if __name__ == '__main__':
    once = '--once' in sys.argv
    while True:
        try:
            n, w = tick(); print(time.strftime('%H:%M:%SZ', time.gmtime()), 'todo', n, 'winners', w, flush=True)
        except Exception as e:
            print('feeder error', repr(e), flush=True)
        if once: break
        time.sleep(60)
