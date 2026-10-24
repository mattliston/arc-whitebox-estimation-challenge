"""[opus-late] (2) zero-bill dials of S2tr 46e9315d, OPPOSITE direction to the screened grows; value-only, except k21k which first applies the
engineer2 make_k21k_v1.py text route (reproduced here with its assertions: both _gn_dcov defs, the k21n row factor -> _k21k_rowf; empty table =
parent bit for bit) and then sets one value: +0.10 at L8-14 (physicist 10-12 23:29Z item 4, 'first test k21k +0.05/+0.10 at L8-14')."""
import hashlib, re
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
frag = lambda tag, body: f"\n\n# BEGIN [opus-late] DIAL {tag} (append LAST to S2tr 46e9315d; zero-bill value)\n{body}\n# END [opus-late] DIAL {tag}\n"
lines = src.split('\n'); OLD, NEW = "_e2_H(th['k21n']) * _e2_H(1 - kink)", "_k21k_rowf(l, th, kink, _e2_H)"
defs = [i for i, s in enumerate(lines) if s.startswith('def _gn_dcov(l, mu, Cpre, K21, k4, aux):')]
assert len(defs) == 2 and src.count(OLD) == 2 and not [s for s in lines if re.match(r'_gn_dcov\s*=', s)]
for d in defs:
    end = next(i for i in range(d + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
    hits = [i for i in range(d, end) if OLD in lines[i]]
    assert len(hits) == 1 and lines[hits[0]].startswith('    dK21 = ')
    assert any(lines[i] == '    kink = (_e2_H(fnp.abs(a)) < 1.5).astype(fnp.float64)' for i in range(d, hits[0]))
    lines[hits[0]] = lines[hits[0]].replace(OLD, NEW) + '   # [opus-late] k21k route (as make_k21k_v1.py)'
helper = '''
# BEGIN [opus-late] K21K BY LAYER (route as engineer2 make_k21k_v1.py)
_K21K_BY_LAYER = {}


def _k21k_rowf(l, th, kink, hold):
    rowf = hold(th['k21n']) * hold(1 - kink)
    c = _K21K_BY_LAYER.get(l)
    if c is None:
        return rowf
    return hold(rowf) + hold(c * kink)
# END [opus-late] K21K BY LAYER'''
ksrc = '\n'.join(lines).rstrip('\n') + '\n' + helper + '\n'
V = {'ssb85': src + frag('ssb85', "_SS_BETA = (0.85,) * 15   # live 0.9105 (1.0 screened +0.35%)"),
     'k4l03': src + frag('k4l03', "_K4L34_BY_LAYER = {l: (0.3, 0.5) for l in range(8, 16)}   # K4L age-3 strength 0.5 -> 0.3 (1.0 screened +0.21%)"),
     'gk04': src + frag('gk04', "_GK4 = 0.4   # live 0.6"),
     'k21k10': ksrc + frag('k21k10', "_K21K_BY_LAYER = {l: 0.10 for l in range(8, 15)}"),
     'k21k00': ksrc}
for k, s in V.items():
    (O/f'lead/runs/opus_late_v1/cand_s2tr_late_{k}_v1.py').write_text(s); print(k, hashlib.sha256(s.encode()).hexdigest()[:12])
