"""[opus-dialx] value-only variants of S2tr 46e9315d (one knob each, appended last): _NP_STRENGTH x0.92 / x1.08 (all layers),
_BSK 0.34 -> 0.2 / 0.5, _GBK 0 -> 0.5 (live: _gq's polynomial path is on since _GSK = 1.0, _GK4 = 0.6)."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
D = O/'lead/runs/opus_dialx_v1'
frag = lambda tag, body: f"\n\n# BEGIN [opus-dialx] VARIANT {tag} (append LAST to S2tr 46e9315d; one knob, values only)\n{body}\n# END [opus-dialx] VARIANT {tag}\n"
V = {'np92': "_NP_STRENGTH = tuple(x * 0.92 for x in _NP_STRENGTH)", 'np108': "_NP_STRENGTH = tuple(x * 1.08 for x in _NP_STRENGTH)",
     'bsk20': "_BSK = 0.2", 'bsk50': "_BSK = 0.5", 'gbk50': "_GBK = 0.5"}
for k, body in V.items():
    s = src + frag(k, body)
    p = D/f'cand_s2tr_dialx_{k}_v1.py'
    p.open('x').write(s)
    print(k, hashlib.sha256(s.encode()).hexdigest()[:12])
