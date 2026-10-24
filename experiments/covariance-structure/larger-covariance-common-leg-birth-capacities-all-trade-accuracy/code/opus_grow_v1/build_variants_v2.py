"""[opus-grow] v2 variants of S2tr 46e9315d: star-sum rank grows (-1, 32 at every layer) and half steps of the over-gate knobs."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
G = O/'lead/runs/opus_grow_v1'
BIRTH = "selected, births = select(_e2_gate_min, gate_floor)"
BIRTHL = "selected, births = select(_e2_gate_min, _GROW_BF.get(l, gate_floor))   # [opus-grow] per-layer births floor"
i = src.rindex(BIRTH); d = src.rindex('\ndef _astra_fhm_core('); assert i > d and src.find('\ndef ', d + 1) > i
bsrc = src[:i] + BIRTHL + src[i + len(BIRTH):]
frag = lambda tag, body: f"\n\n# BEGIN [opus-grow] VARIANT {tag} (append LAST to S2tr 46e9315d; one knob, values only)\n{body}\n# END [opus-grow] VARIANT {tag}\n"
V = {'ssm1': src + frag('ssm1', "_SS_RANK = (-1,) * 15"), 'ss32': src + frag('ss32', "_SS_RANK = (32,) * 15"),
     'lcr320': src + frag('lcr320', "_LCR_RU = 320"), 'bfe75': bsrc + frag('bfe75', "_GROW_BF = {l: 0.0075 for l in range(1, 8)}")}
for k, s in V.items():
    p = G/f'cand_s2tr_grow_{k}_v1.py'; p.write_text(s); print(k, hashlib.sha256(s.encode()).hexdigest()[:12])
