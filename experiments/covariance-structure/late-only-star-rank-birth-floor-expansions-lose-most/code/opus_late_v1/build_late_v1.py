"""[opus-late] late-only one-knob variants of S2tr 46e9315d (append-last value fragments; per-layer u-leg override wraps the LIVE _uvp_state;
births floor = the asserted one-line per-layer births patch of the live _astra_fhm_core, as in opus_grow_v1)."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
BIRTH = "selected, births = select(_e2_gate_min, gate_floor)"
BIRTHL = "selected, births = select(_e2_gate_min, _GROW_BF.get(l, gate_floor))   # [opus-late] per-layer births floor"
i = src.rindex(BIRTH); d = src.rindex('\ndef _astra_fhm_core('); assert i > d and src.find('\ndef ', d + 1) > i
bsrc = src[:i] + BIRTHL + src[i + len(BIRTH):]
RU = '''_LATE_RU = {RU}
_LATE_UV_PARENT = _uvp_state
def _uvp_state(state):
    # per-layer u-leg rank at late layers only; L15 joins _LCR_LAYERS for the call so the LCR rule reads _LCR_RU there
    global _LCR_RU, _LCR_LAYERS
    l = _UVP_CUR[0]
    if l not in _LATE_RU:
        return _LATE_UV_PARENT(state)
    saved = (_LCR_RU, _LCR_LAYERS)
    _LCR_RU = _LATE_RU[l]
    _LCR_LAYERS = tuple(sorted(set(saved[1]) | {{l}}))
    try:
        return _LATE_UV_PARENT(state)
    finally:
        _LCR_RU, _LCR_LAYERS = saved'''
frag = lambda tag, body: f"\n\n# BEGIN [opus-late] VARIANT {tag} (append LAST to S2tr 46e9315d; one knob, late layers only)\n{body}\n# END [opus-late] VARIANT {tag}\n"
V = {'ss8L': src + frag('ss8L', "_SS_RANK = (-1,) * 12 + (8, 8, 8)   # stars moved into L13, L14, L15"),
     'ss8F': src + frag('ss8F', "_SS_RANK = (-1,) * 14 + (8,)   # star moved into L15 only"),
     'ss32L': src + frag('ss32L', "_SS_RANK = (-1,) * 12 + (32, 32, 32)"),
     'ru320L': src + frag('ru320L', RU.format(RU="{13: 320, 14: 320, 15: 320}")),
     'ru384L': src + frag('ru384L', RU.format(RU="{13: 384, 14: 384, 15: 384}")),
     'ru384F': src + frag('ru384F', RU.format(RU="{15: 384}")),
     'bfL2': bsrc + frag('bfL2', "_GROW_BF = {13: 0.005, 14: 0.005}")}
for k, s in V.items():
    (O/f'lead/runs/opus_late_v1/cand_s2tr_late_{k}_v1.py').write_text(s); print(k, hashlib.sha256(s.encode()).hexdigest()[:12])
