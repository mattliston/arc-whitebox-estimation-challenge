"""[opus-grow] build one-knob grow variants of S2tr 46e9315d (append-last value fragments; two asserted single-line patches of LIVE defs)."""
import hashlib, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
G = O/'lead/runs/opus_grow_v1'
def patch_last(s, old, new, live_def):
    assert s.count(old) >= 1; i = s.rindex(old); d = s.rindex('\ndef ' + live_def + '(')
    assert i > d and s.find('\ndef ', d + 1) > i, f'last {old[:40]!r} not inside the live {live_def}'
    return s[:i] + new + s[i + len(old):]
SORT = "_jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])"
ENER = "_jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # [opus-grow] JORD: energy order (UVP leading r_u coordinates)"
BIRTH = "selected, births = select(_e2_gate_min, gate_floor)"
BIRTHL = "selected, births = select(_e2_gate_min, _GROW_BF.get(l, gate_floor))   # [opus-grow] per-layer births floor"
frag = lambda tag, body: f"\n\n# BEGIN [opus-grow] VARIANT {tag} (append LAST to S2tr 46e9315d; one knob, values only)\n{body}\n# END [opus-grow] VARIANT {tag}\n"
V = {
  'jord':   patch_last(src, SORT, ENER, '_astra_bcc_push_impl') + frag('jord', "_GROW_TAG = 'jord'"),
  'lcr384': src + frag('lcr384', "_LCR_RU = 384"),
  'lrk48':  src + frag('lrk48', "_LR_K = 48"),
  'lrk64':  src + frag('lrk64', "_LR_K = 64"),
  'k4l3f':  src + frag('k4l3f', "_K4L34_BY_LAYER = {l: (1.0, 0.5) for l in range(8, 16)}"),
  'f4c3':   src + frag('f4c3', "_F4C_AGE = 3\n_F4C_THETA = (0.704467345174, 0.665177791806, 0.628)\n_F4C_C = ((0.0, 0.459128468809), (0.0, 0.258157425747), (0.0, 0.145))\n"
                                "_F4C_C_BY_BIRTH = {6: ((0.0, 0.473219284687), (0.0, 0.258157425747), (0.0, 0.145))}"),
  'bfe':    patch_last(src, BIRTH, BIRTHL, '_astra_fhm_core') + frag('bfe', "_GROW_BF = {l: 0.005 for l in range(1, 8)}"),
  'bfl':    patch_last(src, BIRTH, BIRTHL, '_astra_fhm_core') + frag('bfl', "_GROW_BF = {l: 0.005 for l in range(8, 15)}"),
}
for k, s in V.items():
    p = G/f'cand_s2tr_grow_{k}_v1.py'; p.write_text(s); print(k, hashlib.sha256(s.encode()).hexdigest()[:12], len(s) - len(src))
