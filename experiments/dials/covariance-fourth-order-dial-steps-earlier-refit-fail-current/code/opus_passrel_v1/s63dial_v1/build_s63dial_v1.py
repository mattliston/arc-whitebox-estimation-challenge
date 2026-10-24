"""[opus-s63dial] S63's three pre-registered dial arms (cert/RAD/S63 keep_v1.json; trial size 0.02 each, the descent direction of their
64-network gradients: kc 7-10 t -4.05, x2a1 7 t -4.29, k4la3 4-7 t -2.9) as VALUE-ONLY overrides of the native constants on S2tr 46e9315d,
each = S2tr's own current value + 0.02 (the arm's step; S2tr sits at a later point than S63's S60 start):
  KC     _GN_KC[7..10] + 0.02            (kappa_C: C = C0 - (1 + rc_l - kc_l) S; read per layer at call time)
  X2A1   _F4C_C_BY_BIRTH[6] S2 + 0.02     (F4c exact-term S2 strength of age 1 at layer 7 = birth 6; made at setup by _f4c_const)
  K4LA3  _K4L34_BY_LAYER[4..7] = (0.02, 0.0)  (k4la3 shift -1 (off, S2tr) -> -0.98; age-4 stays 0; S60 had -1.0415 here)"""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parents[1]
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
assert "_F4C_C_BY_BIRTH = {6: ((0.0, 0.473219284687), (0.0, 0.258157425747))}" in src
assert "7: 0.477104209804189, 8: 0.477104209804189, 9: 0.477104209804189, 10: 0.44698926193513366" in src
ARMS = {'KC': '_GN_KC = dict(_GN_KC)\n_GN_KC.update({7: 0.477104209804189 + 0.02, 8: 0.477104209804189 + 0.02, 9: 0.477104209804189 + 0.02, 10: 0.44698926193513366 + 0.02})',
        'X2A1': '_F4C_C_BY_BIRTH = {6: ((0.0, 0.473219284687 + 0.02), (0.0, 0.258157425747))}',
        'K4LA3': '_K4L34_BY_LAYER = dict(_K4L34_BY_LAYER)\n_K4L34_BY_LAYER.update({4: (0.02, 0.0), 5: (0.02, 0.0), 6: (0.02, 0.0), 7: (0.02, 0.0)})'}
for tag, ov in ARMS.items():
    o = D/f'cand_s2tr_s63_{tag}_v1.py'
    with o.open('x') as f:
        f.write(src + f'\n\n# BEGIN OPUS-S63DIAL OVERRIDE {tag} (value only, appended last)\n{ov}\n# END OPUS-S63DIAL OVERRIDE {tag}\n')
    compile(o.read_text(), str(o), 'exec'); print(tag, o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
