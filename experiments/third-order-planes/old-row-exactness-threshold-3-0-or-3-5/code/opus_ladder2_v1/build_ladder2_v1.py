"""[opus-ladder2] u64 share: S2tr 46e9315d + ONE value-only override appended last (functions read these globals at call time).
DCoff  _DEADCUT_LAYERS = (99, 98)    (no dead-side row cut in the bank push: every layer takes the full push)
DC35   _DEADCUT_Z = 3.5              (from 3.0902)
L15RC  _L15_ROWCUT_Z = 1e9           (L15 k3: CP + star parts on every row, from |mu/sigma| <= 3.5)
OTT30  _OTT_Z = _OTT_NEG_Z = 3.0     (from 2.5)
OTT35  _OTT_Z = _OTT_NEG_Z = 3.5
UVP15  _UVP_ONLY_LAYERS = _LCR_LAYERS (no UVP split at L15: the original full read there; L10-14 unchanged)"""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
ARMS = {'DCoff': '_DEADCUT_LAYERS = (99, 98)', 'DC35': '_DEADCUT_Z = 3.5', 'L15RC': '_L15_ROWCUT_Z = 1e9',
        'OTT30': '_OTT_Z = 3.0\n_OTT_NEG_Z = 3.0', 'OTT35': '_OTT_Z = 3.5\n_OTT_NEG_Z = 3.5', 'UVP15': '_UVP_ONLY_LAYERS = _LCR_LAYERS'}
for tag, ov in ARMS.items():
    o = D/f'cand_s2tr_l2_{tag}_v1.py'
    with o.open('x') as f:
        f.write(src + f'\n\n# BEGIN OPUS-LADDER2 OVERRIDE {tag} (value only, appended last)\n{ov}\n# END OPUS-LADDER2 OVERRIDE {tag}\n')
    compile(o.read_text(), str(o), 'exec'); print(tag, o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
