"""[opus-lowres] physics-module switches on S2tr 46e9315d (static, appended last), for (warm requests, S8 noise-free xMSE, bill) pricing:
edge  _EDGE_ENABLED = False                    (the message edge; the governor's LEAN flag)
gncov _GN_COV_LAYERS = ()                      (the GN covariance term; LEAN flag)
k4l   K4L local term off: _k4l_local skipped, _k4l_dk and _k4l_final zeros (births stay: NP reads them; as LEAN)
f4c   F4c off: _f4c_step and _f4c_birth skipped (as LEAN)"""
import hashlib, sys
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
SKIP = '''
def _modsw_skip(fn):
    def skip(*args, **kwargs):
        return None
    return skip
'''
ARMS = {
 'edge': '_EDGE_ENABLED = False',
 'gncov': '_GN_COV_LAYERS = ()',
 'k4l': SKIP + '''
_k4l_local = _modsw_skip(_k4l_local)
_MODSW_K4L_DK = _k4l_dk
_MODSW_K4L_FINAL = _k4l_final


def _k4l_dk(*args, **kwargs):
    return fnp.zeros(args[2].shape, dtype=fnp.float64)


def _k4l_final(mu, v):
    z = fnp.zeros(v.shape, dtype=fnp.float64)
    _K415_ARMED[0] = z if _K4_15 else None
    return z
''',
 'f4c': SKIP + '''
_f4c_step = _modsw_skip(_f4c_step)
_f4c_birth = _modsw_skip(_f4c_birth)
'''}
for tag in sys.argv[1:]:
    o = D/f'cand_modsw_{tag}_v1.py'
    with o.open('x') as f:
        f.write(src + f'\n\n# BEGIN OPUS-LOWRES MODULE SWITCH {tag} (static, appended last)\n{ARMS[tag]}\n# END OPUS-LOWRES MODULE SWITCH {tag}\n')
    compile(o.read_text(), str(o), 'exec'); print(tag, o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
