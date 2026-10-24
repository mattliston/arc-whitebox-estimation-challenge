"""[opus-retune] restore variants of S2tr 46e9315d: the S48 young-read column cut (a >= -2.0, zero beyond) restored through
RTT2's existing rank-2 (vc/mu, power-refined) completion: _RC_COLZ None = complete the whole live roster beyond z1.5;
2.2 = complete to z2.2. Only edits: the two predict-time asserts that pin _RC_COLZ == 2.0 (inside the live _rtt_predict) + one value."""
import hashlib, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
D = O/'lead/runs/opus_retune_v1'
PIN = '_RC_COLZ==2.0'
assert src.count(PIN) == 2
d = src.index('\ndef _rtt_predict('); e = src.find('\nEstimator.predict=_rtt_predict', d)
assert src.count('\ndef _rtt_predict(') == 1 and d < src.index(PIN) and src.rindex(PIN) < e
frag = lambda tag, body: f"\n\n# BEGIN [opus-retune] VARIANT {tag} (append LAST to S2tr 46e9315d; one value)\n{body}\n# END [opus-retune] VARIANT {tag}\n"
V = {'rcoff': (src.replace(PIN, '_RC_COLZ is None') + frag('rcoff', '_RC_COLZ = None')),
     'rc22': (src.replace(PIN, '_RC_COLZ==2.2') + frag('rc22', '_RC_COLZ = 2.2'))}
for k, s in V.items():
    p = D/f'cand_s2tr_retune_{k}_v1.py'; assert not p.exists(); p.write_text(s); print(k, hashlib.sha256(s.encode()).hexdigest()[:12], len(s) - len(src))
