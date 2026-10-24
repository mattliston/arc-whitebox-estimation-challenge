"""[opus-lean] S2t-lean variants: S2t (e0553d2e) with ONLY the _ASTRA_SM40_PLANS literal reduced (dropped shapes fall back to the
pre-Smirnov read path: exact algebra, rounding-level change). VA = minus b709's 62 added keys; VB = VA minus L6g4 (13); VC = VB minus L6g8 (14)."""
import ast, hashlib, json
from pathlib import Path
L = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs')
W = Path(__file__).resolve().parent; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
S2T = L/'opus_release_v1/s2t/candidate_S2t_cblazy_v1.py'; S1 = L/'opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py'
assert sha(S2T).startswith('e0553d2e1536')
lines = S2T.read_text().splitlines(keepends=True)
idx = [i for i, l in enumerate(lines) if l.startswith('_ASTRA_SM40_PLANS = ')]; assert len(idx) == 1; i = idx[0]
plans = ast.literal_eval(lines[i].split(' = ', 1)[1])
s1l = [l for l in S1.read_text().splitlines() if l.startswith('_ASTRA_SM40_PLANS = ')]; s1 = ast.literal_eval(s1l[-1].split(' = ', 1)[1])
added = {k for k in plans if k not in s1}; assert len(plans) == 167 and len(added) == 62
drops = dict(VA=added, VB=added | {k for k, v in plans.items() if (v[1], v[2]) == (6, 4)},
             VC=added | {k for k, v in plans.items() if (v[1], v[2]) in ((6, 4), (6, 8))})
man = {}
for name, drop in drops.items():
    keep = {k: v for k, v in plans.items() if k not in drop}
    out = list(lines); out[i] = '_ASTRA_SM40_PLANS = ' + repr(keep) + '\n'
    assert ast.literal_eval(out[i].split(' = ', 1)[1]) == keep
    p = W/f'candidate_S2t_lean_{name}_v1.py'; p.write_text(''.join(out)); compile(''.join(out), str(p), 'exec')
    man[name] = dict(path=str(p), sha256=sha(p), keys=len(keep), dropped=len(drop))
(W/'LEAN_MANIFEST_v1.json').write_text(json.dumps(dict(parent=dict(path=str(S2T), sha256=sha(S2T)), variants=man), indent=2) + '\n')
print(json.dumps({k: (v['keys'], v['sha256'][:12]) for k, v in man.items()}))
