"""[opus-lowres] static request diet on S2tr 46e9315d, NO governor / pushSM / read grid. Base LRB = S2tr -> dm2 (release's asserted
byte-identical edits, opus_exact_v1/dm2/build_dm2_v1.py) -> lazybind v2 fragment (bill -0.10%, requests unchanged). Optional SAFE pair
(two-strip HELD_v1 -> SAFE prebind patch v2 -> five-full HELD_v2, as on the final stack) and diet-flag overrides appended LAST
(the same flags the governor's EXACT tier flips at run time). usage: build_lowres_v1.py TAG [--safe] [--twostrip] [FLAG=False ...]"""
import hashlib, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
R = O/'lead/runs'; D = Path(__file__).resolve().parent
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
tag, args = sys.argv[1], sys.argv[2:]
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert sha(src).startswith('46e9315d')
code = (R/'opus_exact_v1/dm2/build_dm2_v1.py').read_text()
ns = dict(base=src); exec(compile(code[code.index('def sub('):code.index('Path(sys.argv[3]).write_text(s)')], 'dm2', 'exec'), ns); src = ns['s']
lz = (R/'opus_lazybind_v1/fragment_lazybind_v2.py.txt').read_text(); src += lz
parts = []
if '--safe' in args or '--twostrip' in args:
    parts.append(R/'astra_gop2_distinct_safe_policy_v1/two_strip_v1/fragment_covariance_two_strip_HELD_v1.py')
if '--safe' in args:
    parts += [R/'opus_stack_v1/fragment_safe_pb_five_full_key_v2.py.txt', R/'astra_rtt2_safe_two_strip_v1/request_margin_v2/fragment_safe_five_full_HELD_v2.py']
for p in parts:
    t = p.read_text(); src += ('' if t.startswith('\n') else '\n\n') + t
flags = [a for a in args if '=' in a]
if flags:
    src += '\n\n# BEGIN OPUS-LOWRES DIET FLAGS (static, appended last)\n' + '\n'.join(f.replace('=', ' = ') for f in flags) + '\n# END OPUS-LOWRES DIET FLAGS\n'
o = D/f'cand_lowres_{tag}_v1.py'
with o.open('x') as f: f.write(src)
compile(src, str(o), 'exec'); print(tag, o.name, sha(src)[:8], 'parts', [p.name for p in parts], 'flags', flags)
