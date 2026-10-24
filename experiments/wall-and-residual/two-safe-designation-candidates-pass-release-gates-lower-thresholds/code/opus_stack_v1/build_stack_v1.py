"""[opus-stack] final-stack builder v1: base (GOV v4f c8dc9989 or its parity copy cfbc563b) + dm2 (asserted edits of live defs) + the
lazybind v2 fragment + optionally the smtable grid-plan dictionary fragment + optionally more appended fragments (in the given order).
Writes the release file and, from the parity copy, the parity copy of the same stack (identical but for the governor thresholds).
usage: build_stack_v1.py <tag> [--smtable] [--frag path ...]"""
import hashlib, importlib.util, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_stack_v1'
BASES = {'release': (O/'lead/runs/opus_governor_v1/candidate_GOV_v4f.py', 'c8dc9989'),
         'parity': (O/'lead/runs/opus_governor_v1/test_parity_v4f.py', 'cfbc563b')}
tag = sys.argv[1]; args = sys.argv[2:]
frags = []
if '--smtable' in args:
    s = (O/'lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py').read_text()
    frags.append(('smtable', s[s.index('\n\n# BEGIN [opus-smtable] GRID PLANS v1'):]))
for i, a in enumerate(args):
    if a == '--frag':
        p = Path(args[i + 1]); frags.append((p.name, p.read_text()))
spec = importlib.util.spec_from_file_location('dm2', O/'lead/runs/opus_exact_v1/dm2/build_dm2_v1.py')
lazy = (O/'lead/runs/opus_lazybind_v1/fragment_lazybind_v2.py.txt').read_text()
out = {}
for kind, (path, prefix) in BASES.items():
    text = path.read_text(); assert hashlib.sha256(text.encode()).hexdigest().startswith(prefix), kind
    src = text
    # dm2: the same three asserted single-occurrence edits as build_dm2_v1.py
    exec_ns = {}
    code = (O/'lead/runs/opus_exact_v1/dm2/build_dm2_v1.py').read_text()
    body = code[code.index('def sub('):code.index('Path(sys.argv[3]).write_text(s)')]
    exec_ns['base'] = src
    exec(compile(body.replace('s = sub(base,', 's = sub(base,', 1), 'dm2', 'exec'), exec_ns)
    src = exec_ns['s'] + lazy
    for name, f in frags:
        src += f
    compile(src, kind, 'exec')
    p = D/f'cand_{tag}_{kind}.py'; p.write_text(src); out[kind] = hashlib.sha256(src.encode()).hexdigest()
    print(kind, p.name, out[kind][:12], len(src))
a, b = (D/f'cand_{tag}_release.py').read_text().splitlines(), (D/f'cand_{tag}_parity.py').read_text().splitlines()
diff = [(x, y) for x, y in zip(a, b) if x != y]
print('release vs parity differing lines:', len(diff), [d[0][:40] for d in diff])
