#!/bin/bash
# fable-lead 2026-10-20: stack of tail arms = science copy (for the judge) and release form (for the gates) of the v4a base + the listed tail files, in order.
# usage: make_stack_v1.sh <name> <tail file> [<tail file> ...]   -> candidate_<name>_sci_v1.py and candidate_rel_<name>.py
# astra 2026-10-20: every NEW stack ends with the permanent no-governor tail.
set -euo pipefail
K=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
"$PY" - "$K" "$@" <<'PY'
import ast
import hashlib
from pathlib import Path
import re
import sys

if len(sys.argv) < 4:
    raise SystemExit('usage: make_stack_v1.sh <name> <tail file> [<tail file> ...]')
k, name = Path(sys.argv[1]), sys.argv[2]
if re.fullmatch(r'[A-Za-z0-9_]+', name) is None:
    raise SystemExit('stack name must contain only letters, digits and underscores')
tails = [Path(p) for p in sys.argv[3:]]
for p in tails:
    if not p.is_file():
        raise SystemExit(f'missing tail {p}')
outputs = [k / f'candidate_{name}_sci_v1.py', k / f'candidate_rel_{name}.py']
for p in outputs:
    if p.exists():
        raise SystemExit(f'exists: {p}')
disable = k / 'agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py'
disable_bytes = disable.read_bytes()
if hashlib.sha256(disable_bytes).hexdigest() != 'be4293e44004415fb9faba1b42a26c8428a447ae226475a77a707054aac54607':
    raise SystemExit('permanent no-governor tail hash mismatch')
header = f'\n\n# [fable-lead 2026-10-20] STACK {name} = v4a + tails: {" ".join(sys.argv[3:])}\n'
body = ''.join(p.read_text() for p in tails) + '\n\n' + disable_bytes.decode()
texts = [(k / base).read_text() + header + body
         for base in ['candidate_rel_v4a_sci_v1.py', 'candidate_rel_v4a.py']]
for p, text in zip(outputs, texts):
    compile(text, str(p), 'exec')
    if ast.unparse(ast.parse(text).body[-1]) != '_FAST_RESCUE_FRAC = {}':
        raise SystemExit('permanent no-governor tail must be the final statement')
created = []
try:
    for p, text in zip(outputs, texts):
        with p.open('x') as stream:
            created.append(p)
            stream.write(text)
except BaseException:
    for p in created:
        p.unlink()
    raise
hashes = [hashlib.sha256(p.read_bytes()).hexdigest()[:8] for p in outputs]
print(f'built stack {name}: sci {hashes[0]} rel {hashes[1]}; rescue permanently disabled')
PY
