"""TAG fable-lead 2026-10-19: build ZERO-BILL dial finite-difference arms on the science base (value-only: one global reassigned at the
tail, read at predict time; bill unchanged). Each arm = candidate_<tag>_sci_v1.py with two lines inserted before the final
_FAST_RESCUE_CAP_S line: a comment and `<NAME> = <python literal>`. The new literal is the base literal transformed by a scale
(tuple/float/nested tuple scaled elementwise) or given verbatim.
usage: build_dial_fd_v3.py <base sci candidate> <tag> <NAME> <scale or 'lit:<python literal>'> <note>"""
import ast
import re
import sys
from pathlib import Path

base, tag, name, spec, note = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
out = base.with_name('candidate_%s_sci_v1.py' % tag)
assert not out.exists(), out
lines = base.read_text().split('\n')
assert lines[-1] == '' and lines[-2].startswith('_FAST_RESCUE_CAP_S = '), lines[-2][:60]
defs = [i for i, l in enumerate(lines) if l.startswith(name + ' = ')]
assert defs, name
src = lines[defs[-1]][len(name) + 3:].split('#')[0].strip()
try:
    cur = ast.literal_eval(src)
except Exception:
    raise SystemExit('base value of %s is not a literal: %r' % (name, src))


def scale(v, s):
    if isinstance(v, (int, float)):
        return v * s
    if isinstance(v, dict):
        return {k: scale(x, s) for k, x in v.items()}
    return tuple(scale(x, s) for x in v)


if spec.startswith('lit:'):
    new = ast.literal_eval(spec[4:])
else:
    new = scale(cur, float(spec))
lit = repr(new)
compile(lit, 'lit', 'eval')
comment = '# [fable-lead 2026-10-19] ZERO-BILL DIAL FD on store ranks (arm %s): %s %s (base %s at line %d)' % (tag, name, note, src[:80], defs[-1] + 1)
lines[-2:-2] = [comment, '%s = %s' % (name, lit)]
text = '\n'.join(lines)
compile(text, str(out), 'exec')
out.write_text(text)
print('%s: %s = %s' % (out.name, name, lit[:100]))
