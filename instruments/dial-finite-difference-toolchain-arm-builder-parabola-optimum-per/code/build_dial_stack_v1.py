"""TAG fable-lead 2026-10-19: stack ZERO-BILL dial moves onto a base candidate (release or science): appends one block at the END of the
file (after any science cap line, since the release base has none) with `<NAME> = <literal>` per move, each literal either verbatim
('NAME=lit:<python literal>') or the base's last literal scaled ('NAME=x<scale>', tuples/dicts elementwise). A line `_K4_PL = {...}` is
only valid on a base that reads _K4_PL (the k4pl science base or a release base patched the same way).
usage: build_dial_stack_v1.py <base candidate> <out candidate> <note> NAME=spec [NAME=spec ...]"""
import ast
import sys
from pathlib import Path

base, out, note = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert not out.exists(), out
text = base.read_text()
lines = text.rstrip('\n').split('\n')


def scale(v, s):
    if isinstance(v, (int, float)):
        return v * s
    if isinstance(v, dict):
        return {k: scale(x, s) for k, x in v.items()}
    return tuple(scale(x, s) for x in v)


block = ['', '# BEGIN FABLE-LEAD DIAL STACK 2026-10-19 (%s): zero-bill dial moves read at predict time; bill unchanged' % note]
for item in sys.argv[4:]:
    name, spec = item.split('=', 1)
    defs = [l for l in lines if l.startswith(name + ' = ')]
    if spec.startswith('lit:'):
        new = ast.literal_eval(spec[4:])
        src = defs[-1][len(name) + 3:].split('#')[0].strip() if defs else '(absent)'
    else:
        assert defs, name
        src = defs[-1][len(name) + 3:].split('#')[0].strip()
        new = scale(ast.literal_eval(src), float(spec[1:]))
    block.append('%s = %r   # was %s' % (name, new, src[:70]))
block.append('# END FABLE-LEAD DIAL STACK')
result = '\n'.join(lines + block) + '\n'
compile(result, str(out), 'exec')
out.write_text(result)
print('written', out.name, 'moves', len(sys.argv) - 4)
