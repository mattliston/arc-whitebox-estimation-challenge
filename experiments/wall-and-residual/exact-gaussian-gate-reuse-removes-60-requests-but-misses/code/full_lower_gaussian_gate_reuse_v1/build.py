"""Checked local edits; never rewrite or execute the submitted parent."""
import ast
from common import HERE, PARENT, PARENT_SHA, sha, write, pin

assert sha(PARENT) == PARENT_SHA
text = PARENT.read_text()
lines = text.splitlines(keepends=True)
nodes = ast.parse(text).body
edits = []

def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)

for name in ['gaussian_raw', '_nonlinear_pair_batch', '_nonlinear_first_gaussian', '_nonlinear_terminal_live', '_predict_binary']:
    candidates = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name]
    assert candidates
    node = candidates[-1]
    original = ''.join(lines[node.lineno-1:node.end_lineno])
    body = original
    if name == 'gaussian_raw':
        body = replace_once(body, 'def gaussian_raw(mu, v):', 'def gaussian_raw(mu, v, _prepared=None):')
        body = replace_once(body,
            '    v = fnp.maximum(v, 1e-10)\n    s = fnp.sqrt(v)\n    a = mu / s\n',
            '    if _prepared is None:\n        v = fnp.maximum(v, 1e-10)\n        s = fnp.sqrt(v)\n        a = mu / s\n    else:\n        v, s, a, g = _prepared\n')
        body = replace_once(body, '    g = flops.stats.norm.cdf(a)\n',
                            '    if _prepared is None:\n        g = flops.stats.norm.cdf(a)\n')
    elif name != '_predict_binary':
        body = replace_once(body, "metric='identity'):", "metric='identity', _gaussian_prepared=None):")
        body = replace_once(body, 'gaussian_raw(mu, v)', 'gaussian_raw(mu, v, _gaussian_prepared)')
    else:
        body = replace_once(body,
            '            response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))\n',
            '            _gaussian_prepared = None\n'
            '            if _GAUSSIAN_GATE_REUSE:\n'
            '                _gr_v = fnp.maximum(vv, 1e-10)\n'
            '                _gr_s = fnp.sqrt(_gr_v)\n'
            '                _gr_a = mu / _gr_s\n'
            '                response_gate = flops.stats.norm.cdf(_gr_a)\n'
            '                _gaussian_prepared = (_gr_v, _gr_s, _gr_a, response_gate)\n'
            '                del _gr_v, _gr_s, _gr_a\n'
            '            else:\n'
            '                response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))\n')
        body = replace_once(body,
            '            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n',
            '            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric, _gaussian_prepared=_gaussian_prepared)\n'
            '            del _gaussian_prepared\n')
    edits.append((node.lineno-1, node.end_lineno, body, name, original))

out = lines.copy()
for first, last, body, name, original in sorted(edits, reverse=True):
    out[first:last] = [body]
enabled = ''.join(out) + '\n\n# Current-layer tuple only; no cross-call numerical cache.\n_GAUSSIAN_GATE_REUSE = True\n'
disabled = enabled.replace('_GAUSSIAN_GATE_REUSE = True\n', '_GAUSSIAN_GATE_REUSE = False\n')
assert enabled.count('_GAUSSIAN_GATE_REUSE = True\n') == 1
for name, source in [('candidate.py', enabled), ('disabled.py', disabled)]:
    ast.parse(source)
    with (HERE/name).open('x') as f:
        f.write(source)
write(HERE/'build.json', dict(parent=pin(PARENT), candidate=pin(HERE/'candidate.py'), disabled=pin(HERE/'disabled.py'),
    changed_functions=[e[3] for e in edits], numerical_work_executed=False, labels_accessed=False,
    disabled_differs_only_in_flag=True, protocol=pin(HERE/'PROTOCOL.md'), builder=pin(__file__)))
print(dict(candidate=pin(HERE/'candidate.py'), disabled=pin(HERE/'disabled.py')), flush=True)
