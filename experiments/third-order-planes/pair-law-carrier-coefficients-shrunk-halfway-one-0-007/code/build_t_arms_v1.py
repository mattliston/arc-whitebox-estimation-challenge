"""[fable-lead 2026-10-19] t-family of Astra's current-roster carrier calibration on the BASE a7ffd713 (same generator as
astra_oct18_accuracy_v1/build_current_carrier_half_v1.py): theta(s) = 1 + (1 - s) (theta0 - 1): s = shrink fraction of the current source calibration toward the formal law (s=0 current, s=0.5 Astra's HALF, s=1 law),
coefH rows 1-6 regenerated with the parent's own pure _np_tables; one literal assignment appended; science line appended.
usage: build_t_arms_v1.py <base.py> <out dir> t [t ...]"""
import ast, copy, hashlib, json, sys
from pathlib import Path
base_path, out = Path(sys.argv[1]), Path(sys.argv[2]); ts = [float(x) for x in sys.argv[3:]]
A = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1')
baseline = json.loads((A / 'BUILD_GROW_CARRIER_CALIBRATION_v1.json').read_text())['baseline_theta']
src = base_path.read_text(); tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [x.id for x in node.targets if isinstance(x, ast.Name)]
        if any(n.startswith('_NP_') or n == '_S19_LAM' for n in names):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            for name in names:
                ns[name] = value
for name in ('_np_ages', '_np_key', '_np_tables'):
    node = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name][0]
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<base-pure-table>', 'exec'), ns)
print('base theta literals:', {k: ns.get(k) for k in baseline}, 'baseline_theta:', baseline)
base = ns['_np_tables'](1024)
assert base['key'] == ns['_np_key'](1024)
for t in ts:
    theta = {k: 1 + (1 - t) * (v - 1) for k, v in baseline.items()}
    ns2 = dict(ns); ns2.update(theta)
    for name in ('_np_ages', '_np_key', '_np_tables'):
        node = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name][0]
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<base-pure-table>', 'exec'), ns2)
    other = ns2['_np_tables'](1024)
    for key in base:
        if key not in ('coefH', 'key'):
            assert base[key] == other[key], key
    patched = copy.deepcopy(base); patched['coefH'][1:7] = other['coefH'][1:7]
    tag = 's%03d' % round(t * 100)
    frag = ('\n\n# [fable-lead] current-roster carrier calibration shrink s=%r of the current source calibration toward the formal law (theta (births/rows/diag %r); coefH births 1-6 only.\n'
            "_NP_TABLES['coefH'][1:7] = %r\n_FAST_RESCUE_CAP_S = 1e9   # science variant: concurrent judges must not trip the rescue\n" % (t, theta, patched['coefH'][1:7]))
    body = src + frag
    compile(body, tag, 'exec')
    p = out / ('candidate_%s_sci_v1.py' % tag); p.write_text(body)
    print(tag, hashlib.sha256(body.encode()).hexdigest()[:8], 'theta', {k: round(v, 4) for k, v in theta.items()})
