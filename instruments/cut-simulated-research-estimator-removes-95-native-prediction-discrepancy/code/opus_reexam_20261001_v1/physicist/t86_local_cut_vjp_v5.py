#!/usr/bin/env python3
"""Bounded local VJP/FD check of AST-extracted production rowcut functions.

No network, truth or engine is loaded. A separately observed used-rank fixture
must reproduce its production endpoints before any derivative is interpreted.
Run under the normal memory guard after coordinating the single compute slot.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

SOURCE_SHA = 'bb824a6c4772ab5a394e6ee5da7c09602c54f5fe6d72d9b2470b21a7ee05a288'
FUNCTIONS = ('top', 'push', 'repair_diagonal', 'final_star', 'radial_columns')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone_tree(t, value, double=False):
    if isinstance(value, t.Tensor):
        # Production closures never mutate fixture leaves; copy containers only.
        # The single differentiated leaf is explicitly cloned before AD.
        x = value.detach()
        return x.double() if double and x.is_floating_point() else x
    if isinstance(value, dict):
        return {k: clone_tree(t, v, double) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(t, v, double) for v in value)
    return value


def at(value, path):
    for key in path:
        value = value[key]
    return value


def replace(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


def extracted(path, double):
    text = Path(path).read_text()
    root = ast.parse(text)
    install = [n for n in root.body if isinstance(n, ast.FunctionDef) and n.name == 'install']
    require(len(install) == 1, 'Unexpected production installer')
    nodes = [n for n in install[0].body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    require({n.name for n in nodes} == set(FUNCTIONS), 'Missing production closure')
    push = next(n for n in nodes if n.name == 'push')
    kept = []
    replaced = []
    for node in push.body:
        name = node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
        if name in ('mu', 'vv', 'alpha'):
            replaced.append(name)
            if name == 'alpha':
                kept.append(ast.parse('alpha = fixture_alpha').body[0])
        else:
            kept.append(node)
    require(replaced == ['mu', 'vv', 'alpha'], 'Selection source sites changed')
    push.body = kept
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    raw = ast.unparse(module)
    # Float64 pass diagnoses the same algebra without FP32 rounding. The mixed
    # pass compiles the original AST and is the actual production derivative.
    if double:
        raw = raw.replace('t.float32', 't.float64')
    return compile(raw, '<extracted pinned rowcuts ' + ('float64' if double else 'production') + '>', 'exec'), hashlib.sha256(raw.encode()).hexdigest()


def run_functions(t, fixture, code):
    layer = int(fixture['layer'])
    st = dict(fixture['pre_state'], layer=layer, repair=None, blocks=())
    inputs = fixture['push_inputs']
    cd = SimpleNamespace(ws={layer: fixture['pre_state']['ws']}, native_push_context=dict(blocks=fixture['blocks']))
    parent = lambda *args, **kwargs: tuple(fixture['parent_result'])
    namespace = dict(t=t, math=math, cd=cd, st=st, o=fixture['options'], parent_push=parent, log=[], fixture_alpha=fixture['alpha'])
    exec(code, namespace)
    original_top = namespace['top']
    observed_top = []
    def checked_top(score, multiple):
        selected, omitted = original_top(score, multiple)
        observed_top.append((multiple, selected, omitted))
        return selected, omitted
    namespace['top'] = checked_top
    args = [inputs[name] for name in ('w', 'p', 'lam', 'previous', 'Q', 'F', 'core', 'stops')]
    pushed = namespace['push'](t, layer, *args, rank=inputs.get('rank', 384))
    require(len(observed_top) == len(fixture['top_calls']), 'Top-call count mismatch')
    for actual, saved in zip(observed_top, fixture['top_calls']):
        multiple, selected, omitted = actual
        require(multiple == saved['multiple'] and selected.numel() == saved['count'], 'Top-call rule mismatch')
        require(t.equal(t.cat((selected, omitted)), saved['order']), 'Top-call roster mismatch')
    require(t.equal(st['keep'], fixture['expected']['keep']) and t.equal(st['mask'], fixture['expected']['mask']), 'Saved kept/masked roster mismatch')
    diagonal = namespace['repair_diagonal'](layer, fixture['diagonal']['input'])
    radial = fixture['radial']
    energy, column = namespace['radial_columns'](layer, radial['energy_in'], radial['column_in'], radial['aa'], radial['beta'])
    result = dict(repair=st['repair'], energy=energy, column=column, pp=pushed[0], diagonal=diagonal)
    # Explicitly exclude detached NP law capture: its lack of AD is intentional.
    namespace.clear()  # break exec function/globals cycles; tensor AD graph remains valid
    return result, st


def compare(t, actual, expected):
    require(actual.shape == expected.shape, 'Endpoint shape mismatch')
    return dict(exact=bool(t.equal(actual, expected)), maxabs=float((actual.double()-expected.double()).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture'); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    require(not Path(args.out).exists(), 'Existing report must be preserved')
    import torch as t
    t.set_num_threads(4); t.set_num_interop_threads(1)
    fixture = t.load(args.fixture, map_location='cpu', weights_only=True)
    require(fixture['schema'] == 'cutsim_local_vjp_fixture_v1', 'Unknown fixture schema')
    require(int(fixture['rank']) == 651 and int(fixture['layer']) in (7, 13), 'Only scheduled used651 fixtures accepted')
    require(sha(fixture['source_path']) == fixture['source_sha256'] == SOURCE_SHA, 'Production source changed')
    base, _ = run_functions(t, clone_tree(t, fixture), extracted(fixture['source_path'], False)[0])
    expected = fixture['expected']
    checks = {name: compare(t, base[name], expected[name]) for name in ('repair', 'pp')}
    checks.update({name: compare(t, base[name], fixture['radial'][name + '_out']) for name in ('energy', 'column')})
    checks['diagonal'] = compare(t, base['diagonal'], fixture['diagonal']['output'])
    require(all(x['exact'] for x in checks.values()), 'Fixture does not reproduce production endpoints exactly')
    paths = []
    for i, block in enumerate(fixture['blocks']):
        paths += [('block%d_U' % i, ('blocks', i, 'U'), ('repair', 'energy', 'column')),
                  ('block%d_F' % i, ('blocks', i, 'F'), ('repair', 'energy', 'column'))]
    paths += [('young_p', ('push_inputs', 'p'), ('repair',)),
              ('lam', ('push_inputs', 'lam'), ('repair',)),
              ('aa', ('radial', 'aa'), ('energy',)),
              ('beta', ('radial', 'beta'), ('column',)),
              ('diagonal_input', ('diagonal', 'input'), ('diagonal',)),
              ('energy_input', ('radial', 'energy_in'), ('energy',)),
              ('column_input', ('radial', 'column_in'), ('column',))]
    report = dict(schema='cutsim_local_vjp_report_v1', utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                  fixture=str(Path(args.fixture).resolve()), fixture_sha256=sha(args.fixture), source_sha256=SOURCE_SHA,
                  rank=651, layer=int(fixture['layer']), threads=4, endpoint_checks=checks, derivatives=[],
                  scope='Local pinned repair code with observed alpha held fixed; chosen targets do not affect alpha. Does not certify whole-card, covariance-push or upstream QR derivative.')
    for double in (True, False):
        code, code_sha = extracted(fixture['source_path'], double)
        reference = clone_tree(t, fixture, double)
        for ti, (name, path, endpoints) in enumerate(paths):
            x0 = at(reference, path)
            if not x0.numel():
                continue
            generator = t.Generator().manual_seed(6101100 + ti)
            direction = t.randn(x0.shape, generator=generator, dtype=t.float64)
            direction *= max(float(x0.double().square().mean().sqrt()), 1e-4) / float(direction.square().mean().sqrt())
            x = x0.detach().clone().requires_grad_(True)
            values = clone_tree(t, reference)
            replace(values, path, x)
            outputs, state = run_functions(t, values, code)
            for ei, endpoint in enumerate(endpoints):
                y = outputs[endpoint].double()
                cotangent = t.randn(y.shape, generator=t.Generator().manual_seed(991 + ei), dtype=t.float64) / math.sqrt(y.numel())
                scalar = (y*cotangent).sum()
                gradient = t.autograd.grad(scalar, x, retain_graph=True, allow_unused=True)[0] if scalar.requires_grad else None
                gradient = t.zeros_like(x) if gradient is None else gradient
                require(bool(t.isfinite(gradient).all()), 'Nonfinite local gradient')
                points = []
                for epsilon in ((1e-3, 3e-4, 1e-4) if double else (1e-2, 3e-3, 1e-3)):
                    xx = [(x0.double() + sign*epsilon*direction).to(x0.dtype) for sign in (1, -1)]
                    effective = (xx[0].double()-xx[1].double())/(2*epsilon)
                    ad = float((gradient.double()*effective).sum())
                    yy = []
                    for perturbed in xx:
                        vv = clone_tree(t, reference)
                        replace(vv, path, perturbed)
                        out, roster = run_functions(t, vv, code)
                        require(t.equal(roster['keep'], state['keep']), 'Local perturbation changed kept roster')
                        require(len(roster['blocks']) == len(state['blocks']) and all(t.equal(a[0], b[0]) for a,b in zip(roster['blocks'],state['blocks'])), 'Local perturbation changed old-tail roster')
                        yy.append(float((out[endpoint].double()*cotangent).sum()))
                    fd = (yy[0]-yy[1])/(2*epsilon)
                    points.append(dict(epsilon=epsilon, ad=ad, fd=fd, abs_error=abs(fd-ad),
                                       relative_error=abs(fd-ad)/max(abs(fd),abs(ad),1e-20)))
                report['derivatives'].append(dict(mode='float64_algebra' if double else 'production_mixed', extracted_sha256=code_sha,
                                                  input=name, endpoint=endpoint, finite=True, points=points))
    report['conclusion'] = 'Endpoint identity passed; inspect float64 convergence and mixed-precision plateau. No automatic whole-path PASS.'
    with Path(args.out).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(report=str(Path(args.out).resolve()), layer=report['layer'], checks=len(report['derivatives']))))


if __name__ == '__main__':
    main()
