"""Official paid49bd GNN tariff for one fixed transport-gate times first128 readout.

Current paid GNN plus the frozen extra readout. No training/capture/truth reads.
The fixed synthetic perturbation is not a fitted candidate or accuracy claim.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import ast, base64, hashlib, inspect, json, subprocess, sys, zlib
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp

HERE = Path(__file__).resolve().parent
D = HERE.parent
V = D.parent/'astra_oct20_vm_frontier_v1'
RUNS = Path('/home/arcworker/arc-runs')
OUT = RUNS/'first128_gate90_price_v1'
SOURCE = D/'runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py'
SCI = '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80'
PARENT = D/'breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py'
PARENT_SHA = '4da68821a5db9212035531a25dc8ab0c197a44d84a1e5482168d4223d349032c'
PREREG = 'bfb1f9a1de49665b5427e98a5d83e3d13fac151741f450e73403e98f58ee7438'
TAIL_SHA = 'a928f7a23d7c2baafce825876653e6027fee792d20c3156960958cd173593b36'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
sys.path.insert(0, str(V))
import fit_frontier_v1 as f
import check_frontier_tiny_v6 as tiny
assert sha(f.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
assert sha(tiny.__file__) == 'a6246110ed4e3aaeba9f49b51b9bb3bd855e6656fc3508e8703b6c1c45aac376'


def encoded(params):
    return {key: (list(value.shape), base64.b64encode(zlib.compress(value.astype('<f4').tobytes())).decode())
            for key, value in params.items()}


def current_functions(source):
    tree = ast.parse(source)
    names = ('_E2Hold', 'silu', '_e2_gnn_consts', '_e2_gnn_layer', '_gop2_contract',
             '_gop2_messages', '_e2_messages', '_gop2_gnn_step', '_gop2_message_product',
             '_w3r_metric', '_w3r_product', '_unpack_static_fp32', '_hc90_hidden_head', '_fh90_first_head')
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    body = [definitions[name] for name in names]
    for name in ('_fg90_gate_feature', '_fg90_gate_head'):
        if name in definitions:
            body.append(definitions[name])
    # Preserve both actual _hc_bias definitions and the intervening alias.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_hc_bias', '_hcf_bias'):
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_HCF_PARENT_HC_BIAS' for t in node.targets):
            body.append(node)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('_HC_U', '_HC_COLUMNS', '_HCF_U', '_HCF_COLUMNS', '_VMFR_ORIGINAL_W3_ENCODER'):
                values[key] = ast.literal_eval(node.value)
    assert np.shape(values['_HC_U']) == (5, 128) and np.shape(values['_HCF_U']) == (1, 128)
    assert np.shape(values['_VMFR_ORIGINAL_W3_ENCODER']) == (128, 8)
    return compile(ast.Module(body=body, type_ignores=[]), '<exact-current-paid-gnn>', 'exec'), values


def environment(code, literals, selected):
    env = dict(fnp=fnp, _static_base64=base64, _static_zlib=zlib,
               _GOP2_ON=True, _GOP2_ACTIVE=[True], _GNN_PARAMS=None,
               _E2_GNN_STATE={}, _E2_MSG={}, _FAST_RESCUE_FRAC={},
               _HC_CONST={}, _HCF_CONST={}, _MESSAGE_EDGE_PARAMETERS=None,
               _GOP2_STATS=dict(calls=0, layers=[], energy_blocks=0, message_misses=0),
               _W3R_STATS=dict(calls=0, metric_builds=0), _VMFR_CM=8,
               _VMFR_ORIGINAL_W3_ENCODER=literals['_VMFR_ORIGINAL_W3_ENCODER'],
               _VMFR_W3_ENCODER_ARRAY=None, _W3R_METRIC=None,
               _HC_COLUMNS=literals['_HC_COLUMNS'], _HCF_COLUMNS=literals['_HCF_COLUMNS'],
               _HC_STATE=dict(l=None, raw={}), _HCF_STATE=dict(raw={}, ctx=None, rows15=None),
               _HC_STATS=dict(columns=0, used=0), _HCF_STATS=dict(columns=0, used=0))
    env['_HC_U'] = np.asarray(literals['_HC_U'])[:, selected].tolist()
    env['_HCF_U'] = np.asarray(literals['_HCF_U'])[:, selected].tolist()
    exec(code, env)
    return env


def standardized_bias(params, literals, selected, hc, hcf, n):
    out = params['enc0_b'].copy()
    for raw, columns, weights, final in ((hc, literals['_HC_COLUMNS'], literals['_HC_U'], False),
                                        (hcf, literals['_HCF_COLUMNS'], literals['_HCF_U'], True)):
        if not raw:
            continue
        z = []
        for key in columns:
            x, mask, count = raw.get(key, (np.zeros(n), np.zeros(n) if final else None, n))
            mask = np.ones(n) if mask is None else mask
            xx = x*mask
            centered = (xx-xx.sum()/max(count, 1))*mask
            scale = np.sqrt(max(count-1, 0))/np.sqrt(max(np.sum(centered*centered), 1e-300))
            z.append(np.clip(centered*scale, -8, 8).astype('f4'))
        out = out + np.einsum('kn,kc->nc', np.stack(z), np.asarray(weights, 'f4')[:, selected], optimize=False)
    return out


def native_bills(out, source_sha):
    complete, manifest = read(out/'COMPLETE.json'), read(out/'RUN_MANIFEST.json')
    assert complete['successful'] and not complete['pending'] and len(complete['results']) == 12
    assert all(code == 0 for code in complete['results'].values())
    assert manifest['candidate_sha256'] == source_sha and manifest['blas_threads'] == 4
    rows, pins = {}, []
    for index in range(1, 13):
        guardpath = out/('part%02d_GUARD.json'%index)
        guard = read(guardpath)
        assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit'] == 0 and not guard['cleanup_errors']
        assert guard['guard_sha256'] == GUARD and guard['max_owned_swap_bytes'] == 0
        assert guard['thread_provenance']['blas_threads'] == 4
        assert all(value == 0 for value in guard['cgroup_events_final'].values())
        part = out/('part%02d'%index)
        assert read(part/'manifest.json')['sha256'] == source_sha
        for line in (part/'rows.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert row['phase'] == 'scored' and row['rank'] not in rows
            rows[row['rank']] = int(row['flops'])
        pins.append(dict(rows_path=str(part/'rows.jsonl'), rows_sha256=sha(part/'rows.jsonl'), guard_sha256=sha(guardpath)))
    assert len(rows) == 32
    return rows, dict(out=str(out), source_sha256=source_sha, complete_sha256=sha(out/'COMPLETE.json'),
                     manifest_sha256=sha(out/'RUN_MANIFEST.json'), parts=pins)


def main():
    assert sys.platform == 'linux' and fs.__version__.startswith('0.12.3')
    assert not os.environ.get('FLOPSCOPE_SERVER_URL') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
    assert not OUT.exists() and sha(SOURCE) == SCI and sha(PARENT) == PARENT_SHA
    assert sha(HERE/'PREREG_v1.json') == PREREG
    literals, _ = f.literals_and_functions(SOURCE.read_text())
    parent_literals, _ = f.literals_and_functions(PARENT.read_text())
    params = f.decode(literals['_GNN_LITERAL'])
    parent_params = f.decode(parent_literals['_GNN_LITERAL'])
    assert set(params) == set(parent_params) and all(np.array_equal(params[k], parent_params[k]) for k in params)
    assert params['first_hidden_head'].shape == (16, 128)
    code, constants = current_functions(SOURCE.read_text())
    selected = np.arange(128)
    student = {key: value.copy() for key, value in params.items()}
    student['gate_first_head'] = (1e-4*np.sin(np.arange(16*128, dtype='f8')).reshape(16,128)).astype('f4')
    student['gate_first_head'][0] = 0
    assert set(student) == set(params)|{'gate_first_head'}
    assert all(np.array_equal(student[key], params[key]) for key in params)
    tail = HERE/'GATE_FIRST128_TAIL_v1.py'
    assert sha(tail) == TAIL_SHA
    candidate_code, candidate_constants = current_functions(SOURCE.read_text()+'\n'+tail.read_text())
    assert candidate_constants == constants
    # Tiny independent oracle exposes both nonlinear stages without invoking estimator code.
    oracle_node = ast.parse(inspect.getsource(tiny.oracle)).body[0]
    oracle_node.name = 'oracle_hidden'
    ret = [node for node in oracle_node.body if isinstance(node, ast.Return)][-1]
    ret.value = ast.Tuple(elts=[ast.Name(id=k, ctx=ast.Load()) for k in ('out', 'delta', 'h', 'first_hidden')], ctx=ast.Load())
    first = [node for node in oracle_node.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'h' for t in node.targets)][0]
    oracle_node.body.insert(oracle_node.body.index(first)+1, ast.parse('first_hidden=h.copy()').body[0])
    ast.fix_missing_locations(oracle_node)
    oracle_env = dict(np=np, gop=tiny.gop, w3=tiny.w3)
    exec(compile(ast.Module(body=[oracle_node], type_ignores=[]), '<independent-width-oracle>', 'exec'), oracle_env)
    rng = np.random.default_rng(202610210738)
    results, cases = [], []
    feature_checks = 0
    parent_states = {}
    for n in (16, 1024):
        weights = rng.normal(0, np.sqrt(2/n), (n, n)).astype('f4')
        state = rng.normal(0, .3, (n, 64)).astype('f4')
        sigma = rng.uniform(.2, 1.2, n).astype('f4')
        inputs = {layer: (rng.normal(0, .5, (n, 30))*params['scales'][layer]+params['centers'][layer]).astype('f4') for layer in range(1, 16)}
        raw = {}
        for layer in range(1, 16):
            mask = (np.arange(n)%5 != layer%5).astype('f8')
            hc = {key: (rng.normal(size=n), None if j%2 else mask, n if j%2 else int(mask.sum()))
                  for j, key in enumerate(constants['_HC_COLUMNS']) if not (layer%3 == 0 and j == 4)}
            hcf = {constants['_HCF_COLUMNS'][0]: (rng.normal(size=n), mask, int(mask.sum()))} if layer == 15 else {}
            raw[layer] = hc, hcf
        for terminal_hc in (False, True):
            models = []
            for label, pp, keep in [('parent_first128_paid', params, np.arange(128)), ('gate_first128_synthetic_only', student, selected)]:
                is_candidate = label == 'gate_first128_synthetic_only'
                env = environment(candidate_code if is_candidate else code, constants, keep)
                records, saved = [], {}
                with fs.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                    before, count = ctx.flops_used, len(ctx.op_log)
                    p = env['_unpack_static_fp32'](encoded(pp))
                    env['_GNN_PARAMS'] = p
                    setup, setup_calls = ctx.flops_used-before, len(ctx.op_log)-count
                    sig, ww, st = [fnp.asarray(value) for value in (sigma, weights, state)]
                    xx = {layer: fnp.asarray(value) for layer, value in inputs.items()}
                    rr = {layer: tuple({key: (fnp.asarray(x), None if mask is None else fnp.asarray(mask), count)
                                        for key, (x, mask, count) in item.items()} for item in pair) for layer, pair in raw.items()}
                    w2 = ww*ww; w3 = w2*ww; w4 = w3*ww; w6 = w3*w3
                    powers = dict(T013=fnp.stack((ww.T, w2.T, w3.T)), T135=fnp.stack((w2.T, w4.T, w6.T)))
                    env['_e2_wpow'] = lambda W, need6=False: powers
                    for phase in ('cold', 'warm'):
                        for layer in range(1, 16):
                            env['_E2_MSG'].clear()
                            if 8 <= layer <= 14:
                                env['_e2_messages'](env['_e2_gnn_consts'](p, n), ww, st)
                            hc, hcf = rr[layer]
                            env['_HC_STATE'].update(l=layer, raw={} if layer == 15 and not terminal_hc else dict(hc))
                            env['_HCF_STATE']['raw'] = dict(hcf)
                            before, count = ctx.flops_used, len(ctx.op_log)
                            # Observe existing helper locals on return; no extra fnp operation inside the bill.
                            traced = []
                            if is_candidate:
                                helper_code = env['_fg90_gate_feature'].__code__
                                def trace_feature(frame, event, value):
                                    if event == 'return' and frame.f_code is helper_code:
                                        traced.append(tuple(frame.f_locals[key] for key in ('h','gate','feature')))
                                sys.setprofile(trace_feature)
                            try:
                                delta, got = env['_gop2_gnn_step'](layer, xx[layer], sig, ww, st, p)
                            finally:
                                if is_candidate:
                                    sys.setprofile(None)
                            bill, calls = ctx.flops_used-before, len(ctx.op_log)-count
                            values = np.asarray(got).copy(), np.asarray(delta).copy()
                            assert all(np.isfinite(value).all() for value in values)
                            state_key = n, terminal_hc, phase, layer
                            if is_candidate:
                                assert values[0].tobytes() == parent_states[state_key]
                                assert len(traced) == 1
                                hh, gg, bb = [np.asarray(value).copy() for value in traced[0]]
                                assert hh.shape == bb.shape == (n,128) and gg.shape == (n,1)
                                assert hh.dtype == gg.dtype == bb.dtype == np.dtype('f4')
                                assert bb.tobytes() == (hh*gg).tobytes()
                                normalized = np.clip((inputs[layer]-pp['centers'][layer])/pp['scales'][layer],-8,8)
                                if layer == 15:
                                    normalized *= np.asarray(env['_e2_gnn_consts'](p,n)['mask'])
                                assert gg.tobytes() == normalized[:,5:6].tobytes()
                                feature_checks += 1
                            else:
                                parent_states[state_key] = values[0].tobytes()
                            if phase == 'cold':
                                saved[layer] = values
                                if n == 16:
                                    hc_raw, hcf_raw = raw[layer]
                                    bias = standardized_bias(pp, constants, keep, {} if layer == 15 and not terminal_hc else hc_raw, hcf_raw, n)
                                    expected, edelta, hidden2, first_hidden = oracle_env['oracle_hidden'](pp, np.asarray(constants['_VMFR_ORIGINAL_W3_ENCODER'], 'f8'), layer, inputs[layer], sigma, weights, state, bias)
                                    scale = sigma*pp['rms'][layer]*(1+.25*np.tanh(pp['gain'][layer:layer+1]))
                                    edelta += scale*(hidden2@pp['hidden_head'][layer]+first_hidden@pp['first_hidden_head'][layer])
                                    if is_candidate:
                                        edelta += scale*((first_hidden*normalized[:,5:6])@pp['gate_first_head'][layer])
                                    np.testing.assert_allclose(values[0], expected, rtol=5e-5, atol=5e-5)
                                    np.testing.assert_allclose(values[1], edelta, rtol=1e-4, atol=2e-8)
                            else:
                                np.testing.assert_array_equal(values[0], saved[layer][0])
                                np.testing.assert_array_equal(values[1], saved[layer][1])
                            records.append(dict(layer=layer, phase=phase, flops=int(bill), calls=calls))
                    np.testing.assert_array_equal(np.asarray(st), state)
                    np.testing.assert_array_equal(np.asarray(ww), weights)
                models.append(dict(model=label, parameter_decode_flops=int(setup), parameter_decode_calls=setup_calls, records=records))
                cases.append(dict(n=n, terminal_hc_present=terminal_hc, model=label, actual_hc_hcf_functions=True,
                                  current_first_skip_present=True, preserved_original_w3_shape=[128, 8], cold_warm_byte_exact=True))
                print(json.dumps(dict(utc=utc(), n=n, terminal_hc_present=terminal_hc, model=label, complete=True)), flush=True)
            for phase in ('cold', 'warm'):
                old, new = [[r for r in model['records'] if r['phase'] == phase] for model in models]
                setup_delta = models[1]['parameter_decode_flops']-models[0]['parameter_decode_flops']
                call_delta = sum(r['flops'] for r in new)-sum(r['flops'] for r in old)
                results.append(dict(n=n, terminal_hc_present=terminal_hc, phase=phase,
                                    whole_gnn_flop_delta=int(call_delta), parameter_decode_flop_delta=int(setup_delta),
                                    total_geometry_flop_delta=int(call_delta+setup_delta),
                                    charged_call_delta=sum(r['calls'] for r in new)-sum(r['calls'] for r in old),
                                    parameter_decode_call_delta=models[1]['parameter_decode_calls']-models[0]['parameter_decode_calls'],
                                    parent=old, student=new))
    baseline, baseline_pin = native_bills(RUNS/'edge8_local90_per_layer960_markline32_blas4_v1', '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4')
    paid, paid_pin = native_bills(RUNS/'first128_paid_nocensus_native32_v1', SCI)
    assert set(baseline) == set(paid)
    projections = []
    for result in results:
        if result['n'] != 1024:
            continue
        delta = result['total_geometry_flop_delta']
        rows = [dict(rank=rank, verified466_flops=baseline[rank], measured_paid_stack_flops=paid[rank],
                     projected_gate_first_paid_flops=paid[rank]+delta,
                     geometry_bill_pct=100*delta/baseline[rank],
                     paid_repair_and_first128_bill_pct=100*(paid[rank]/baseline[rank]-1),
                     projected_total_bill_pct=100*((paid[rank]+delta)/baseline[rank]-1)) for rank in sorted(baseline)]
        projections.append(dict(phase=result['phase'], terminal_hc_present=result['terminal_hc_present'], rows=rows,
                                mean_geometry_bill_pct=float(np.mean([r['geometry_bill_pct'] for r in rows])),
                                mean_paid_total_bill_pct=float(np.mean([r['projected_total_bill_pct'] for r in rows]))))
    assert feature_checks == 120
    passed = all(0 <= row['mean_geometry_bill_pct'] <= .01 for row in projections) and all(row['whole_gnn_flop_delta'] >= 0 for row in results)
    report = dict(utc=utc(), successful=True, status='PAID49BD_GATE_FIRST128_OFFICIAL_ADDED_TARIFF', passes_price_gate=passed,
                  source_sha256=SCI, scientific_parent_sha256=PARENT_SHA, script_sha256=sha(__file__), prereg_sha256=PREREG, tail_sha256=TAIL_SHA,
                  official_version=fs.__version__, rows=results, cases=cases, full_paid_stack_projections=projections,
                  measured_verified466_bill_custody=baseline_pin, measured_paid_stack_bill_custody=paid_pin,
                  estimator_imports=0, new_network_predictions=0, no_capture_or_truth_arrays_read=True,
                  literal_decode_constant_cache_and_hc_hcf_included=True, all15_current_gnn_calls_measured=True,
                  independent_n16_oracle_pass=True, all120_float32_product_and_normalized_gate_checks=feature_checks,
                   own_step_state_bytes_preserved=True, synthetic_readout_not_fitted=True, no_native_accuracy_or_runtime_claim=True,
                  no_adjusted_gain_claim=True, no_candidate_export=True,
                  limits=['Synthetic fixed interaction head; not a fitted model or accuracy result.',
                          'Unchanged message precompute and weight powers excluded identically.',
                          'Changed trajectories may alter whole-network sparse rosters and actual tariff.',
                          'All paid runtime debit included from measured49bd; a fitted source needs its own full-stack native bill and runtime margin.'])
    OUT.mkdir(exist_ok=False)
    f.jwrite(OUT/'PRICE_v1.json', report)
    print(json.dumps(dict(utc=report['utc'], status=report['status'], passes_price_gate=passed,
                         projections=[{k: p[k] for k in ('phase', 'terminal_hc_present', 'mean_geometry_bill_pct', 'mean_paid_total_bill_pct')} for p in projections])), flush=True)


if __name__ == '__main__':
    main()
