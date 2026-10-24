"""Official native parent-feature parity and measured tariff for fixed carriers.

Three actual function bundles: original parent, current narrow bank, correction.
The parent functions are reloaded from their pinned source after fixture setup.
No predictions, truth, fitting, or accuracy selection occurs here.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib
import numpy as np
import check_frontier_tiny_v5 as tiny


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(source, name):
    found = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            found = ast.literal_eval(node.value)
    assert found is not None, name
    return found


def environment(source, tail, fnp):
    env = dict(_O20_FLATEXTRA_ON=True,
        _O20_FLATEXTRA_OWNED_PARENT=lambda: None, _O20_FLATEXTRA_HZD_PARENT=lambda: None,
        fnp=fnp, _static_base64=base64, _static_zlib=zlib,
        _GOP2_ON=True, _GOP2_ACTIVE=[True], _GNN_PARAMS=None,
        _E2_GNN_STATE={}, _E2_MSG={}, _FAST_RESCUE_FRAC={},
        _HC_CONST={}, _HCF_CONST={}, _MESSAGE_EDGE_PARAMETERS=None,
        _GOP2_STATS=dict(calls=0, layers=[], energy_blocks=0, message_misses=0),
        _W3R_STATS=dict(calls=0, metric_builds=0), _W3R_METRIC=None)
    exec(tiny.load_functions(source), env)
    exec(tail, env)
    return env


def evaluate(env, P, n, layer, x, sigma, W, state, bias, fs, fnp):
    with fs.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
        Q = {key: fnp.asarray(value) for key, value in P.items()}
        env['_GNN_PARAMS'] = Q
        x_, sigma_, w_, state_, bias_ = [fnp.asarray(value) for value in (x, sigma, W, state, bias)]
        w2 = w_ * w_; w3 = w2 * w_; w4 = w3 * w_; w6 = w3 * w3
        powers = dict(T013=fnp.stack((w_.T, w2.T, w3.T)), T135=fnp.stack((w2.T, w4.T, w6.T)))
        env['_e2_wpow'] = lambda W, need6=False: powers
        env['_hc_bias'] = lambda l, unused: bias_
        outputs, bills, calls = [], [], []
        for repeat in range(3):
            env['_E2_MSG'].clear()
            if repeat == 2:
                env['_e2_messages'](env['_e2_gnn_consts'](Q, n), w_, state_)
            before, count = ctx.flops_used, len(ctx.op_log)
            delta, result = env['_gop2_gnn_step'](layer, x_, sigma_, w_, state_, Q)
            bills.append(ctx.flops_used - before); calls.append(len(ctx.op_log) - count)
            outputs.append((np.asarray(result).copy(), np.asarray(delta).copy(), np.asarray(env['_E2_GNN_STATE'][n]['H']).copy()))
        for other in outputs[1:]:
            for a, b in zip(outputs[0], other):
                np.testing.assert_array_equal(a, b)
        assert env['_E2_GNN_STATE'][n]['P'] is Q
        assert len(env['_E2_GNN_STATE']) == 1
        if '_vmc_parent_energy' in env['_gop2_gnn_step'].__code__.co_names:
            carrier = env['_E2_GNN_STATE'][n]['_vmc_parent_buffers']
            assert carrier['SRC3'].shape == (3, n, env['_VMC_PARENT_WIDTH'])
            assert env['_E2_GNN_STATE'][n]['SRC3'].shape == (3, n, env['_VMFR_CM'])
        return dict(state=outputs[0][0], delta=outputs[0][1], H=outputs[0][2], flops=bills, calls=calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-tail', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--bank-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.official))
    import flopscope as fs
    from flopscope import numpy as fnp
    assert fs.__version__.startswith('0.12.3')
    manifest = json.loads(args.manifest.read_text())
    candidate, = manifest['candidates']
    source = Path(candidate['science']['path']).read_text()
    assert sha(candidate['science']['path']) == candidate['science']['sha256']
    base = Path(candidate['science']['parent_path']).read_text()
    assert sha(candidate['science']['parent_path']) == candidate['science']['parent_sha256']
    parent = Path(manifest['original_teacher_source']).read_text()
    assert sha(manifest['original_teacher_source']) == candidate['teacher_source_sha256']
    tail = Path(candidate['tail']).read_text()
    assert sha(candidate['tail']) == candidate['tail_sha256']
    P = tiny.decode(literal(base, '_GNN_LITERAL'))
    original = tiny.decode(literal(parent, '_GNN_LITERAL'))
    assert literal(source, '_GNN_LITERAL') == literal(base, '_GNN_LITERAL')
    keep = candidate['retained_by_power']
    parent_width, width = candidate['parent_carrier_width'], candidate['message_width']
    h_indices = np.concatenate([np.arange(30)] + [30 + block * parent_width + np.asarray(keep[block % 3]) for block in range(6)])
    rng = np.random.default_rng(20261021)
    cases = []
    for n, layer in ((16, 5), (32, 15), (1024, 5)):
        x = (rng.normal(0, .5, (n, 30)) * P['scales'][layer] + P['centers'][layer]).astype('f4')
        sigma = rng.uniform(.2, 1.2, n).astype('f4')
        W = rng.normal(0, .2 / np.sqrt(n), (n, n)).astype('f4')
        state = rng.normal(0, .3, (n, 64)).astype('f4')
        bias = (P['enc0_b'] + rng.normal(0, .03, (n, len(P['enc0_b'])))).astype('f4')
        oracleP = {key: value.copy() for key, value in P.items()}
        base_bias = bias.copy()
        if '_VMFR_NL_LITERALS' in base:
            base_nl = tiny.decode(literal(base, '_VMFR_NL_LITERALS')[layer])
            base_bias += base_nl['enc0_b'] - P['enc0_b']
        if '_VMFR_NL_LITERALS' in source:
            oracleP.update(tiny.decode(literal(source, '_VMFR_NL_LITERALS')[layer]))
            bias += oracleP['enc0_b'] - P['enc0_b']
        parent_env = environment(base, args.base_tail.read_text(), fnp)
        exec(tiny.load_functions(parent), parent_env)
        parent_env['_VMFR_CM'] = parent_width
        parent_env['_VMFR_ORIGINAL_W3_ENCODER'] = literal(source, '_VMC_PARENT_ENCODER')
        parent_env['_W3R_METRIC'] = None
        parent_env['_GNN_LITERAL'] = literal(parent, '_GNN_LITERAL')
        results = {}
        for name, env, params in (
            ('parent', parent_env, original),
            ('base', environment(base, args.base_tail.read_text(), fnp), P),
            ('corrected', environment(source, tail, fnp), P)):
            results[name] = evaluate(env, params, n, layer, x, sigma, W, state, base_bias if name == 'base' else bias, fs, fnp)
        expected_H = results['parent']['H'][:, h_indices]
        got = results['corrected']
        np.testing.assert_allclose(got['H'], expected_H, atol=5e-5, rtol=5e-5)
        silu = lambda z: z / (1 + np.exp(-z))
        hidden = silu(expected_H @ oracleP['enc0_w'].T + bias)
        hidden = silu(hidden @ oracleP['enc2_w'].T + oracleP['enc2_b'])
        expected = hidden @ oracleP['enc4_w'].T + oracleP['enc4_b']
        delta = (sigma * P['rms'][layer]) * (expected @ P['head'][layer] + P['head_b'][layer]) * (1 + .25 * np.tanh(P['gain'][layer:layer+1]))
        np.testing.assert_allclose(got['state'], expected, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(got['delta'], delta, atol=2e-8, rtol=1e-4)
        cases.append(dict(n=n, layer=layer,
            max_parent_selected_feature_error=float(np.max(np.abs(got['H'] - expected_H))),
            original_narrow_feature_discrepancy=float(np.max(np.abs(results['base']['H'] - expected_H))),
            max_state_error=float(np.max(np.abs(got['state'] - expected))),
            max_delta_error=float(np.max(np.abs(got['delta'] - delta))),
            flops={key: value['flops'] for key, value in results.items()},
            calls={key: value['calls'] for key, value in results.items()},
            repeat_and_prepopulated_cache_byte_exact=True))
        print(json.dumps(cases[-1]), flush=True)
    bank = json.loads(args.bank_report.read_text())
    rows = bank.get('rows', bank.get('per_rank', []))
    assert len(rows) == 32
    bank_flops = [float(row.get('flops1', row.get('candidate_flops', row.get('flops')))) for row in rows]
    assert min(bank_flops) > 0
    large = cases[-1]
    # Every network has 15 GNN layers; cold metric overhead counted once.
    delta_warm = large['flops']['corrected'][1] - large['flops']['base'][1]
    delta_cold_extra = (large['flops']['corrected'][0] - large['flops']['base'][0]) - delta_warm
    bound = 100 * (15 * delta_warm + max(0, delta_cold_extra)) / min(bank_flops)
    output = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
        status='OFFICIAL_PARENT_FEATURE_CACHE_PARITY_PASS', cases=cases,
        manifest_sha256=sha(args.manifest), checker_sha256=sha(__file__),
        candidate_sha256=candidate['science']['sha256'], parent_sha256=candidate['teacher_source_sha256'],
        bank_report_sha256=sha(args.bank_report), minimum_actual_bank_flops=min(bank_flops),
        warm_gnn_added_flops=delta_warm, one_time_added_flops=delta_cold_extra,
        conservative_network_bill_debit_percent=bound,
        bank_point_bill_ceiling_pass=bound <= .10, no_accuracy_or_network_score_claim=True)
    with args.out.open('x') as stream:
        json.dump(output, stream, indent=2); stream.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
