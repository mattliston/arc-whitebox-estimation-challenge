"""One fixed saved-gradient balance direction, symmetric probes, one endpoint."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
OUT, OLD = HERE/'balance_v1', HERE/'information_v1'
RANKS = (35, 36, 38, 40)
GRADIENT_RANKS = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
MODEL_SHA = 'af5ab0386bde2561543c631ace208f3649836d6581a04a69967705aec008c907'
PINS = {
    HERE/'information.py': 'fee1b6457d39fefbbe72004314efcbf42fff8f5b7f31ac51c6de0fad5ba697cc',
    HERE/'INFORMATION_PROTOCOL.md': '72fbb43d12b4fffc52d5318f358f0dad323282c11c6785ead8e57973dfa99885',
    OLD/'result.json': '73bec2488604faebd40ecd2024990969a0983df8fe6b8c9b9004bb04333cf3f9',
    OLD/'completion.json': '302d0b2d5e3a2337a9133543467c11fe504b866ab79721ab15abae9945af49d2',
    OLD/'finalization.json': 'ec6dc2ed5f96930259e2b8c7e6599adbc7c2b4ea92c729ec295049a74d006602',
    HERE/'information_watch_v1/result.json': '5cd65c4fbcd69f0a6daf0a3f82307edfa06bdf818e31e5345b283d230139478c',
    HERE/'information_watch_v1/completion.json': 'eda36a0ff6a238946f585c54514d078c8ecfb27f99fb3731b9f0dae2715ed7ca',
    HERE/'information_watch_v1/finalization.json': '0d73122f67aadf2c28a2debc6668ff9f37791950dfa6f8562aff4ff38ef0e58c',
    HERE/'INFORMATION_FAILURE_RESULT.md': 'db22baaabf9267e7bb8f4d39c4e3c4daaf1e5dbf070b3fe362563ebc20adec90',
    HERE/'gradient_geometry_diagnostic.py': '3701e1cc0327f69981ccfb0281dfeebc1d182354bdfe0b8c7d4e343ff02759fa',
    HERE/'GRADIENT_GEOMETRY_PROTOCOL.md': '82754a427dfcd90f394bf642d6780c5958aa20d0e3d517cd851cdd56ee593e07',
    HERE/'gradient_geometry_diagnostic_v1/result.json': 'f7f559db836ecbaac4a0e18c5c3f9d7e5c2cfd75204f2672ffe79e0f4f0812ad',
    HERE/'gradient_geometry_diagnostic_v1/completion.json': '20b0dbb6024d7604b72e6c5ff8c3ca36a2e3688f96c8b53816e74031acddab5b',
    HERE/'BALANCE_WATCH_SPEC.md': 'bc4dbc2c60e9068f02c7ae5c95ff701b8b5864c16470761547290f5abb33c6ff',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for key in ('source', 'model', 'protocol', 'watcher'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed180s balance experiment')
    signal.signal(signal.SIGALRM, expired); signal.alarm(180)
    require(not OUT.exists(), 'Exclusive balance experiment; no resume')
    OUT.mkdir()
    result = dict(schema='current_comp_receiver_birth_balance_v1', passed=False, completed=False,
        source_bindings={}, input_bindings={}, target_bindings={}, artifacts={}, calls=[], outputs={},
        ranks=list(RANKS), full_model_calls=0, retired_call_count=0, gradient_calls=0, jvp_calls=0,
        optimizer_updates=0, frozen_parent_parameter_updates=0, parameter_proposals=0,
        fixed_probe_points=3, fitted_points_evaluated=0, training_targets_accessed=False,
        validation_targets_accessed=False, information_promising=False,
        old_information_pilot_still_failed=True, native_gradient_qualified=False,
        no_native_gradient_claim=True, native_score_measured=False, maximum_sampled_mps_driver_bytes=0)
    try:
        require(args.model_sha256 == MODEL_SHA, 'Frozen receiver model')
        def pin(path, expected, group=None):
            path = Path(path).resolve()
            if group is None:
                group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
            require('truthpost' not in str(path) or group == 'target_bindings', 'Separate target ownership')
            require(group != 'target_bindings' or result.get('first12_outputs_sealed_sha256'),
                    'No target-file hash or decode before the first12-output seal')
            require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected,
                    'Pinned bytes: '+str(path))
            result[group][str(path)] = expected
        for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
                HERE/'model.py': args.model_sha256, HERE/'BALANCE_PROTOCOL.md': args.protocol_sha256,
                HERE/'watch_balance.py': args.watcher_sha256}.items():
            pin(path, digest)
        owner = module(HERE/'information.py', 'balance_information_source_owner')
        for path, digest in owner.PINS.items():
            pin(path, digest)
        old, old_end, old_final = (read(OLD/name) for name in ('result.json', 'completion.json', 'finalization.json'))
        require(not old['passed'] and not old['completed'] and old['gradient_calls'] == 20
            and old['jvp_calls'] == 13 and old['nonlinear_calls'] == 0 and old['parameter_proposals'] == 0
            and old['retired_call_count'] == 33 and old['ranks'] == list(GRADIENT_RANKS)
            and old_end['result_sha256'] == PINS[OLD/'result.json'] and not old_end['passed']
            and old_final['completion_sha256'] == PINS[OLD/'completion.json'] and not old_final['passed'],
            'Original numerical failure remains failed, no step or partial score')
        old_watch = read(HERE/'information_watch_v1/result.json')
        require(not old_watch['passed'] and old_watch['cleanup_passed'] and old_watch['retired_calls'] == 33,
                'Old failed worker independently retired')
        qual, qw = read(HERE/'qualification_v3/result.json'), read(HERE/'qualification_watch_v3/result.json')
        for folder, record in ((HERE/'qualification_v3', qual), (HERE/'qualification_watch_v3', qw)):
            end, final = read(folder/'completion.json'), read(folder/'finalization.json')
            require(record['passed'] and end['passed'] and final['passed']
                and end['result_sha256'] == owner.PINS[folder/'result.json']
                and final['completion_sha256'] == owner.PINS[folder/'completion.json'], 'Qualified V3 seals')
        require(qual['completed'] and qual['hashes_unchanged'] and qual['sources_unchanged']
            and qual['retired_call_count'] == 8 and qual['full_model_calls'] == 7 and qual['tiny_model_calls'] == 8
            and not qual['training_targets_accessed'] and qw['whole_owned_tree_retired'] and qw['cleanup_passed'],
            'Target-free V3 qualification unchanged')
        for record in (old, old_watch, qual, qw):
            for group in ('source_bindings', 'input_bindings', 'artifacts'):
                for path, digest in record.get(group, {}).items():
                    pin(path, digest)
        diagnostic = read(HERE/'gradient_geometry_diagnostic_v1/result.json')
        de = read(HERE/'gradient_geometry_diagnostic_v1/completion.json')
        require(diagnostic['passed'] and diagnostic['information_premise'] and de['passed']
            and de['result_sha256'] == PINS[HERE/'gradient_geometry_diagnostic_v1/result.json']
            and diagnostic['source_sha256'] == PINS[HERE/'gradient_geometry_diagnostic.py']
            and diagnostic['original_failed_result_sha256'] == PINS[OLD/'result.json']
            and diagnostic['ranks'] == list(GRADIENT_RANKS) and diagnostic['direction_rules_evaluated'] == 1
            and diagnostic['model_calls'] == 0 and not diagnostic['target_files_read'], 'Sole saved-data premise')
        gradients = old['calls'][:20]
        for index, call in enumerate(gradients):
            require(call['call'] == index and call['rank'] == GRADIENT_RANKS[index]
                and call['kind'] == 'backward' and call['passed'] and call['completed'] and call['graph_retired'],
                'Complete historical proposal gradient')
            item = call['gradient']
            require(diagnostic['gradient_bindings'][item['path']] == item['sha256']
                == old['artifacts'][item['path']], 'Same20 seed arrays, not a selected subset')
            pin(item['path'], item['sha256'])
        m = module(HERE/'model.py', 'balance_frozen_receiver_model')
        for path, digest in m.authenticate().items():
            pin(path, digest)
        packet, proof = owner.helper_source()
        require(proof == old['helper_proof'], 'Exact original qualified helper packet, no AD repair')
        result['helper_proof'] = dict(proof, exclusively_forward_mode=True, synthetic_target_always_zero=True)
        p1, end = read(owner.P1/'training_v1/result.json'), read(owner.P1/'training_v1/completion.json')
        require(p1['passed'] and p1['completed'] and p1['hashes_unchanged'] and p1['sources_unchanged']
            and end['passed'] and end['completed'] and end['result_sha256'] == owner.PINS[owner.P1/'training_v1/result.json']
            and not p1['validation_targets_accessed'], 'Original TRAIN metadata provenance')
        indexed = {x['rank']: x for x in read(owner.INDEX)['target_sources']}
        paths, target_pins = {}, {}
        for rank in RANKS:
            wp = owner.STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            row = wp.with_name('row_manifest.json'); tp = owner.TRUTH/f'truthpost_rank{rank:04d}.npz'
            item = indexed[rank]
            require(rank in p1['train_ranks'] and item['weight_file'] == str(wp) and item['target_file'] == str(tp)
                and p1['target_bindings'][str(tp)] == item['target_sha256'], 'Exact four TRAIN labels/weights')
            pin(wp, p1['weight_bindings'][str(wp)]); pin(row, p1['weight_bindings'][str(row)])
            paths[rank], target_pins[rank] = wp, (tp, item['target_sha256'])
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed AP/BLAS/MPS environment')
        require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Complete source and metadata admission before numerical imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified numerical environment')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory()); require(recommended > 0, 'Positive MPS allowance')
        fraction = min(1., 16_000_000_000/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        result['mps_allocator'] = dict(configured=True, recommended_max_memory_bytes=recommended,
            configured_fraction=fraction, api='torch.mps.set_per_process_memory_fraction', low_watermark_ratio='0.1')
        m.runtime()
        def guard():
            torch.mps.synchronize()
            driver = int(torch.mps.driver_allocated_memory())
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            require(driver < 16_000_000_000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 12_000_000_000
                and time.monotonic()-began < 180, 'Internal balance time/host/driver gate')
            return dict(mps_driver_bytes=driver, host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        def save(name, value):
            path = OUT/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            digest = sha(path); result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest, raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                dtype=str(value.dtype), shape=list(value.shape))
        def document(name, value):
            path = OUT/(name+'.json'); write(path, value)
            digest = sha(path); result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest)
        def load(item):
            require(sha(item['path']) == item['sha256'], 'Saved array seal')
            value = np.load(item['path'], allow_pickle=False)
            require(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
                and hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256'], 'Saved array bytes/header')
            return value
        rows = []
        for call in gradients:
            g = load(call['gradient'])
            require(g.shape == (64, 64) and g.dtype == np.float32 and np.isfinite(g).all(), 'Finite complete seed gradient')
            rows.append(g.astype(np.float64).reshape(-1))
        G = np.stack(rows); K = G@G.T; trace = float(np.trace(K))
        require(math.isfinite(trace) and trace > 0, 'Nontrivial seed Gram')
        damping = 1e-3*trace/20
        alpha = np.linalg.solve(K+damping*np.eye(20), np.ones(20))
        direction64 = -(G.T@alpha)
        require(np.isfinite(direction64).all() and np.linalg.norm(direction64) > 0, 'One finite nonzero direction')
        direction64 *= 64/np.linalg.norm(direction64)
        direction = direction64.reshape(64, 64).astype(np.float32)
        dots = G@direction.astype(np.float64).reshape(-1)
        require(np.isfinite(dots).all() and (dots < 0).all(), 'The sole cast proposal retains20 negative recorded-vector dots')
        result['direction_seal'] = document('direction_sealed', dict(direction=save('direction32', direction),
            gradient_ranks=list(GRADIENT_RANKS), gradient_diagnostic_sha256=PINS[HERE/'gradient_geometry_diagnostic_v1/result.json'],
            damping=damping, seed_gram=save('seed_gram64', K), alpha=save('seed_alpha64', alpha),
            recorded_vector_dots=dots.tolist(), direction_count=1, derivatives_certified=False))
        env = dict(np=np, torch=torch, m=m, math=math, gc=gc, weakref=weakref, hashlib=hashlib,
            require=require, save=save, guard=guard, theta_np=np.zeros((64, 64), np.float32), direction_np=direction)
        exec(compile(packet, str(__file__)+':unchanged_forward_helpers', 'exec'), env)
        synthetic = np.zeros((16, 1024), np.float32)
        values, references = {}, {x['rank']: x for x in gradients[:4]}
        def forward(case, rank, theta):
            index = result['full_model_calls']
            env['theta_np'] = theta
            weights = np.load(paths[rank], allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32, 'Canonical full-card weights')
            answer = env['run_card'](rank, 'forward', 'mps', weights, synthetic, zero=False)
            require(answer['gradient'] is None and answer['tangent'] is None and answer['served_jvp'] is None,
                    'Ordinary forward only, no AD payload')
            record = {k: v for k, v in answer.items() if k not in ('raw', 'served', 'gradient', 'diagnostics')}
            record.update(call=index, rank=rank, case=case, kind='forward', graph_retired=True,
                scalar_is_synthetic_zero_target=True, raw=save(f'call_{index:02d}_raw', answer['raw']),
                served=save(f'call_{index:02d}_served', answer['served']),
                diagnostics=document(f'call_{index:02d}_diagnostics', answer['diagnostics']))
            document(f'call_{index:02d}_attempt', record)
            if case == 'zero':
                ref = references[rank]; rd = read(ref['diagnostics']['path'])
                require(all(record[k]['raw_sha256'] == ref[k]['raw_sha256'] for k in ('raw', 'served'))
                    and answer['frozen_hashes'] == ref['frozen_hashes']
                    and all(answer['diagnostics'][k] == rd[k] for k in ('variance_domain', 'selection_cells')),
                    'Zero exact historical current-COMP raw/served/frozen/domain/ordered selectors')
                record['historical_baseline_byte_exact'] = True
            values[(case, rank)] = answer['served'].astype(np.float64)
            record.update(passed=True, completed=True)
            result['outputs'].setdefault(case, {})[str(rank)] = dict(raw=record['raw'], served=record['served'],
                diagnostics=record['diagnostics'], call=index)
            result['calls'].append(record); result['full_model_calls'] += 1; result['retired_call_count'] += 1
            del answer, weights; gc.collect(); guard()
            print(json.dumps(dict(event='retired', call=index)), flush=True)
        h = 1/8
        operands = dict(zero=np.zeros((64, 64), np.float32),
            plus=(h*direction.astype(np.float64)).astype(np.float32),
            minus=(-h*direction.astype(np.float64)).astype(np.float32))
        result['probe_parameters'] = {key: save('theta_'+key, value) for key, value in operands.items()}
        for case, theta in operands.items():
            for rank in RANKS:
                forward(case, rank, theta)
        require(result['full_model_calls'] == 12 and not result['training_targets_accessed']
            and not result['target_bindings'], 'All12 points sealed before target-file access')
        result['pretruth_seal'] = document('first12_outputs_sealed', dict(calls=result['calls'], outputs=result['outputs'],
            source_bindings=result['source_bindings'], direction_seal=result['direction_seal'],
            probe_parameters=result['probe_parameters'], training_targets_accessed=False, full_model_calls=12))
        result['first12_outputs_sealed_sha256'] = result['pretruth_seal']['sha256']
        require(sha(result['pretruth_seal']['path']) == result['first12_outputs_sealed_sha256'], 'Pretruth seal finalized')
        targets, rows = {}, []
        for rank in RANKS:
            path, digest = target_pins[rank]; pin(path, digest, 'target_bindings')
            result['training_targets_accessed'] = True
            with np.load(path, allow_pickle=False) as src:
                target = src['mc_post'].copy()
            require(target.shape == (16, 1024) and target.dtype.kind == 'f' and np.isfinite(target).all(), 'Original finite1e9 target')
            targets[rank] = target.astype(np.float64)
            y0, yp, ym = (values[(key, rank)][-1] for key in ('zero', 'plus', 'minus'))
            j, e = (yp-ym)/(2*h), y0-targets[rank][-1]
            even, odd = (yp+ym)/2-y0, (yp-ym)/2
            rows.append(dict(rank=rank, target_original_dtype=str(target.dtype), a=float(np.mean(e*e)),
                b=float(np.mean(e*j)), c=float(np.mean(j*j)),
                plus_raw=float(np.mean((yp-targets[rank][-1])**2)), minus_raw=float(np.mean((ym-targets[rank][-1])**2)),
                even_defect_rms=float(np.sqrt(np.mean(even*even))), odd_change_rms=float(np.sqrt(np.mean(odd*odd))),
                even_to_odd_norm_ratio=float(np.linalg.norm(even)/max(np.linalg.norm(odd), 1e-30))))
        a, b, c = (math.fsum(row[k] for row in rows)/4 for k in ('a', 'b', 'c'))
        require(all(math.isfinite(x) for x in (a, b, c)) and a > 0 and c > 0, 'Defined single fixed secant fit')
        step = min(1., max(0., -b/c)); predicted = a+2*step*b+step*step*c
        theta_star = (step*direction.astype(np.float64)).astype(np.float32)
        result['step_seal'] = document('single_step_sealed', dict(step=step, theta=save('single_theta32', theta_star),
            h=h, a=a, b=b, c=c, secant_predicted_raw=predicted, secant_predicted_ratio=predicted/a,
            rows=rows, first12_outputs_sealed_sha256=result['first12_outputs_sealed_sha256'],
            parameter_proposals=1, optimizer_updates=0, derivative_claim=False))
        result['parameter_proposals'] = 1
        theta_star = load(read(result['step_seal']['path'])['theta'])
        for rank in RANKS:
            forward('step', rank, theta_star)
        result['fitted_points_evaluated'] = 1
        for row in rows:
            rank = row['rank']; actual = values[('step', rank)][-1]
            j = (values[('plus', rank)][-1]-values[('minus', rank)][-1])/(2*h)
            defect = actual-(values[('zero', rank)][-1]+step*j)
            row.update(actual_raw=float(np.mean((actual-targets[rank][-1])**2)),
                endpoint_secant_defect_rms=float(np.sqrt(np.mean(defect*defect))))
            row['actual_raw_ratio'] = row['actual_raw']/row['a']
        actual = math.fsum(row['actual_raw'] for row in rows)/4; ratio = actual/a
        result['decision'] = dict(baseline_raw=a, actual_raw=actual, actual_raw_ratio=ratio,
            step=step, secant_predicted_ratio=predicted/a, standalone_raw_lead=bool(ratio <= .66),
            useful_raw_component=bool(ratio <= .90), raw_wins=sum(row['actual_raw_ratio'] < 1 for row in rows),
            planning_cost_factor=(468_313_646_101.175+4_000_000_000)/468_313_646_101.175,
            actual_native_cost_measured=False, derivative_or_native_score_claim=False)
        result['raw_rows'] = rows; result['information_promising'] = bool(ratio <= .90)
        require(result['full_model_calls'] == result['retired_call_count'] == 16
            and result['gradient_calls'] == result['jvp_calls'] == 0, 'Exactly16 forward-only calls')
        for group in ('source_bindings', 'input_bindings', 'target_bindings', 'artifacts'):
            require(all(sha(path) == digest for path, digest in result[group].items()), 'Unchanged admitted bytes: '+group)
        guard(); result.update(passed=True, completed=True, hashes_unchanged=True, sources_unchanged=True,
            all_graphs_retired=True, execution_integrity_passed=True, no_checkpoint_selection=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result); digest = sha(OUT/'result.json')
    write(OUT/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    end_digest = sha(OUT/'completion.json'); elapsed = time.monotonic()-began
    ok = result['passed'] and elapsed < 180
    write(OUT/'finalization.json', dict(passed=ok, completed=ok, result_sha256=digest,
        completion_sha256=end_digest, elapsed_s=elapsed))
    signal.alarm(0)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
