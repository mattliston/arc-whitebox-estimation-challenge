"""One fixed TRAIN direction/step, using the qualified V3 card-call owner."""
import argparse
import ast
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
import textwrap
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
OUT = HERE/'information_v1'
RANKS = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
MODEL_SHA = 'af5ab0386bde2561543c631ace208f3649836d6581a04a69967705aec008c907'
PINS = {
    HERE/'model.py': MODEL_SHA,
    HERE/'qualification_v3.py': 'ce5dfc095fe7fb16f3538ae200a046ca19e8b311324c2035e21914722fe043cf',
    HERE/'qualification_v3/result.json': '1213c0901ef6c0e9224034bbd099c7f727742c7e2336b9cda29adfbd4811496a',
    HERE/'qualification_v3/completion.json': 'bed253bd64803b7a625108caa40ec27a12f93de81a4d7963947af76503b92733',
    HERE/'qualification_v3/finalization.json': '50462f5f3a90c10379348c3a0aa60c5556d335f19d7bd3ec26c8e64e061087ad',
    HERE/'qualification_watch_v3/result.json': '5d52380663114eff21f51222cb4927ead00344278358735d7df634ea3da9e7e0',
    HERE/'qualification_watch_v3/completion.json': '411761a3bdd0867cad799e92af7db536dc4a3d8e190d13759941ea8cbfee62c4',
    HERE/'qualification_watch_v3/finalization.json': '2721db7e8063dbce90120872a43c3a462ce63a9740cef6a31dc2383e8077c6bb',
    P1/'training_v1/result.json': 'aaaef3d5f5a3c60583af779895b8db4f769276c64cb0cc032e93755169dbab91',
    P1/'training_v1/completion.json': '6964011b2262a70e56fafd74b040432065513706c3fde5586341a09b85fd0253',
    P1/'worker.py': 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c',
    INDEX: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
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


def helper_source():
    """Reuse only four qualified functions, inverse-proving the new loss."""
    require(sha(HERE/'qualification_v3.py') == PINS[HERE/'qualification_v3.py'], 'Qualified helper source')
    owner = module(HERE/'qualification_v3.py', 'information_v3_source_owner')
    source, parent_proof = owner.derived_source()
    lines, parts = source.splitlines(keepends=True), {}
    for name in ('array', 'frozen_hash', 'diagnostics', 'run_card'):
        found = [x for x in ast.walk(ast.parse(source)) if isinstance(x, ast.FunctionDef) and x.name == name]
        require(len(found) == 1, 'Unique qualified function '+name)
        node = found[0]
        parts[name] = textwrap.dedent(''.join(lines[node.lineno-1:node.end_lineno]))
    original = parts['run_card']
    changes = (
        ("        def scalar_value(raw):\n"
         "            if card.depth == 16:\n"
         "                return original_scalar(torch, m._CO, raw, cot, card.dtype)\n"
         "            y = m._CO.served(raw).to(card.dtype)\n"
         "            q = torch.as_tensor(cot, dtype=card.dtype, device=y.device)\n"
         "            return (y[-1]*q[-1]).sum()/math.sqrt(card.n) + .05*(y[:-1]*q[:-1]).sum()/math.sqrt((card.depth-1)*card.n)\n",
         "        def scalar_value(raw):\n"
         "            y = m._CO.served(raw)[-1]\n"
         "            target = torch.as_tensor(cot[-1], dtype=y.dtype, device=y.device)\n"
         "            return (y-target).square().mean()/2e-8\n"),
        ("                    j64, q64 = vector.astype(np.float64), np.asarray(cot, dtype=np.float64)\n"
         "                    tangent = float(np.sum(j64[-1]*q64[-1])/math.sqrt(card.n)\n"
         "                              + .05*np.sum(j64[:-1]*q64[:-1])/math.sqrt((card.depth-1)*card.n))\n",
         "                    j64, y64 = vector.astype(np.float64), array(yp).astype(np.float64)\n"
         "                    q64 = np.asarray(cot, dtype=np.float32).astype(np.float64)\n"
         "                    tangent = float(2*np.mean((y64[-1]-q64[-1])*j64[-1])/2e-8)\n"),
        ("                    del dual_served, yp, yt, vector, j64, q64, raw_primal\n",
         "                    del dual_served, yp, yt, vector, j64, y64, q64, raw_primal\n"),
    )
    changed = original
    for old, new in changes:
        require(changed.count(old) == 1, 'Unique prescribed information loss/readout span')
        changed = changed.replace(old, new)
    inverse = changed
    for old, new in reversed(changes):
        require(inverse.count(new) == 1, 'Unique information inverse span')
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Exact inverse of qualified run_card outside loss/readout')
    parts['run_card'] = changed
    packet = '\n\n'.join(parts.values())
    return packet, dict(parent=parent_proof, inverse_source_byte_exact=True,
        original_run_card_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_run_card_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        extracted_packet_sha256=hashlib.sha256(packet.encode()).hexdigest(),
        model_checkpoint_domain_frozen_retirement_unchanged=True,
        loss='FP32-target final served MSE/2e-8 only', actual_served_vector_jvp=True)


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
        raise TimeoutError('Fixed600s whole information deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    require(not OUT.exists(), 'Exclusive information experiment, no resume')
    OUT.mkdir()
    result = dict(schema='current_comp_receiver_birth_information_v1', passed=False, completed=False,
        source_bindings={}, input_bindings={}, target_bindings={}, artifacts={}, calls=[],
        ranks=list(RANKS), gradient_calls=0, jvp_calls=0, nonlinear_calls=0, retired_call_count=0,
        training_targets_accessed=False, validation_targets_accessed=False,
        optimizer_updates=0, frozen_parent_parameter_updates=0, parameter_proposals=0, evaluated_interventions=0,
        native_gradient_qualified=False, native_score_measured=False, no_native_gradient_claim=True,
        maximum_sampled_mps_driver_bytes=0, information_promising=False,
        decision=dict(predicted_promising=False))
    try:
        require(args.model_sha256 == MODEL_SHA and len(RANKS) == 20, 'Fixed model and20 TRAIN')
        bindings = dict(PINS)
        bindings.update({Path(__file__).resolve(): args.source_sha256, HERE/'model.py': args.model_sha256,
            HERE/'INFORMATION_PROTOCOL.md': args.protocol_sha256, HERE/'watch_information.py': args.watcher_sha256})
        def pin(path, expected, group=None):
            path = Path(path).resolve()
            require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected, 'Pinned bytes: '+str(path))
            if group is None:
                group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
            require('truthpost' not in str(path) or group == 'target_bindings', 'TRAIN target bindings stay separate')
            result[group][str(path)] = expected
        for path, digest in bindings.items():
            pin(path, digest)
        qual = read(HERE/'qualification_v3/result.json')
        qw = read(HERE/'qualification_watch_v3/result.json')
        for folder, record in ((HERE/'qualification_v3', qual), (HERE/'qualification_watch_v3', qw)):
            end, final = read(folder/'completion.json'), read(folder/'finalization.json')
            require(record['passed'] and end['passed'] and final['passed']
                and end['result_sha256'] == PINS[folder/'result.json']
                and final['completion_sha256'] == PINS[folder/'completion.json'], 'Passed independent qualification seals')
        require(qual['completed'] and qual['hashes_unchanged'] and qual['sources_unchanged']
            and qual['retired_call_count'] == 8 and qual['full_model_calls'] == 7 and qual['tiny_model_calls'] == 8
            and not qual['training_targets_accessed'] and not qual['validation_targets_accessed']
            and qw['whole_owned_tree_retired'] and qw['cleanup_passed'] and qw['retired_calls'] == 8
            and qw['child_result_sha256'] == PINS[HERE/'qualification_v3/result.json']
            and qw['child_completion_sha256'] == PINS[HERE/'qualification_v3/completion.json']
            and qw['child_finalization_sha256'] == PINS[HERE/'qualification_v3/finalization.json'], 'Exact qualified seven-call frame')
        for record in (qual, qw):
            for group in ('source_bindings', 'input_bindings', 'artifacts'):
                for path, digest in record.get(group, {}).items():
                    pin(path, digest)
        m = module(HERE/'model.py', 'information_frozen_receiver_model')
        for path, digest in m.authenticate().items():
            pin(path, digest)
        packet, proof = helper_source(); result['helper_proof'] = proof
        p1, end = read(P1/'training_v1/result.json'), read(P1/'training_v1/completion.json')
        require(p1['passed'] and p1['completed'] and p1['hashes_unchanged'] and p1['sources_unchanged']
            and end['passed'] and end['completed'] and end['result_sha256'] == PINS[P1/'training_v1/result.json']
            and not p1['validation_targets_accessed'] and all(r in p1['train_ranks'] for r in RANKS), 'Completed TRAIN-only P1 target provenance')
        indexed = {x['rank']: x for x in read(INDEX)['target_sources']}
        paths, target_pins = {}, {}
        for rank in RANKS:
            wp = STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            row = wp.with_name('row_manifest.json')
            tp = TRUTH/f'truthpost_rank{rank:04d}.npz'
            item = indexed[rank]
            require(item['weight_file'] == str(wp) and item['target_file'] == str(tp)
                    and p1['target_bindings'][str(tp)] == item['target_sha256'], 'Exact TRAIN labels/weights in both original receipts')
            pin(row, p1['weight_bindings'][str(row)]); pin(wp, p1['weight_bindings'][str(wp)])
            paths[rank], target_pins[rank] = wp, (tp, item['target_sha256'])
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed AP/nice/BLAS/MPS environment')
        require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Source/helper/qualification/target-metadata admission completed before numerical imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified numerical environment')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory())
        require(recommended > 0, 'Positive MPS allocator allowance')
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
                and time.monotonic()-began < 600, 'Internal information time/host/driver gate')
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
            require(sha(item['path']) == item['sha256'], 'Saved same-run array seal')
            value = np.load(item['path'], allow_pickle=False)
            require(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
                and hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256'], 'Saved array bytes/header')
            return value
        env = dict(np=np, torch=torch, m=m, math=math, gc=gc, weakref=weakref, hashlib=hashlib,
            require=require, save=save, guard=guard, theta_np=np.zeros((64, 64), np.float32),
            direction_np=np.zeros((64, 64), np.float32))
        exec(compile(packet, str(__file__)+':qualified_functions', 'exec'), env)
        run_card = env['run_card']
        targets, baseline, gradients = {}, {}, {}
        def attempt(index, rank, kind, zero):
            weights = np.load(paths[rank], allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32, 'Original full-card weights')
            answer = run_card(rank, kind, 'mps', weights, targets[rank], zero=zero)
            record = {k: v for k, v in answer.items() if k not in ('raw', 'served', 'gradient', 'diagnostics')}
            record.update(call=index, rank=rank, kind=kind, graph_retired=True,
                raw=save(f'call_{index:02d}_raw', answer['raw']), served=save(f'call_{index:02d}_served', answer['served']),
                diagnostics=document(f'call_{index:02d}_diagnostics', answer['diagnostics']))
            if answer['gradient'] is not None:
                record['gradient'] = save(f'call_{index:02d}_gradient', answer['gradient'])
            record['raw_final_mse'] = float(np.mean((answer['served'][-1].astype(np.float64)-targets[rank][-1])**2))
            document(f'call_{index:02d}_attempt', record)
            del weights
            return answer, record
        def retired(index, record):
            record.update(passed=True, completed=True)
            result['calls'].append(record); result['retired_call_count'] += 1
            gc.collect(); guard()
            print(json.dumps(dict(event='retired', call=index)), flush=True)

        for index, rank in enumerate(RANKS):
            tp, digest = target_pins[rank]
            pin(tp, digest, 'target_bindings')
            result['training_targets_accessed'] = True
            with np.load(tp, allow_pickle=False) as src:
                values = src['mc_post'].copy()
            require(values.shape == (16, 1024) and values.dtype.kind == 'f' and np.isfinite(values).all(), 'Original finite1e9 mc_post frame')
            targets[rank] = values.astype(np.float64)
            answer, record = attempt(index, rank, 'backward', True)
            record['target_original_dtype'] = str(values.dtype)
            if rank == 35:
                q = qual['calls'][0]
                require(q['rank'] == 35 and q['kind'] == 'zero'
                    and all(record[k]['raw_sha256'] == q[k]['raw_sha256'] for k in ('raw', 'served'))
                    and all(answer['diagnostics'][k] == q['diagnostics'][k] for k in ('variance_domain', 'selection_cells')),
                    'Exact qualified current-COMP zero35 compatibility')
                record['qualified_zero35_byte_exact'] = True
            baseline[rank] = answer; gradients[rank] = answer['gradient'].astype(np.float64)
            result['gradient_calls'] += 1
            del values; retired(index, record)
        pooled = np.zeros((64, 64), np.float64)
        for rank in RANKS:
            pooled += gradients[rank]
        pooled /= 20
        norm = float(np.linalg.norm(pooled))
        require(math.isfinite(norm) and norm > 0, 'One nonzero finite pooled direction; no alternative')
        direction = (-64*pooled/norm).astype(np.float32)
        env['direction_np'] = direction
        result['direction_seal'] = document('direction_sealed', dict(ranks=list(RANKS), gradient_calls=20,
            pooled_gradient=save('pooled_gradient64', pooled), norm=norm, direction=save('direction32', direction),
            direction_count=1, optimizer_constructed=False, qualification_result_sha256=PINS[HERE/'qualification_v3/result.json']))
        stats = []
        for offset, rank in enumerate(RANKS):
            index = 20+offset
            answer, record = attempt(index, rank, 'jvp', True)
            old = baseline[rank]
            require(all(answer[k].tobytes() == old[k].tobytes() for k in ('raw', 'served'))
                and answer['scalar_raw_sha256'] == old['scalar_raw_sha256']
                and all(answer['diagnostics'][k] == old['diagnostics'][k] for k in ('variance_domain', 'selection_cells')),
                'Same zero raw/served/loss/domain/ordered selectors on actual JVP')
            reverse = float(np.sum(gradients[rank]*direction.astype(np.float64)))
            tangent = answer['tangent']; error = abs(reverse-tangent)
            require(math.isfinite(tangent) and error <= 1e-10+1e-4*max(abs(reverse), abs(tangent)),
                'FP32-target reverse-dot/actual served JVP identity')
            j = load(answer['served_jvp'])[-1].astype(np.float64)
            residual = old['served'][-1].astype(np.float64)-targets[rank][-1]
            row = dict(rank=rank, a=float(np.mean(residual*residual)), b=float(np.mean(residual*j)),
                c=float(np.mean(j*j)), reverse_dot=reverse, forward_readout=tangent, identity_abs_error=error)
            record['quadratic'] = row; stats.append(row); result['jvp_calls'] += 1
            del answer, j, residual; retired(index, record)
        a, b, c = (math.fsum(row[k] for row in stats)/20 for k in ('a', 'b', 'c'))
        require(all(math.isfinite(v) for v in (a, b, c)) and a > 0 and c > 0, 'Finite nontrivial fixed quadratic')
        step = min(1., max(0., -b/c))
        predicted = a+2*step*b+step*step*c
        require(predicted >= 0 and math.isfinite(predicted), 'Nonnegative finite predicted raw')
        ratio = predicted/a
        theta_star = (step*direction.astype(np.float64)).astype(np.float32)
        result['decision'] = dict(predicted_promising=bool(ratio <= .90), baseline_raw=a,
            b=b, c=c, step=step, predicted_raw=predicted, predicted_raw_ratio=ratio,
            standalone_raw_lead=False, useful_raw_component=False, single_direction_not_capacity_bound=True)
        result['step_seal'] = document('single_step_sealed', dict(result['decision'],
            theta=save('single_theta32', theta_star), direction_seal=result['direction_seal'],
            gradient_calls=20, jvp_calls=20, nonlinear_calls=0, optimizer_updates=0,
            frozen_parent_parameter_updates=0, parameter_proposals=1))
        result['parameter_proposals'] = 1
        if result['decision']['predicted_promising']:
            env['theta_np'] = load(read(result['step_seal']['path'])['theta'])
            result['evaluated_interventions'] = 1
            actual = []
            for offset, rank in enumerate(RANKS):
                index = 40+offset
                answer, record = attempt(index, rank, 'forward', False)
                actual.append(record['raw_final_mse']); result['nonlinear_calls'] += 1
                del answer; retired(index, record)
            actual_mean = math.fsum(actual)/20; actual_ratio = actual_mean/a
            planned_factor = (468_313_646_101.175+4_000_000_000)/468_313_646_101.175
            result['decision'].update(actual_nonlinear_raw=actual_mean, actual_nonlinear_raw_ratio=actual_ratio,
                standalone_raw_lead=bool(actual_ratio <= .66), useful_raw_component=bool(actual_ratio <= .90),
                fixed_geometry_planning_adjusted_ratio=actual_ratio*planned_factor,
                planning_cost_factor=planned_factor, actual_native_cost_measured=False)
            result['information_promising'] = bool(actual_ratio <= .90)
        expected = 60 if result['decision']['predicted_promising'] else 40
        require(result['gradient_calls'] == result['jvp_calls'] == 20 and result['retired_call_count'] == expected
            and result['nonlinear_calls'] == expected-40, 'Exact conditional model-call roster')
        for group in ('source_bindings', 'input_bindings', 'target_bindings', 'artifacts'):
            require(all(sha(p) == h for p, h in result[group].items()), 'All fixed bytes unchanged: '+group)
        guard()
        result.update(passed=True, completed=True, hashes_unchanged=True, sources_unchanged=True,
            all_graphs_retired=True, execution_integrity_passed=True, no_checkpoint_selection=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result); digest = sha(OUT/'result.json')
    write(OUT/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    end_digest = sha(OUT/'completion.json'); elapsed = time.monotonic()-began
    ok = result['passed'] and elapsed < 600
    write(OUT/'finalization.json', dict(passed=ok, completed=ok, result_sha256=digest,
        completion_sha256=end_digest, elapsed_s=elapsed))
    signal.alarm(0)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
