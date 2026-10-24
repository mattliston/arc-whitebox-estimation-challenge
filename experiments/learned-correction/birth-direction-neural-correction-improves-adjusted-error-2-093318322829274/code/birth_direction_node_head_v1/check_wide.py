"""Fixed node-head wide fidelity; inert without --stage, no fitting.

CPU/MPS stages require their own tiny-stage seal and read no targets. The
judge requires the full tiny judge, three actual native roles, and both
shadow stages. All predictions, complete bills and gradients are sealed
before only the existing TRAIN35/100 truths are opened for MSE fidelity.
The fixed cotangent is float32(sin(arange(1024)*.37))/1024; only the fixed
nonzero A is differentiated. No parameter update, head reset or control.

Shadow stages have a600s watchdog, RSS<12GB, MPS driver<16GB. The judge
uses OFFPY,300s/RSS<2GB. Native8GB/120s gates are preserved byte-for-byte
inside the qualified verifier, with only candidate-map/init-map edits.
"""
import argparse
import ast
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
RANKS, CASES = (35, 100), ('zero', 'nonzero')
SCHEMA = 'birth_direction_node_head_wide_v1'
TINY_SHA = '06e136818954e039626d49585770ad972e4ef1d4061e1d16741d556c1d3c485c'
NATIVE_SHA = '70506904442e58f0931aa254a41cc7be25522b73363db2118eefec6761cdf2e0'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
TARGET_MANIFEST_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
MEAN_LIMIT, MSE_LIMIT, GRAD_LIMIT, GRAD_FLOOR = 1e-5, .01, .05, 1e-10


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def binding(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def pinning(result):
    def pin(path, expected=None):
        item = binding(path)
        require(expected is None or item['sha256'] == expected, 'Input SHA: ' + item['path'])
        old = result['source_bindings'].get(item['path'])
        require(old is None or old == item, 'Conflicting/changed input: ' + item['path'])
        result['source_bindings'][item['path']] = item
        return item['sha256']
    return pin


def read(path):
    return json.loads(Path(path).read_text())


def bind_items(items, pin):
    for key, item in items.items():
        if isinstance(item, str):
            pin(key, item)
        else:
            pin(item['path'], item['sha256'])
            actual = binding(item['path'])
            require(all(actual[k] == v for k, v in item.items() if k in actual), 'Binding metadata changed')


def sealed(folder, digest, pin, completed=True):
    require(isinstance(digest, str) and len(digest) == 64, 'Explicit completed-result SHA required')
    pin(folder / 'result.json', digest)
    pin(folder / 'completion.json')
    record, completion = read(folder / 'result.json'), read(folder / 'completion.json')
    require(record['passed'] and completion['passed'] and completion['result_sha256'] == digest,
            'Successful completion seal')
    if completed:
        require(record['completed'] and completion['completed'], 'Complete fixed stage')
    return record


def tiny_stage(stage, digest, tiny, result):
    pin = pinning(result)
    folder = HERE / ('stage_' + stage)
    record = sealed(folder, digest, pin)
    require(record['schema'] == 'birth_direction_node_head_tiny_v1' and record['stage'] == stage
            and record['sources_unchanged'] and record['no_fit'] and record['no_network_or_target_access'],
            'Fixed, target-free tiny qualification')
    require(record['source_bindings'][str(HERE / 'check_tiny.py')]['sha256'] == TINY_SHA,
            'Actual tiny checker source')
    bind_items(record['source_bindings'], pin)
    bind_items(record['artifacts'], pin)
    require(record['prototype_seals'] == result['prototype_seals'], 'Same fixed prototypes')
    require(record['elapsed_s'] < 60 and record['peak_rss_bytes'] < (2e9 if stage in ('cpu', 'mps') else 1e9)
            and record['mps_driver_bytes'] < 2e9, 'Tiny resource gates')
    if stage == 'judge':
        require(record['small_fidelity_passed'] and record['all_stage_artifacts_verified_before_read']
                and record['wide_or_training_admitted_by_this_receipt'] is False
                and not record['artifacts'] and set(record['stage_seals']) == {'native', 'cpu', 'mps'},
                'Full tiny judge, not an isolated stage')
        for name, value in record['stage_seals'].items():
            tiny_stage(name, value, tiny, result)
    else:
        require(set(record['artifacts']) == {'arrays.npz'}
                and Path(record['artifacts']['arrays.npz']['path']) == folder / 'arrays.npz',
                'Tiny exact artifact ownership')
    return record


def resources(stage, start, torch=None):
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if sys.platform == 'darwin' else value * 1024)
    if stage == 'mps' and torch is None:
        torch = sys.modules.get('torch')
    driver = (int(torch.mps.driver_allocated_memory())
              if stage == 'mps' and torch and torch.backends.mps.is_available() else 0)
    return dict(elapsed_s=time.perf_counter() - start, peak_rss_bytes=rss, mps_driver_bytes=driver)


def guard(stage, start, torch=None):
    value = resources(stage, start, torch)
    require(value['elapsed_s'] < (300 if stage == 'judge' else 600), 'Wide stage wall cap')
    require(value['peak_rss_bytes'] < (2e9 if stage == 'judge' else 12e9), 'Wide stage RSS cap')
    require(value['mps_driver_bytes'] < 16e9, 'Wide MPS driver cap')
    return value


def weights(result):
    pin = pinning(result)
    pin(STORE / 'manifest.json', '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50')
    rows = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    values = {}
    for rank in RANKS:
        matches = [x for x in rows if x['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical weight row')
        path = STORE / matches[0]['path']
        pin(path, matches[0]['sha256'])
        row = read(path)
        require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Weight frame')
        wp = path.parent / 'weights.npy'
        pin(wp, row['files']['weights.npy']['sha256'])
        values[str(rank)] = dict(row=binding(path), weights=binding(wp))
    result['weight_bindings'] = values
    return values


def diagnostics(value, replay=False):
    require(value['frozen_P'] and value['frozen_cstar']
            and value['allocated_trainable_parameters'] == 256, 'Only A trainable')
    domain = value['variance_domain']
    require(set(map(int, domain)) == set(range(16)), 'All16 physical prevariance checks')
    require(all(x['finite'] and math.isfinite(x['minimum']) and x['minimum'] >= 1e-10
                and x['below_floor_count'] == 0 for x in domain.values()), 'No variance-floor event')
    expected = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    require(set(value['selection_cells']) == expected, 'All47 checkpoint selection cells')
    for ids in value['selection_cells'].values():
        require(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < 1024 for i in ids),
                'Physical selection IDs')
    calls = value['node_head_calls']
    require(set(map(int, calls)) == set(range(1, 15))
            and all(x >= (2 if replay else 1) for x in calls.values()), 'All14 node heads/replays')


def save_array(np, folder, name, array, result):
    path = folder / name
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    result['artifacts'][name] = binding(path)
    return str(path)


def acquire(np, torch, model, build, stage, folder, result, start, tiny_record, tiny):
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
            'No AP threadpoolctl/FNP import')
    require(result['environment']['versions'] == tiny_record['environment']['versions'],
            'Same qualified tiny CPU/MPS libraries')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(9309093)
    require(stage != 'mps' or torch.backends.mps.is_available(), 'No MPS fallback')
    pin, skipped = pinning(result), []
    for imported in tuple(sys.modules.values()):
        path = getattr(imported, '__file__', None)
        if not path or Path(path).suffix != '.py':
            continue
        path = Path(path)
        if path.is_file() and path.resolve().is_relative_to(NEW):
            pin(path)
        elif not path.is_file():
            skipped.append(str(path))
    result['skipped_nonfile_module_names'] = sorted(set(skipped))
    tiny.write(folder / 'acquisition_bindings.json', result['source_bindings'])
    result['acquisition_bindings_sha256'] = sha(folder / 'acquisition_bindings.json')
    result['entries'] = []
    for rank in RANKS:
        for case in CASES:
            matrix = np.asarray(build.prototype_matrix(case), dtype=np.float32)
            A = torch.tensor(matrix, dtype=torch.float32, device=stage, requires_grad=(case == 'nonzero'))
            card = model.NodeHeadCard(rank, A=A, device=stage, audit_selections=True)
            # Offline checks only: returned state tensors are never changed.
            original_step = card.step
            finite_visits = {}
            def checked_step(layer, state, control, payload):
                returned = original_step(layer, state, control, payload)
                require(all(bool(torch.isfinite(x.detach()).all().cpu()) for x in returned
                            if isinstance(x, torch.Tensor) and x.is_floating_point()), 'Finite full carried state')
                finite_visits[layer] = finite_visits.get(layer, 0) + 1
                guard(stage, start, torch)
                return returned
            card.step = checked_step
            with torch.no_grad():
                raw, trace = card.run(capture=True)
                prediction = raw.clamp(min=0).cpu().float().numpy().copy()
            require(prediction.shape == (16, 1024) and prediction.dtype == np.float32
                    and np.isfinite(prediction).all() and (prediction >= 0).all(), 'Finite served means')
            forward = card.diagnostics()
            diagnostics(forward)
            require(set(finite_visits) == set(range(16)) and len(trace) == 16
                    and all(math.isfinite(x['radial']) for x in trace), 'Every forward state checked')
            entry = dict(rank=rank, case=case, A_bytes_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                output=save_array(np, folder, f'{rank}_{case}_prediction.npy', prediction, result),
                output_bytes_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                forward=forward, trace=trace, all_forward_states_finite=True)
            result['entries'].append(entry)
            if case == 'nonzero':
                differentiated, _ = card.run(training=True)
                cot_numpy = np.sin(np.arange(1024, dtype=np.float64) * .37).astype(np.float32) / 1024
                cot = torch.tensor(cot_numpy, device=stage, dtype=torch.float32)
                loss = (differentiated[-1].clamp(min=0) * cot).sum()
                loss.backward()
                require(A.grad is not None, 'A gradient exists')
                gradient = A.grad.detach().cpu().float().numpy().copy()
                norm = float(np.linalg.norm(gradient.astype(np.float64)))
                require(gradient.shape == (64, 4) and np.isfinite(gradient).all()
                        and math.isfinite(norm) and norm > GRAD_FLOOR, 'Non-negligible finite A gradient')
                gd = card.diagnostics()
                diagnostics(gd, replay=True)
                require(gd['selection_cells'] == forward['selection_cells'], 'Checkpoint/forward identical selections')
                grad_prediction = differentiated.detach().clamp(min=0).cpu().float().numpy()
                maximum = float(np.max(np.abs(grad_prediction.astype(np.float64)-prediction.astype(np.float64))))
                require(maximum <= MEAN_LIMIT and np.isfinite(grad_prediction).all(), 'Unchanged checkpoint forward')
                require(all(not x.requires_grad and x.grad is None for x in card.P0.values()), 'P stays frozen')
                entry.update(gradient=save_array(np, folder, f'{rank}_nonzero_gradient.npy', gradient, result),
                    gradient_norm=norm, differentiated=gd, checkpoint_max_mean_difference=maximum,
                    cot_bytes_sha256=hashlib.sha256(cot_numpy.tobytes()).hexdigest(),
                    fixed_cot_loss=float(loss.detach().cpu()))
                del differentiated, cot, loss, gradient, grad_prediction
            require(np.asarray(A.detach().cpu()).tobytes() == matrix.tobytes(), 'No A update or fit')
            entry.update(guard(stage, start, torch))
            tiny.write(folder / f'{rank}_{case}.json', entry)
            result['artifacts'][f'{rank}_{case}.json'] = binding(folder / f'{rank}_{case}.json')
            print(json.dumps(dict(stage=stage, rank=rank, case=case, **guard(stage, start, torch))), flush=True)
            del card, original_step, checked_step, raw, A, prediction
            gc.collect()
            if stage == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules, 'No late AP forbidden imports')
    result['acquisition_complete'] = True
    result['shadow_is_not_participant_bill'] = True


def native_verifier(native, result):
    path = BD / 'qualify_native_v2.py'
    pinning(result)(path, VERIFIER_SHA)
    text = path.read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == 'verify_native')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    changes = (
        ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
         "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)",
         '    expected_sha = NODE_CANDIDATES.get(source)'),
        ('    expected_init = 1383913666 + (0 if source == BANK else 120)',
         '    expected_init = NODE_INITIALIZATION[source]'),
    )
    source = original
    for old, new in changes:
        require(source.count(old) == 1, 'Exact native-verifier edit')
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(changes):
        require(inverse.count(new) == 1, 'Unique inverse verifier edit')
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Native five-artifact/resource/lifecycle verifier unchanged')
    base = module(path, 'bdnh_wide_preserved_native_verifier')
    candidates = {p.resolve(): digest for p, digest in native.CANDIDATES.values()}
    initialization = {p.resolve(): 1383913786 + (0 if role == 'parent' else 512)
                      for role, (p, _) in native.CANDIDATES.items()}
    env = dict(vars(base), NODE_CANDIDATES=candidates, NODE_INITIALIZATION=initialization)
    exec(compile(source, str(__file__) + ':native_verifier', 'exec'), env)
    result['native_verifier_inverse_byte_exact'] = True
    return env['verify_native']


def shadow_record(stage, digest, tiny_judge, result):
    pin = pinning(result)
    folder = HERE / f'wide_{stage}_v1'
    record = sealed(folder, digest, pin)
    require(record['schema'] == SCHEMA and record['stage'] == stage and record['sources_unchanged']
            and record['no_fit'] and record['frozen_P'] and record['frozen_cstar']
            and record['truth_accessed'] is False and record['acquisition_complete']
            and record['tiny_stage_sha256'] == tiny_judge['stage_seals'][stage], 'Fixed shadow-stage lineage')
    require(record['source_bindings'][str(Path(__file__).resolve())]['sha256'] == sha(__file__)
            and record['prototype_seals'] == result['prototype_seals']
            and record['weight_bindings'] == result['weight_bindings'], 'Identical source/weights/prototypes')
    bind_items(record['source_bindings'], pin)
    bind_items(record['artifacts'], pin)
    require([(x['rank'], x['case']) for x in record['entries']] == [(r, c) for r in RANKS for c in CASES],
            'Exactly four fixed shadow means')
    expected = {f'{r}_{c}_prediction.npy' for r in RANKS for c in CASES}
    expected |= {f'{r}_{c}.json' for r in RANKS for c in CASES} | {f'{r}_nonzero_gradient.npy' for r in RANKS}
    require(set(record['artifacts']) == expected, 'Complete exact shadow artifacts')
    require(all(Path(item['path']) == folder / name for name, item in record['artifacts'].items()),
            'Every shadow artifact is the exact owned output path')
    require(record['elapsed_s'] < 600 and record['peak_rss_bytes'] < 12e9
            and record['mps_driver_bytes'] < 16e9, 'Recorded shadow resources')
    for entry in record['entries']:
        rank, case = entry['rank'], entry['case']
        require(entry['A_bytes_sha256'] == read(HERE / f'build_{case}_v1/build.json')['A_bytes_sha256'],
                'Fixed prototype A bytes, not a fitted or altered head')
        require(Path(entry['output']) == folder / f'{rank}_{case}_prediction.npy'
                and read(folder / f'{rank}_{case}.json') == entry and entry['all_forward_states_finite'],
                'Exact shadow output/record ownership')
        diagnostics(entry['forward'])
        if case == 'nonzero':
            diagnostics(entry['differentiated'], replay=True)
            require(Path(entry['gradient']) == folder / f'{rank}_nonzero_gradient.npy'
                    and entry['forward']['selection_cells'] == entry['differentiated']['selection_cells']
                    and entry['gradient_norm'] > GRAD_FLOOR and entry['checkpoint_max_mean_difference'] <= MEAN_LIMIT,
                    'Checkpoint selections/gradient/forward')
    return record


def judge(np, args, tiny, native, folder, result, start):
    pin = pinning(result)
    tj = tiny_stage('judge', args.tiny_judge_sha256, tiny, result)
    result['stage_seals'] = dict(tiny_judge=args.tiny_judge_sha256,
        native_parent=args.parent_native_sha256, native_zero=args.zero_native_sha256,
        native_nonzero=args.nonzero_native_sha256, cpu=args.cpu_sha256, mps=args.mps_sha256)
    verify = native_verifier(native, result)
    native_records, native_bytes = {}, {}
    for role in ('parent', 'zero', 'nonzero'):
        path = HERE / f'wide_native_{role}_v1'
        rec = sealed(path, getattr(args, role + '_native_sha256'), pin, completed=False)
        require(rec['schema'] == 'birth_direction_node_head_native_wide_v1' and rec['role'] == role
                and rec['prerequisites']['tiny_native_result_sha256'] == tj['stage_seals']['native']
                and rec['own_complete_bills'] and rec['score_measured'] is False
                and rec['state_instrumentation'] is False, 'Own target-free native role')
        require(rec['source_bindings'][str(HERE / 'native_wide.py')] == NATIVE_SHA
                and Path(rec['source']) == native.CANDIDATES[role][0], 'Actual role/source mapping')
        native_bytes[role] = verify(rec, path, RANKS, pin)
        native_records[role] = rec
    require(native_bytes['zero'] == native_bytes['parent'], 'Zero-node all16 native bytes including repeat')
    require(all(a[:4096] == b[:4096] for a, b in zip(native_bytes['nonzero'], native_bytes['parent'])),
            'Native fixed-nonzero layer0 byte identity')
    result['native_zero_all16_parent_bytes_exact'] = True
    result['native_nonzero_layer0_bytes_exact'] = True
    result['native_complete_bills'] = {role: dict(initialization_flops=rec['initialization_flops'],
        repeat_excluded=True, trials=rec['trials'],
        mean_first_two_complete_flops=sum(x['flops'] for x in rec['trials'][:2])/2)
        for role, rec in native_records.items()}
    shadows = {s: shadow_record(s, getattr(args, s + '_sha256'), tj, result) for s in ('cpu', 'mps')}
    require(shadows['cpu']['environment']['versions'] == shadows['mps']['environment']['versions'],
            'Same CPU/MPS library versions')
    outputs, gradients, comparisons = {}, {}, []
    for stage, rec in shadows.items():
        for entry in rec['entries']:
            rank, case = entry['rank'], entry['case']
            value = np.load(entry['output'], allow_pickle=False)
            require(value.shape == (16, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
                    and (value >= 0).all() and hashlib.sha256(value.tobytes()).hexdigest() == entry['output_bytes_sha256'],
                    'Finite authenticated shadow output')
            reference = np.frombuffer(native_bytes[case][RANKS.index(rank)], dtype='<f4').reshape(16, 1024)
            maximum = float(np.max(np.abs(value.astype(np.float64)-reference.astype(np.float64))))
            comparisons.append(dict(stage=stage, rank=rank, case=case, max_mean_difference=maximum))
            require(maximum <= MEAN_LIMIT, 'All16 native/shadow1e-5 fidelity')
            outputs[stage, rank, case] = value
            if case == 'nonzero':
                grad = np.load(entry['gradient'], allow_pickle=False)
                require(grad.shape == (64, 4) and grad.dtype == np.float32 and np.isfinite(grad).all(), 'Finite A gradient')
                gradients[stage, rank] = grad.astype(np.float64)
    result['mean_comparisons'], result['gradient_comparisons'] = comparisons, []
    for rank in RANKS:
        c, m = gradients['cpu', rank], gradients['mps', rank]
        cn, mn = float(np.linalg.norm(c)), float(np.linalg.norm(m))
        relative = float(np.linalg.norm(m-c)/max(cn, 1e-300))
        row = dict(rank=rank, cpu_norm=cn, mps_norm=mn, relative_error=relative)
        result['gradient_comparisons'].append(row)
        require(cn > GRAD_FLOOR and mn > GRAD_FLOOR and relative <= GRAD_LIMIT, 'Fixed-cot A gradient5% fidelity')
        ce = next(x for x in shadows['cpu']['entries'] if x['rank'] == rank and x['case'] == 'nonzero')
        me = next(x for x in shadows['mps']['entries'] if x['rank'] == rank and x['case'] == 'nonzero')
        require(ce['cot_bytes_sha256'] == me['cot_bytes_sha256'], 'Identical fixed cotangent')
    manifest = NEW / 'joint_prefix_final_fit_v1/manifest.json'
    pin(manifest, TARGET_MANIFEST_SHA)
    expected = {x['rank']: x['target_sha256'] for x in read(manifest)['target_sources']}
    tiny.rehash(result['source_bindings'])
    guard('judge', start)
    tiny.write(folder / 'pretruth_seal.json', dict(stage_seals=result['stage_seals'],
        source_bindings=result['source_bindings'], native_complete_bills=result['native_complete_bills'],
        mean_comparisons=result['mean_comparisons'], gradient_comparisons=result['gradient_comparisons'],
        truth_accessed=False, utc=datetime.now(timezone.utc).isoformat()))
    result['artifacts']['pretruth_seal.json'] = binding(folder / 'pretruth_seal.json')
    result['pretruth_seal_sha256'] = sha(folder / 'pretruth_seal.json')
    result['MSE_comparisons'], result['truth_bindings'] = [], {}
    for rank in RANKS:
        path = P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
        result['truth_accessed'] = True
        require(sha(path) == expected[rank], 'Fixed1e9 TRAIN truth SHA')
        result['truth_bindings'][str(path)] = binding(path)
        with np.load(path, allow_pickle=False) as src:
            truth = src['mc_post'].copy()
        require(truth.shape == (16, 1024) and np.isfinite(truth).all(), 'Finite fixed TRAIN truth')
        for case in CASES:
            reference = np.frombuffer(native_bytes[case][RANKS.index(rank)], dtype='<f4').reshape(16, 1024)
            native_mse = float(np.mean((reference[-1].astype(np.float64)-truth[-1])**2))
            require(math.isfinite(native_mse) and native_mse > 0, 'Positive native final MSE')
            for stage in ('cpu', 'mps'):
                error = float(np.mean((outputs[stage, rank, case][-1].astype(np.float64)-truth[-1])**2))
                ratio = error/native_mse
                row = dict(stage=stage, rank=rank, case=case, native_final_MSE=native_mse,
                           shadow_final_MSE=error, final_MSE_ratio=ratio)
                result['MSE_comparisons'].append(row)
                require(math.isfinite(error) and error > 0 and abs(ratio-1) <= MSE_LIMIT, 'Final MSE within1%')
    tiny.rehash(result['truth_bindings'])
    result.update(fidelity_passed=True, actual_score_measured=False, training_executed=False,
        all_stage_artifacts_verified_before_truth=True,
        truth_frame='Existing1e9 mc_post, TRAIN35/100 only, numerical fidelity not accuracy improvement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu', 'mps', 'judge'))
    parser.add_argument('--source-sha256')
    parser.add_argument('--tiny-sha256')
    parser.add_argument('--mps-slot-clear', action='store_true')
    for name in ('tiny-judge', 'parent-native', 'zero-native', 'nonzero-native', 'cpu', 'mps'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if args.stage is None:
        print('Inert. --stage cpu|mps|judge --source-sha256 SHA; explicit prerequisite seals required.')
        return 0
    require(sha(__file__) == args.source_sha256, 'Reviewed wide checker SHA required')
    require(sha(HERE / 'check_tiny.py') == TINY_SHA and sha(HERE / 'native_wide.py') == NATIVE_SHA,
            'Frozen tiny/native helpers')
    tiny = module(HERE / 'check_tiny.py', 'bdnh_wide_pinned_tiny_helpers')
    folder = HERE / f'wide_{args.stage}_v1'
    folder.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(schema=SCHEMA, stage=args.stage, passed=False, completed=False,
        source_bindings={str(Path(__file__).resolve()): binding(__file__)}, artifacts={},
        no_fit=True, frozen_P=True, frozen_cstar=True, truth_accessed=False,
        training_executed=False, started_utc=datetime.now(timezone.utc).isoformat())
    tiny.write(folder / 'started.json', result)
    def timeout(_signum, _frame):
        raise TimeoutError('Fixed wide-stage watchdog')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(295 if args.stage == 'judge' else 595)
    try:
        result['environment'] = tiny.environment(args.stage, args.mps_slot_clear)
        build, _ = tiny.authenticate(result)
        native = module(HERE / 'native_wide.py', 'bdnh_wide_pinned_native_admission')
        bind_items(native.provenance(), pinning(result))
        pinning(result)(HERE / 'check_tiny.py', TINY_SHA)
        pinning(result)(HERE / 'native_wide.py', NATIVE_SHA)
        weights(result)
        if args.stage == 'judge':
            import numpy as np
            judge(np, args, tiny, native, folder, result, start)
        else:
            result['tiny_stage_sha256'] = args.tiny_sha256
            prerequisite = tiny_stage(args.stage, args.tiny_sha256, tiny, result)
            import numpy as np
            import torch
            model = module(HERE / 'model.py', 'bdnh_wide_frozen_model')
            acquire(np, torch, model, build, args.stage, folder, result, start, prerequisite, tiny)
        tiny.rehash(result['source_bindings'])
        tiny.rehash(result['artifacts'])
        result.update(guard(args.stage, start), passed=True, completed=True, sources_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
        try:
            tiny.rehash(result['source_bindings'])
            result['sources_unchanged'] = True
        except BaseException:
            result['sources_unchanged'] = False
    finally:
        signal.alarm(0)
    result.update(resources(args.stage, start), ended_utc=datetime.now(timezone.utc).isoformat())
    try:
        guard(args.stage, start)
        tiny.rehash(result['source_bindings'])
        tiny.rehash(result['artifacts'])
    except BaseException:
        result.update(passed=False, completed=False, final_guard_error=traceback.format_exc())
    tiny.write(folder / 'result.json', result)
    tiny.write(folder / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
        result_sha256=sha(folder / 'result.json')))
    print(json.dumps(dict(stage=args.stage, passed=result['passed'], elapsed_s=result['elapsed_s'],
                         error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
