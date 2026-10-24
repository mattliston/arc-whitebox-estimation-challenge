"""One fixed20TRAIN/4VALID node-head pilot; inert without --run.

Only shared FP32 A(64,4) is learned. Six epochs/30 updates, terminal only,
no checkpoints, resume, target-selected coefficients or validation scoring.
Completed tiny AND wide fidelity are mandatory. VALID truths are never
opened; all four predictions follow a sealed, restored terminal A payload.
This is an offline fit, not a native complete-bill or release qualification.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
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
CAP = NEW / 'current_paircap4096_e2e_v1'
APENV = Path('/Users/matt/anaconda3')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2 / 'public_g22_operator_features_v1/truthpost'
TARGET_INDEX = NEW / 'joint_prefix_final_fit_v1/manifest.json'
VALID = (37, 39, 42, 48)
TRAIN = tuple(rank for rank in range(35, 59) if rank not in VALID)
SEED, EPOCHS, BATCH, UPDATES = 9309093, 6, 4, 30
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
TINY_SHA = '06e136818954e039626d49585770ad972e4ef1d4061e1d16741d556c1d3c485c'
NATIVE_SHA = '70506904442e58f0931aa254a41cc7be25522b73363db2118eefec6761cdf2e0'
FIXED = {
    HERE / 'model.py': '61833c47e71867f01376a0a3bf3a976b56b67024c485212015f38e7934aadc71',
    HERE / 'core.py': '8abad5831aed8a2fabf47f06b73c9867c482519bcd4dac666c01cc934ccd46a6',
    HERE / 'build.py': '42bef1c5c8aeaf0b60ae96961bb7c29d20535a555bbbfc554a61c5ec0aecb926',
    HERE / 'PROTOCOL.md': '52045c4d089092bc9728fa301a781b39934d8658ad3418ce9d459556554c8056',
    HERE / 'IMPLEMENTATION.md': 'a248f7e8a0edd45359af0efdc25b9d1491c551ae21ebf1ec4bffb9acfa3e1ceb',
    HERE / 'WIDE_SCHEDULING_V1.md': 'cd7f9ca723c51b9288726a494efde1da673b5db40806f51fd51fdb48c03cf14f',
    HERE / 'check_tiny.py': TINY_SHA,
    HERE / 'native_wide.py': NATIVE_SHA,
    CAP / 'train.py': '6780e3e8f9f7ed3e6fddd0c347f6024f756972cf29bf3987e3622b08529cd81c',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    TARGET_INDEX: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
}
CONFIG = dict(seed=SEED, train_ranks=list(TRAIN), validation_ranks=list(VALID),
    reused_development_networks=True, epochs=EPOCHS, batch_nets=BATCH, updates=UPDATES,
    parameter_shape=[64, 4], parameter_dtype='float32', allocated_parameters=256,
    initialization='exact zero once', frozen_P=True, frozen_cstar=True,
    active_birth_layers=list(range(1, 15)), coefficient_axis='birth columns',
    loss='(served final MSE + .05 * mean served earlier-layer MSE)/2e-8',
    optimizer=dict(name='Adam', lr_start=.01, lr_terminal=.001, betas=[.9, .999],
        eps=1e-8, weight_decay=0, gradient_norm_cap=5, foreach=False),
    lr_rule='.001+.5*(.01-.001)*(1+cos(pi*zero_based_update/29))',
    order_rule='six sequential PCG64DXSM(seed) permutations of fixed TRAIN',
    immediate_per_net_backward=True, checkpoint_selection=False,
    parameter_checkpoints=False, resume=False, validation_truth_reads=False,
    monitor_period_s=.5, host_rss_limit_bytes=12000000000,
    mps_driver_limit_bytes=16000000000, native_score_measured=False)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def forbid_validation_truth(path):
    """This trainer has no permission to even hash a VALID truth array."""
    path = Path(path).resolve()
    if path.is_relative_to(TRUTH.resolve()):
        allowed = {TRUTH / f'truthpost_rank{rank:04d}.npz' for rank in TRAIN}
        require(path in allowed, 'Only the20 fixed TRAIN truth paths are allowed')
    return path


def sha(path):
    path = forbid_validation_truth(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = forbid_validation_truth(path)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Input changed while hashing: ' + str(path))
    return dict(path=str(path), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def unchanged(items):
    for item in items.values():
        require(binding(item['path']) == item, 'Sealed input changed: ' + item['path'])


def hashes(items):
    return {path: item['sha256'] for path, item in items.items()}


def absorb(items, destination):
    """Accept existing SHA maps or full bindings, authenticate without arrays."""
    for key, item in items.items():
        path = item['path'] if isinstance(item, dict) else key
        expected = item['sha256'] if isinstance(item, dict) else item
        actual = binding(path)
        require(actual['sha256'] == expected, 'Prerequisite hash: ' + str(path))
        if isinstance(item, dict):
            require(all(actual[field] == item[field] for field in ('bytes', 'mtime_ns')
                        if field in item), 'Prerequisite metadata changed')
        old = destination.get(actual['path'])
        require(old is None or old == actual, 'Conflicting prerequisite binding')
        destination[actual['path']] = actual


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def completed(folder, digest, pins, native=False):
    folder = Path(folder)
    rp, cp = folder / 'result.json', folder / 'completion.json'
    require(sha(rp) == digest, 'Requested completed result seal')
    record, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
    require(record['passed'] and seal['passed'] and seal['result_sha256'] == digest,
            'Completed passing prerequisite required')
    if native:
        require(record['hashes_unchanged'] and not record['truth_accessed'], 'Native prerequisite integrity')
    else:
        require(record['completed'] and seal['completed'] and record['sources_unchanged'],
                'Prerequisite did not complete with unchanged sources')
    absorb(record['source_bindings'], pins)
    absorb(record.get('artifacts', {}), pins)
    # Deliberately do NOT follow truth_bindings: the certificate's result
    # digest seals its earlier fixed fidelity judgment, not training labels.
    absorb({str(rp): digest, str(cp): sha(cp)}, pins)
    return record


def admission(args, result):
    pins = result['source_bindings']
    require(sha(__file__) == args.source_sha256, 'Reviewed trainer source pin')
    absorb({str(path): digest for path, digest in FIXED.items()}, pins)
    require(sha(HERE / 'check_wide.py') == args.wide_source_sha256, 'Reviewed wide checker pin')
    absorb({str(HERE / 'check_wide.py'): args.wide_source_sha256}, pins)
    builder = module(HERE / 'build.py', 'bdnh_train_inert_builder')
    absorb(builder.authenticate(), pins)
    cstar = builder.frozen_coefficients()
    wide = completed(HERE / 'wide_judge_v1', args.wide_judge_sha256, pins)
    require(wide['schema'] == 'birth_direction_node_head_wide_v1' and wide['stage'] == 'judge',
            'Wrong wide certificate frame')
    require(wide['no_fit'] and wide['frozen_P'] and wide['frozen_cstar'], 'Qualified frozen state required')
    require(wide['fidelity_passed'] and wide['all_stage_artifacts_verified_before_truth']
            and not wide['training_executed'], 'Complete wide fidelity, not a partial acquisition')
    seals = wide['stage_seals']
    require(set(seals) == {'tiny_judge', 'native_parent', 'native_zero', 'native_nonzero', 'cpu', 'mps'},
            'Incomplete fixed wide gate')
    tiny = completed(HERE / 'stage_judge', seals['tiny_judge'], pins)
    require(tiny['schema'] == 'birth_direction_node_head_tiny_v1' and tiny['stage'] == 'judge'
            and tiny['small_fidelity_passed'] and tiny['all_stage_artifacts_verified_before_read'],
            'Complete tiny gate required')
    require(tiny['prototype_seals'] == wide['prototype_seals'], 'Tiny/wide prototype mismatch')
    for role in ('parent', 'zero', 'nonzero'):
        native = completed(HERE / f'wide_native_{role}_v1', seals['native_' + role], pins, native=True)
        require(native['schema'] == 'birth_direction_node_head_native_wide_v1'
                and native['role'] == role and [x['rank'] for x in native['trials']] == [35, 100, 35],
                'Fixed native role/sequence')
        require(native['own_complete_bills'] and native['initialization_once'] and native['repeat_excluded'],
                'Native complete-bill ownership')
    for stage in ('cpu', 'mps'):
        shadow = completed(HERE / f'wide_{stage}_v1', seals[stage], pins)
        require(shadow['schema'] == wide['schema'] and shadow['stage'] == stage
                and shadow['no_fit'] and shadow['frozen_P'] and shadow['frozen_cstar'],
                'Wrong fixed autonomous shadow')
        require(shadow['environment']['versions'] == result['environment']['versions'],
                'Training must use the certified CPU/MPS library versions')
    result['wide_judge_sha256'] = args.wide_judge_sha256
    result['wide_source_sha256'] = args.wide_source_sha256
    result['fidelity_stage_seals'] = dict(seals)
    result['frozen_cstar_bytes_sha256'] = hashlib.sha256(builder.matrix_bytes(cstar, 15, 4)).hexdigest()
    unchanged(pins)
    return builder, cstar


def data_bindings():
    """Bind24 weights and20 TRAIN truth files; no numerical file is decoded."""
    items = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    weights, rows = {}, {}
    for rank in TRAIN + VALID:
        matches = [item for item in items if item['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical network row')
        item = matches[0]
        path = STORE / item['path']
        require(sha(path) == item['sha256'], 'Canonical row manifest')
        row = json.loads(path.read_text())
        wp = path.parent / 'weights.npy'
        require(sha(wp) == row['files']['weights.npy']['sha256'], 'Canonical weights')
        absorb({str(path): item['sha256'], str(wp): row['files']['weights.npy']['sha256']}, weights)
        rows[rank] = dict(weights=str(wp.resolve()), mlp_seed=int(row['mlp_seed']))
    index = json.loads(TARGET_INDEX.read_text())
    expected = {item['rank']: item['target_sha256'] for item in index['target_sources']}
    require(set(TRAIN).issubset(expected) and not set(VALID) & set(TRAIN), 'Training label ownership')
    targets = {}
    for rank in TRAIN:
        path = TRUTH / f'truthpost_rank{rank:04d}.npz'
        absorb({str(path): expected[rank]}, targets)
    require(len(TRAIN) == 20 and len(VALID) == 4 and len(targets) == 20, 'Fixed split')
    return weights, rows, targets


def environment(clear):
    require(Path(sys.prefix).resolve() == APENV.resolve(), 'AP Torch environment required')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Requires nice10')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
    require(all(os.environ.get(key) == '1' for key in BLAS), 'Explicit BLAS1 environment')
    require(all(os.environ.get(key, '1') == '1' for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'No extra CPU numerical pools')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
    require(not any(key in sys.modules for key in ('torch', 'flopscope', 'threadpoolctl')),
            'Fresh AP process; never AP plus threadpoolctl/FNP')
    require(clear, '--mps-slot-clear attests sole MPS ownership after existing job exit')
    return dict(executable=sys.executable, prefix=sys.prefix, pid=os.getpid(),
        nice=os.getpriority(os.PRIO_PROCESS, 0), mps_slot_clear_attested=True,
        versions={key: importlib.metadata.version(key) for key in ('torch', 'numpy')},
        variables={key: os.environ.get(key) for key in (*BLAS, 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'PYTORCH_ENABLE_MPS_FALLBACK')})


def imported_sources(pins):
    for loaded in tuple(sys.modules.values()):
        path = getattr(loaded, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                require(str(path) in pins and sha(path) == pins[str(path)]['sha256'],
                        'Unbound local imported source: ' + str(path))
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
            'Unexpected incompatible AP runtime')


def chart(card, A, cstar_bytes, training, forward_ids=None):
    """Host-only records never enter coefficients or checkpoint state."""
    require(card.A is A and card.theta is None and card.audit_selections, 'Only the shared A may vary')
    require(all(not value.requires_grad and value.grad is None for value in card.P0.values()), 'Frozen P changed')
    require(card.coefficients.detach().cpu().numpy().tobytes() == cstar_bytes, 'Frozen cstar bytes')
    diagnostic = card.diagnostics()
    domain = diagnostic['variance_domain']
    require(set(domain) == set(range(16)), 'Missing prevariance layer')
    require(all(item['finite'] and item['minimum'] >= 1e-10 and not item['below_floor_count']
                for item in domain.values()), 'Outside the qualified prevariance chart')
    require(set(card.node_head_calls) == set(range(1, 15)) and all(v >= 1 for v in card.node_head_calls.values()),
            'Unexpected node-head layer ownership')
    expected = {(layer, slot) for layer in range(16) for slot in range(3 if layer < 15 else 2)}
    require(set(card.selection_cells) == expected, 'Incomplete physical selection audit')
    if forward_ids is not None:
        require(card.selection_cells == forward_ids, 'Checkpoint changed forward selection memberships')
    encoded = json.dumps(diagnostic['selection_cells'], sort_keys=True, separators=(',', ':')).encode()
    return dict(variance_domain=domain, node_head_calls=dict(card.node_head_calls),
        selection_cells_sha256=hashlib.sha256(encoded).hexdigest(),
        selection_sizes={key: len(ids) for key, ids in diagnostic['selection_cells'].items()},
        checkpoint_membership_checked=bool(training and forward_ids is not None),
        no_parameter_or_state_override=True)


def save_terminal(np, A, builder, path):
    values = A.detach().cpu().numpy().copy()
    require(values.shape == (64, 4) and values.dtype == np.float32 and np.isfinite(values).all(), 'Terminal A')
    literal = values.tolist()
    encoded = builder.matrix_bytes(literal, 64, 4)
    require(encoded == values.tobytes(), 'Literal A must preserve exact FP32 bytes')
    data = dict(schema='birth_direction_node_head_terminal_A_v1', A=literal,
        shape=[64, 4], dtype='float32', byte_order='little', A_bytes_sha256=hashlib.sha256(encoded).hexdigest(),
        raw_sha256=hashlib.sha256(encoded).hexdigest(),
        terminal_epoch=EPOCHS, updates=UPDATES, learned=True, checkpoint_selection=False,
        coefficient_rule='cstar+(1-abs(cstar))*tanh(Z@A), FP32; cstar unchanged')
    write(path, data)
    return dict(**binding(path), A_bytes_sha256=data['A_bytes_sha256'],
                raw_sha256=data['raw_sha256'], shape=[64, 4], dtype='float32')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--wide-source-sha256')
    parser.add_argument('--wide-judge-sha256')
    parser.add_argument('--mps-slot-clear', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    for key in ('source_sha256', 'wide_source_sha256', 'wide_judge_sha256'):
        value = getattr(args, key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            parser.error('--run requires all three reviewed SHA256 values')
    out = HERE / 'training_v1'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='birth_direction_node_head_training_v1', passed=False, completed=False,
        config=CONFIG, source_bindings={str(Path(__file__).resolve()): binding(__file__)}, artifacts={},
        seed=SEED, epochs=EPOCHS, planned_updates=UPDATES, batch_nets=BATCH,
        train_ranks=list(TRAIN), validation_ranks=list(VALID), allocated_parameters=256,
        training_targets_accessed=False, validation_targets_accessed=False,
        validation_score_measured=False, native_complete_bill_measured=False,
        checkpoint_selection=False, parameter_checkpoints=False, automatic_advancement=False,
        maximum_sampled_mps_driver_bytes=0, last_resource_measurement={},
        updates_completed=0, started_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'started.json', result)
    progress = (out / 'progress.jsonl').open('x')
    torch_module, monitoring = None, False

    def artifact(path):
        result['artifacts'][str(Path(path).resolve())] = binding(path)

    def record(**item):
        row = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - start, **item)
        line = json.dumps(safe(row), sort_keys=True, allow_nan=False)
        progress.write(line + '\n')
        progress.flush()
        print(line, flush=True)

    def monitor(_signum=None, _frame=None):
        nonlocal monitoring
        if monitoring:
            return dict(result['last_resource_measurement'])
        monitoring = True
        try:
            raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            state = dict(peak_rss_bytes=int(raw_rss if sys.platform == 'darwin' else raw_rss * 1024),
                mps_driver_bytes=int(torch_module.mps.driver_allocated_memory()) if torch_module else 0)
            result['last_resource_measurement'] = state
            result['maximum_sampled_mps_driver_bytes'] = max(
                result['maximum_sampled_mps_driver_bytes'], state['mps_driver_bytes'])
            require(state['peak_rss_bytes'] < 12000000000 and state['mps_driver_bytes'] < 16000000000,
                    'Training12GB host/16GB MPS cap: ' + repr(state))
            return state
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        finally:
            monitoring = False

    previous_handler = signal.signal(signal.SIGALRM, monitor)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        result['environment'] = environment(args.mps_slot_clear)
        builder, cstar_literal = admission(args, result)
        cstar_bytes = builder.matrix_bytes(cstar_literal, 15, 4)
        weight_pins, rows, target_pins = data_bindings()
        write(out / 'config.json', CONFIG)
        write(out / 'weight_bindings.json', dict(bindings=weight_pins, rows=rows))
        write(out / 'training_targets.json', dict(bindings=target_pins, train_ranks=list(TRAIN),
            validation_targets_accessed=False, all_authenticated_before_decode=True))
        for name in ('config.json', 'weight_bindings.json', 'training_targets.json'):
            artifact(out / name)
        write(out / 'input_seal.json', dict(source_bindings=hashes(result['source_bindings']),
            config_sha256=sha(out / 'config.json'), weight_bindings_sha256=sha(out / 'weight_bindings.json'),
            training_targets_sha256=sha(out / 'training_targets.json'),
            validation_targets_accessed=False, no_numerical_fit_started=True,
            utc=datetime.now(timezone.utc).isoformat()))
        artifact(out / 'input_seal.json')
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.backends.mps.is_available(), 'MPS unavailable')
        torch_module = torch
        torch.manual_seed(SEED)
        model = module(HERE / 'model.py', 'bdnh_fixed_pilot_model')
        imported_sources(result['source_bindings'])
        require(builder.matrix_bytes(model.CSTAR_LITERAL, 15, 4) == cstar_bytes, 'Model cstar source identity')
        require(all(Path(model.base.weight_path(rank)).resolve() == Path(rows[rank]['weights'])
                    for rank in TRAIN + VALID), 'Autonomous model uses authenticated weight paths')
        # Read only the20 files authenticated and sealed above. No target
        # locator/function can be called with a VALID rank in this program.
        targets = {}
        for rank in TRAIN:
            path = TRUTH / f'truthpost_rank{rank:04d}.npz'
            require(binding(path) == target_pins[str(path.resolve())], 'Training target changed before decode')
            result['training_targets_accessed'] = True
            with np.load(path, allow_pickle=False) as source:
                values = source['mc_post'].copy()
            require(values.shape == (16, 1024) and np.isfinite(values).all(), 'Finite training target')
            targets[rank] = torch.tensor(values, device='mps', dtype=torch.float32)
            del values
        A = torch.nn.Parameter(torch.zeros((64, 4), device='mps', dtype=torch.float32))
        require(A.numel() == 256 and A.detach().cpu().numpy().tobytes() == bytes(256 * 4), 'One exact zero A')
        optimizer = torch.optim.Adam([A], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
        require(len(optimizer.param_groups[0]['params']) == 1
                and optimizer.param_groups[0]['params'][0] is A, 'A is the only optimizer parameter')
        rng = np.random.Generator(np.random.PCG64DXSM(SEED))
        orders = [rng.permutation(TRAIN).tolist() for _ in range(EPOCHS)]
        write(out / 'fixed_orders.json', dict(seed=SEED, orders=orders, selected_before_first_update=True))
        artifact(out / 'fixed_orders.json')
        history, step = [], 0
        record(stage='start', training_nets=20, validation_truths=0, allocated_parameters=256, **monitor())
        for epoch, order in enumerate(orders, 1):
            epoch_raw = []
            for offset in range(0, len(order), BATCH):
                ranks = order[offset:offset+BATCH]
                require(len(ranks) == BATCH and step < UPDATES, 'Fixed batch/update count')
                lr = .001 + .5 * (.01 - .001) * (1 + math.cos(math.pi * step / (UPDATES - 1)))
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                batch_raw = []
                for rank in ranks:
                    card = model.NodeHeadCard(rank, A=A, device='mps', audit_selections=True)
                    prediction, _ = card.run(training=True)
                    require(prediction.shape == (16, 1024) and torch.isfinite(prediction).all().item(),
                            'Finite autonomous prediction')
                    forward = {key: list(ids) for key, ids in card.selection_cells.items()}
                    chart(card, A, cstar_bytes, training=True)
                    prediction = torch.where(prediction < 0, torch.zeros_like(prediction), prediction)
                    error = prediction - targets[rank]
                    final, auxiliary = error[-1].square().mean(), error[:-1].square().mean()
                    loss = (final + .05 * auxiliary) / 2e-8
                    raw = float(final.detach().cpu())
                    require(math.isfinite(raw) and bool(torch.isfinite(loss)), 'Finite training loss')
                    (loss / BATCH).backward()
                    diagnostic = chart(card, A, cstar_bytes, training=True, forward_ids=forward)
                    require(A.grad is not None and bool(torch.isfinite(A.grad).all()), 'Finite accumulated A gradient')
                    batch_raw.append(raw)
                    epoch_raw.append(raw)
                    record(stage='train_net', epoch=epoch, update=step+1, rank=rank,
                        final_raw=raw, auxiliary_mse=float(auxiliary.detach().cpu()),
                        diagnostics=diagnostic, **monitor())
                    del card, prediction, error, final, auxiliary, loss, forward, diagnostic
                    gc.collect()
                    torch.mps.empty_cache()
                norm = float(torch.nn.utils.clip_grad_norm_([A], 5., error_if_nonfinite=True,
                             foreach=False).detach().cpu())
                optimizer.step()
                step += 1
                result['updates_completed'] = step
                require(bool(torch.isfinite(A).all()) and math.isfinite(norm), 'Finite updated A')
                record(stage='batch', epoch=epoch, update=step, ranks=ranks, learning_rate=lr,
                    online_raw_mean=float(np.mean(batch_raw)), gradient_norm=norm, **monitor())
            history.append(dict(epoch=epoch, online_raw_mean=float(np.mean(epoch_raw))))
            record(stage='epoch_complete', **history[-1], parameter_checkpoint_saved=False)
        require(step == UPDATES and len(history) == EPOCHS, 'No shortened or extended fit')
        unchanged(result['source_bindings'])
        unchanged(weight_pins)
        unchanged(target_pins)
        terminal = save_terminal(np, A, builder, out / 'terminal_A.json')
        artifact(out / 'terminal_A.json')
        terminal_seal = dict(schema='birth_direction_node_head_terminal_seal_v1', terminal_A=terminal,
            epochs=EPOCHS, updates=step, seed=SEED, source_bindings=hashes(result['source_bindings']),
            config_sha256=sha(out / 'config.json'),
            weight_bindings_sha256=sha(out / 'weight_bindings.json'),
            training_targets_sha256=sha(out / 'training_targets.json'),
            input_seal_sha256=sha(out / 'input_seal.json'), fixed_orders_sha256=sha(out / 'fixed_orders.json'),
            frozen_cstar_bytes_sha256=result['frozen_cstar_bytes_sha256'],
            validation_predictions_started=False, validation_targets_accessed=False,
            checkpoint_selection=False, utc=datetime.now(timezone.utc).isoformat())
        write(out / 'terminal_seal.json', terminal_seal)
        artifact(out / 'terminal_seal.json')
        record(stage='terminal_sealed', terminal_A_sha256=terminal['sha256'],
            A_bytes_sha256=terminal['A_bytes_sha256'])
        # Remove the optimizer and live trained tensor. VALID uses only the
        # immutable JSON's exact float32 values, restored and checked below.
        del targets, optimizer, A
        gc.collect()
        torch.mps.empty_cache()
        require(binding(terminal['path']) == {k: terminal[k] for k in ('path', 'sha256', 'bytes', 'mtime_ns')},
                'Terminal payload changed before restore')
        data = json.loads(Path(terminal['path']).read_text())
        encoded = builder.matrix_bytes(data['A'], 64, 4)
        require(hashlib.sha256(encoded).hexdigest() == terminal['A_bytes_sha256'], 'Terminal encoded bytes')
        restored = np.asarray(data['A'], dtype=np.float32)
        require(restored.shape == (64, 4) and restored.tobytes() == encoded, 'Terminal NumPy byte restore')
        terminal_A = torch.tensor(restored, device='mps', dtype=torch.float32)
        require(not terminal_A.requires_grad and terminal_A.detach().cpu().numpy().tobytes() == encoded,
                'Terminal MPS byte restore')
        result['terminal_restore_byte_exact'] = True
        outputs = {}
        with torch.no_grad():
            for rank in VALID:
                card = model.NodeHeadCard(rank, A=terminal_A, device='mps', audit_selections=True)
                prediction, _ = card.run(training=False)
                diagnostic = chart(card, terminal_A, cstar_bytes, training=False)
                served = torch.where(prediction < 0, torch.zeros_like(prediction), prediction).cpu().float().numpy().copy()
                require(served.shape == (16, 1024) and served.dtype == np.float32
                        and np.isfinite(served).all() and (served >= 0).all(), 'Finite served VALID output')
                path = out / f'validation_{rank:04d}.npy'
                with path.open('xb') as stream:
                    np.save(stream, served)
                outputs[str(rank)] = dict(**binding(path), shape=[16, 1024], dtype='float32',
                    output_bytes_sha256=hashlib.sha256(served.tobytes()).hexdigest(),
                    raw_sha256=hashlib.sha256(served.tobytes()).hexdigest())
                artifact(path)
                record(stage='validation_prediction_only', rank=rank, diagnostics=diagnostic, **monitor())
                del card, prediction, served, diagnostic
                gc.collect()
                torch.mps.empty_cache()
        require(set(map(int, outputs)) == set(VALID) and len(outputs) == 4, 'All4 fixed predictions required')
        require(terminal_A.detach().cpu().numpy().tobytes() == encoded, 'Frozen terminal tensor changed')
        imported_sources(result['source_bindings'])
        unchanged(result['source_bindings'])
        unchanged(weight_pins)
        unchanged(target_pins)
        unchanged(result['artifacts'])
        sealed = dict(schema='birth_direction_node_head_validation_seal_v1', terminal_A=terminal,
            terminal_seal_sha256=sha(out / 'terminal_seal.json'), validation_outputs=outputs,
            all_4_complete=True, terminal_restore_byte_exact=True, validation_targets_accessed=False,
            native_complete_bill_measured=False, raw_or_adjusted_score_measured=False,
            utc=datetime.now(timezone.utc).isoformat())
        write(out / 'validation_predictions_sealed.json', sealed)
        artifact(out / 'validation_predictions_sealed.json')
        result.update(passed=True, completed=True, history=history, epochs=EPOCHS, updates=step,
            terminal_A=terminal, terminal_seal_sha256=sha(out / 'terminal_seal.json'),
            validation_outputs=outputs, frozen_P=True, frozen_cstar=True, hashes_unchanged=True,
            validation_predictions_sealed_sha256=sha(out / 'validation_predictions_sealed.json'), **monitor())
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        progress.close()
    try:
        result.update(monitor())
        unchanged(result['source_bindings'])
        unchanged(result['artifacts'])
    except BaseException:
        result.update(passed=False, completed=False, final_guard_error=traceback.format_exc())
    artifact(out / 'progress.jsonl')
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic() - start)
    # CAP-compatible immutable result maps; retain exact metadata companions.
    result['source_binding_metadata'] = result['source_bindings']
    result['artifact_metadata'] = result['artifacts']
    result['source_bindings'] = hashes(result['source_binding_metadata'])
    result['artifacts'] = hashes(result['artifact_metadata'])
    write(out / 'result.json', result)
    write(out / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
        result_sha256=sha(out / 'result.json')))
    print(json.dumps(safe(dict(passed=result['passed'], completed=result['completed'],
        updates_completed=result['updates_completed'], wall_s=result['wall_s'],
        last_resource_measurement=result['last_resource_measurement'])), sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
