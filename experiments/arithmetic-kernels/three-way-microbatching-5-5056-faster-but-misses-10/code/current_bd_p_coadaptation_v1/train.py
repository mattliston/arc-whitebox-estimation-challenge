"""One fixed autonomous all-P fit on d50, inert without --run.

No validation targets or scores are ever read here. Only the sealed epoch6
payload is used to create all40 validation predictions. Complete native
adjusted evaluation belongs to a separately authenticated terminal judge.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
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
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MANIFEST = NEW/'joint_prefix_final_fit_v1/manifest.json'
FIXED = {
    MANIFEST: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    NEW/'payload_response_rows_v1.npz': '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c',
    NEW/'joint_prefix_final_data_v1.py': '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
}
SEED, EPOCHS, BATCH, UPDATES = 9309096, 6, 4, 240


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def unchanged(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('Authenticated input changed: ' + str(path))


def completed(folder, expected=None):
    folder = Path(folder)
    rp, cp = folder/'result.json', folder/'completion.json'
    digest = sha(rp)
    if expected is not None and digest != expected:
        raise RuntimeError('Completed result digest mismatch: ' + str(rp))
    result, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
    if not (result['completed'] and result['passed'] and seal['completed'] and seal['passed']
            and seal['result_sha256'] == digest):
        raise RuntimeError('Completed PASS prerequisite required: ' + str(folder))
    unchanged(result.get('source_bindings', {}))
    unchanged(result.get('artifacts', {}))
    return result


def admission(args):
    pins = {str(Path(__file__).resolve()): args.source_sha256,
            str(HERE/'model.py'): args.model_sha256, str(HERE/'PROTOCOL_V1.md'): args.protocol_sha256,
            **{str(p): d for p, d in FIXED.items()}}
    unchanged(pins)
    # This adapter is stdlib-only at import. The target-free result is verified
    # before loading any numerical library or resolving any truth array.
    import model
    if 'numpy' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Model import must remain inert')
    pins.update(model.authenticate())
    judge = completed(HERE/'stage_judge_v1', args.qualification_sha256)
    if judge.get('alltruth_accessed') is not False:
        raise RuntimeError('Qualification must explicitly be target-free')
    pins.update(judge['source_bindings'])
    pins.update(judge.get('artifacts', {}))
    for stage in ('cpu_forward', 'mps_forward', 'cpu_gradient', 'mps_gradient', 'judge'):
        rec = completed(HERE/('stage_'+stage+'_v1'))
        if rec.get('alltruth_accessed') is not False:
            raise RuntimeError('A target-free stage accessed labels')
        if rec['source_bindings'][str(HERE/'model.py')] != args.model_sha256:
            raise RuntimeError('Qualification used a different model')
        if rec['source_bindings'][str(HERE/'PROTOCOL_V1.md')] != args.protocol_sha256:
            raise RuntimeError('Qualification used a different protocol')
        pins.update(rec['source_bindings'])
        for name in ('result.json', 'completion.json'):
            path = HERE/('stage_'+stage+'_v1')/name
            pins[str(path)] = sha(path)
        watched = completed(HERE/('watch_'+stage+'_v1'))
        if watched['child_result_sha256'] != sha(HERE/('stage_'+stage+'_v1')/'result.json'):
            raise RuntimeError('External supervisor is not bound to qualified child')
        pins.update(watched['source_bindings'])
        for name in ('result.json', 'completion.json'):
            path = HERE/('watch_'+stage+'_v1')/name
            pins[str(path)] = sha(path)
    unchanged(pins)
    return pins, model


def weight_bindings(ranks):
    entries = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    pins, rows = {}, {}
    for rank in ranks:
        match = [item for item in entries if item['role_rank'] == rank]
        if len(match) != 1:
            raise RuntimeError('Canonical row is not unique')
        item = match[0]
        row_path = STORE/item['path']
        if sha(row_path) != item['sha256']:
            raise RuntimeError('Canonical row manifest mismatch')
        row = json.loads(row_path.read_text())
        weight_path = row_path.parent/'weights.npy'
        if sha(weight_path) != row['files']['weights.npy']['sha256']:
            raise RuntimeError('Canonical weights mismatch')
        pins[str(row_path)] = item['sha256']
        pins[str(weight_path)] = row['files']['weights.npy']['sha256']
        rows[rank] = str(weight_path.resolve())
    return pins, rows


def save_payload(np, torch, path, P, original, keys, inactive):
    values = {}
    for key, reference in original.items():
        if key in keys:
            value = P[key].detach().cpu().numpy().copy()
            assert value.dtype == np.float32 and value.shape == reference.shape, key
        else:
            assert not P[key].requires_grad
            assert torch.equal(P[key], torch.as_tensor(reference, device='mps', dtype=torch.float32)), key
            value = reference.copy()
        assert np.isfinite(value).all(), key
        values[key] = value
    for key, reference in inactive.items():
        assert torch.equal(P[key][0].detach(), reference), key
        assert values[key][0].tobytes() == original[key][0].tobytes(), key
    with path.open('xb') as stream:
        np.savez(stream, **values)
    return dict(path=str(path), sha256=sha(path), arrays={key: dict(shape=list(value.shape),
        dtype=str(value.dtype), raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--model-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--qualification-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    environment = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    assert all(value == '1' for value in environment.values()), environment
    pins, model = admission(args)
    sys.path.insert(0, str(NEW))
    # Keep our already imported model reference: numerical imports below may
    # adjust sys.path but must not redirect the adapter.
    out = HERE/'training_v1'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='current_d50_p_coadaptation_training_v1', passed=False, completed=False,
        seed=SEED, epochs=EPOCHS, batch_nets=BATCH, planned_updates=UPDATES,
        allocated_parameters=45200, active_parameters=45134, source_bindings=pins, artifacts={},
        training_targets_accessed=False, validation_targets_accessed=False,
        qualification_sha256=args.qualification_sha256, Q_used=False, birth_coefficients_frozen=True,
        full_autonomous_birth_gradients=True, native_terminal_score_measured=False,
        checkpoint_selection=False, environment=environment, maximum_sampled_mps_driver_bytes=0,
        loss='(served final MSE + .05 * served earlier-layer MSE)/2e-8',
        optimizer=dict(name='Adam', lr_start=.0005, lr_terminal=.00005, betas=[.9, .999],
            eps=1e-8, weight_decay=0, gradient_norm_cap=5., foreach=False),
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    progress = (out/'progress.jsonl').open('x')
    torch_module = None

    def record(**item):
        value = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-start, **item)
        line = json.dumps(value, allow_nan=False)
        progress.write(line+'\n')
        progress.flush()
        print(line, flush=True)

    def monitor(signum=None, frame=None):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        driver = int(torch_module.mps.driver_allocated_memory()) if torch_module is not None else 0
        result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
        if rss >= 12_000_000_000 or driver >= 16_000_000_000:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise MemoryError('Prospective offline memory limit: ' + repr((rss, driver)))
        return dict(peak_rss_bytes=rss, mps_driver_bytes=driver)

    previous = signal.signal(signal.SIGALRM, monitor)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.backends.mps.is_available()
        torch_module = torch
        versions = dict(numpy=np.__version__, torch=torch.__version__)
        for stage in ('mps_forward', 'mps_gradient'):
            qualified = json.loads((HERE/('stage_'+stage+'_v1')/'result.json').read_text())
            assert qualified['environment']['versions'] == versions, 'Qualified numerical library frame changed'
        result['environment']['versions'] = versions
        torch.manual_seed(SEED)
        from joint_prefix_final_data_v1 import TRAIN, VALID, PKEYS, weight_path
        assert len(TRAIN) == 160 and len(VALID) == 40 and not set(TRAIN)&set(VALID)
        assert set(TRAIN)|set(VALID) == set(range(35, 235))
        assert tuple(PKEYS) == tuple(model.PKEYS)
        cstar = model.frozen_coefficients()
        cstar_sha = hashlib.sha256(np.asarray(cstar, dtype=np.float32).tobytes()).hexdigest()
        result.update(train_ranks=list(TRAIN), validation_ranks=list(VALID),
                      trainable_keys=list(PKEYS), cstar_raw_sha256=cstar_sha)
        weight_pins, rows = weight_bindings(tuple(TRAIN)+tuple(VALID))
        assert all(str(Path(weight_path(r)).resolve()) == rows[r] for r in rows)
        write(out/'weight_bindings.json', weight_pins)
        result['artifacts'][str(out/'weight_bindings.json')] = sha(out/'weight_bindings.json')
        # Read only the fixed training-label inventory after full qualification.
        expected = {x['rank']: x['target_sha256'] for x in json.loads(MANIFEST.read_text())['target_sources']}
        assert set(expected) == set(TRAIN)
        targets, target_pins = {}, {}
        for rank in TRAIN:
            path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
            assert sha(path) == expected[rank], rank
            target_pins[str(path)] = expected[rank]
            with np.load(path, allow_pickle=False) as source:
                values = source['mc_post'].copy()
            result['training_targets_accessed'] = True
            assert values.shape == (16, 1024) and np.isfinite(values).all(), rank
            targets[rank] = torch.tensor(values, device='mps', dtype=torch.float32)
        write(out/'training_targets.json', target_pins)
        result['artifacts'][str(out/'training_targets.json')] = sha(out/'training_targets.json')
        original = model.payload_numpy()
        P = {key: torch.tensor(value, device='mps', dtype=torch.float32,
                              requires_grad=key in PKEYS) for key, value in original.items()}
        parameters = [P[key] for key in PKEYS]
        assert sum(v.numel() for v in parameters) == 45200
        assert all(original[k].dtype == np.float32 for k in PKEYS)
        inactive = {key: P[key][0].detach().clone() for key in ('head', 'head_b', 'gain')}
        assert sum(v.numel() for v in inactive.values()) == 66
        optimizer = torch.optim.Adam(parameters, lr=.0005, betas=(.9, .999), eps=1e-8,
                                      weight_decay=0, foreach=False)
        rng = np.random.Generator(np.random.PCG64DXSM(SEED))
        history, checkpoints, step = [], [], 0
        record(stage='start', training_nets=len(TRAIN), validation_truths=0, **monitor())
        for epoch in range(EPOCHS):
            order = rng.permutation(TRAIN).tolist()
            metrics = []
            for offset in range(0, len(order), BATCH):
                ranks = order[offset:offset+BATCH]
                assert len(ranks) == BATCH
                lr = .00005+.5*(.0005-.00005)*(1+math.cos(math.pi*step/(UPDATES-1)))
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                batch_raw = []
                for rank in ranks:
                    card = model.PCard(rank, device='mps', P=P, audit_selections=True)
                    prediction, _ = card.run(training=True)
                    model.check_domain(card)
                    served = torch.where(prediction < 0, torch.zeros_like(prediction), prediction)
                    error = served-targets[rank]
                    final, auxiliary = error[-1].square().mean(), error[:-1].square().mean()
                    loss = (final+.05*auxiliary)/2e-8
                    raw = float(final.detach().cpu())
                    assert math.isfinite(raw) and bool(torch.isfinite(loss)), rank
                    (loss/BATCH).backward()
                    model.check_domain(card)
                    model.check_replay(card)
                    # Audit exact freezing without detaching endogenous BD bases.
                    assert not card.coefficients.requires_grad
                    assert torch.equal(card.coefficients,
                        torch.tensor(cstar, device='mps', dtype=torch.float32))
                    batch_raw.append(raw)
                    metrics.append(raw)
                    monitor()
                    del card, prediction, served, error, final, auxiliary, loss
                    gc.collect()
                    torch.mps.empty_cache()
                for key in PKEYS:
                    assert P[key].grad is not None and bool(torch.isfinite(P[key].grad).all()), key
                for key in inactive:
                    assert bool((P[key].grad[0] == 0).all()), key
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True,
                                                            foreach=False).detach().cpu())
                optimizer.step()
                step += 1
                assert all(bool(torch.isfinite(value).all()) for value in parameters)
                assert all(torch.equal(P[key][0].detach(), value) for key, value in inactive.items())
                record(stage='batch', epoch=epoch+1, update=step, learning_rate=lr, ranks=ranks,
                       online_raw_mean=float(np.mean(batch_raw)), gradient_norm=norm, **monitor())
            checkpoint = save_payload(np, torch, out/f'epoch_{epoch+1:02d}_audit_only.npz',
                                      P, original, PKEYS, inactive)
            checkpoint.update(epoch=epoch+1, checkpoint_selection=False)
            checkpoints.append(checkpoint)
            result['artifacts'][checkpoint['path']] = checkpoint['sha256']
            history.append(dict(epoch=epoch+1, online_raw_mean=float(np.mean(metrics))))
            record(stage='epoch_complete', **history[-1], checkpoint_sha256=checkpoint['sha256'])
        assert step == UPDATES
        terminal = save_payload(np, torch, out/'terminal_payload.npz', P, original, PKEYS, inactive)
        result['artifacts'][terminal['path']] = terminal['sha256']
        write(out/'terminal_seal.json', dict(terminal_epoch=EPOCHS, updates=step,
            terminal_payload=terminal, source_bindings=pins, cstar_raw_sha256=cstar_sha,
            weight_bindings_sha256=sha(out/'weight_bindings.json'),
            training_targets_sha256=sha(out/'training_targets.json'), validation_targets_accessed=False,
            checkpoint_selection=False, utc=datetime.now(timezone.utc).isoformat()))
        result['artifacts'][str(out/'terminal_seal.json')] = sha(out/'terminal_seal.json')
        record(stage='terminal_sealed', payload_sha256=terminal['sha256'])
        del targets, optimizer, parameters, P
        gc.collect()
        torch.mps.empty_cache()
        with np.load(terminal['path'], allow_pickle=False) as source:
            terminal_P = {key: torch.tensor(source[key].copy(), device='mps', dtype=torch.float32)
                          for key in source.files}
        assert sha(terminal['path']) == terminal['sha256']
        outputs = {}
        with torch.no_grad():
            for rank in VALID:
                card = model.PCard(rank, device='mps', P=terminal_P, audit_selections=True)
                prediction, _ = card.run()
                model.check_domain(card)
                served = torch.where(prediction < 0, torch.zeros_like(prediction), prediction).cpu().float().numpy().copy()
                assert served.shape == (16, 1024) and served.dtype == np.float32
                assert np.isfinite(served).all() and (served >= 0).all()
                path = out/f'validation_{rank:04d}.npy'
                with path.open('xb') as stream:
                    np.save(stream, served)
                outputs[str(rank)] = dict(path=str(path), sha256=sha(path),
                    raw_sha256=hashlib.sha256(served.tobytes()).hexdigest())
                result['artifacts'][str(path)] = sha(path)
                del card, prediction, served
                gc.collect()
                torch.mps.empty_cache()
                record(stage='validation_prediction_only', rank=rank, **monitor())
        assert set(map(int, outputs)) == set(VALID) and len(outputs) == 40
        assert sha(terminal['path']) == terminal['sha256']
        unchanged(pins)
        unchanged(weight_pins)
        unchanged(target_pins)
        unchanged(result['artifacts'])
        # Also ensure every transitive project numerical source was authenticated.
        for imported in tuple(sys.modules.values()):
            path = getattr(imported, '__file__', None)
            if path:
                path = Path(path).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    assert str(path) in pins and pins[str(path)] == sha(path), path
        write(out/'validation_predictions_sealed.json', dict(terminal_payload=terminal,
            terminal_seal_sha256=sha(out/'terminal_seal.json'), validation_outputs=outputs,
            validation_targets_accessed=False, all_40_complete=True, native_score_measured=False,
            utc=datetime.now(timezone.utc).isoformat()))
        result['artifacts'][str(out/'validation_predictions_sealed.json')] = sha(out/'validation_predictions_sealed.json')
        result.update(passed=True, completed=True, history=history, updates=step,
            audit_checkpoints=checkpoints, terminal_payload=terminal, validation_outputs=outputs,
            terminal_seal_sha256=sha(out/'terminal_seal.json'),
            validation_predictions_sealed_sha256=sha(out/'validation_predictions_sealed.json'),
            hashes_unchanged=True, unused_layer0_parameters_unchanged=True, **monitor())
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        progress.close()
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-start)
    result['artifacts'][str(out/'progress.jsonl')] = sha(out/'progress.jsonl')
    write(out/'result.json', result)
    write(out/'completion.json', dict(completed=result['completed'], passed=result['passed'],
                                      result_sha256=sha(out/'result.json')))
    print(json.dumps({key: result.get(key) for key in ('schema', 'passed', 'completed', 'updates',
          'history', 'wall_s', 'maximum_sampled_mps_driver_bytes', 'error')}, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
