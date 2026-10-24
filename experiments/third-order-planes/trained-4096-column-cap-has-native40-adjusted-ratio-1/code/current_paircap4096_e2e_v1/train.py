"""One preset autonomous all-P cap4096 fit; inert without --run.

Adapt the reviewed BD/train.py one-net graph lifecycle, not its birth parameters.
Six fixed epochs, terminal only. This program NEVER reads validation truths or
scores validation outputs. Native terminal raw-times-complete-cost judgment is
separate; completing this fit is not an accuracy or deployment qualification.
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
from common import BD, HERE, NEW, P2, bindings, completed, sha, unchanged, weight_bindings, write

PARITY_SHA = '10b0cde3244a7e94cfb324ad6ae86fa01a755347b768b09815d642674ef46eeb'
NATIVE_SHA = 'fc542f5604fe2d17afc6d83fa2f3d16ff37f9cfc3b21afa8d1d144b6ac511841'
PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
MANIFEST_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
FIXED_SOURCES = {
    HERE/'model_v2.py': '24457708ce51dcffc87caad4bf01e2ee9200fba541df8f00b4c4caaa427ea9da',
    HERE/'PROTOCOL.md': '851c15aae9c0343af232145646bb8084f82618818fe2dc8e10d014ca6903de58',
    BD/'train.py': 'b0c3c6ba6eadc5a333661a51b06e1636d6e2ee385ea096cc38f6833a204f91ab',
    NEW/'joint_prefix_final_data_v1.py': '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    NEW/'tensor_clamp_min_v1.py': 'c68768a7dc17a09c6e9105817a1b9d958b2409d076dab16f5ea427d5439e9c6b',
    NEW/'payload_response_rows_v1.npz': PAYLOAD_SHA,
    NEW/'joint_prefix_final_fit_v1/manifest.json': MANIFEST_SHA,
}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
SEED, EPOCHS, BATCH, UPDATES = 9309092, 6, 4, 240


def admission(source_digest):
    """Hash/JSON-only prerequisites; never decode parity outputs or truths."""
    assert sha(__file__) == source_digest, 'Reviewed trainer source pin required'
    pins = bindings((__file__, HERE/'common.py'))
    for path, digest in FIXED_SOURCES.items():
        assert sha(path) == digest, path
        pins[str(path)] = digest
    parity = completed(HERE/'parity_v4', PARITY_SHA)
    assert parity['schema'] == 'current_paircap4096_e2e_wide_v4'
    assert parity['fidelity_and_cost_passed'] and not parity['training_executed']
    assert parity['native_result_sha256'] == NATIVE_SHA
    pins.update(parity['source_bindings'])
    # Bind the completed target-free CPU/MPS acquisitions, without decoding
    # their prediction/gradient arrays or following parity.truth_bindings.
    for stage in ('cpu_v4', 'mps_v4'):
        digest = parity['source_bindings'][str(HERE/stage/'result.json')]
        record = completed(HERE/stage, digest)
        assert record['native_result_sha256'] == NATIVE_SHA and not record['truth_accessed']
        assert record['source_bindings'][str(HERE/'model_v2.py')] == FIXED_SOURCES[HERE/'model_v2.py']
        pins.update(record['source_bindings'])
        pins[str(HERE/stage/'completion.json')] = sha(HERE/stage/'completion.json')
    native = completed(HERE/'native', NATIVE_SHA)
    assert not native['truth_accessed'] and native['cost_gate_passed']
    assert [x['rank'] for x in native['trials']] == [35, 100, 35]
    assert all(x['flops'] <= 360000000000 for x in native['trials'][:2])
    assert native['trials'][0]['output_bytes_sha256'] == native['trials'][2]['output_bytes_sha256']
    build = json.loads((HERE/'build.json').read_text())
    assert native['candidate_sha256'] == build['candidate_sha256'] == sha(HERE/'estimator.py')
    pins.update(native['source_bindings'])
    for folder in ('parity_v4', 'native'):
        for name in ('result.json', 'completion.json'):
            pins[str(HERE/folder/name)] = sha(HERE/folder/name)
    unchanged(pins)
    return pins


def resources(torch=None):
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    driver = int(torch.mps.driver_allocated_memory()) if torch is not None else 0
    assert rss < 12000000000 and driver < 16000000000, (rss, driver)
    return dict(peak_rss_bytes=rss, mps_driver_bytes=driver)


def check_domain(card):
    """Retain the qualified no-floor-event chart at all sixteen layers."""
    domain = card.variance_domain
    assert set(domain) == set(range(16)), 'Incomplete variance-domain audit'
    assert all(v['finite'] and v['minimum'] >= 1e-10 and v['below_floor_count'] == 0
               for v in domain.values()), 'Left the qualified positive-variance chart'


def save_payload(np, torch, path, P, original, keys, inactive):
    """Full deployable P payload; untouched normalizers retain original bytes."""
    values = {}
    for key, reference in original.items():
        if key in keys:
            value = P[key].detach().cpu().numpy().copy()
            assert value.dtype == np.float32 and value.shape == reference.shape
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
    return dict(path=str(path), sha256=sha(path),
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
            raw_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    assert all(os.environ.get(k) == '1' for k in BLAS)
    assert all(os.environ.get(k, '1') == '1' for k in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    launch_cwd = str(Path.cwd())
    os.chdir(NEW)  # Native receipt artifact paths are intentionally NEW-relative.
    pins = admission(args.source_sha256)
    out = HERE/'training_v1'
    assert not out.exists(), 'Exclusive one-shot; no automatic resume or overwrite'
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, completed=False, schema='current_paircap4096_e2e_training_v1',
        seed=SEED, epochs=EPOCHS, batch_nets=BATCH, planned_updates=UPDATES,
        allocated_parameters=45200, structurally_active_parameters=45134,
        source_bindings=pins, artifacts={}, validation_targets_accessed=False,
        training_targets_accessed=False, parity_result_sha256=PARITY_SHA,
        native_result_sha256=NATIVE_SHA, launch_cwd=launch_cwd,
        loss='(served final MSE + .05 * served earlier-layer MSE)/2e-8',
        optimizer=dict(name='Adam', lr_start=.0005, lr_terminal=.00005, betas=[.9, .999],
            eps=1e-8, weight_decay=0, gradient_norm_cap=5, foreach=False),
        checkpoint_selection=False, native_terminal_score_measured=False,
        maximum_sampled_mps_driver_bytes=0,
        environment={k: os.environ.get(k) for k in (*BLAS, 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'PYTORCH_ENABLE_MPS_FALLBACK')},
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    progress = (out/'progress.jsonl').open('x')
    torch_module = None

    def record(**item):
        value = dict(utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.monotonic()-start, **item)
        line = json.dumps(value, allow_nan=False)
        progress.write(line+'\n'); progress.flush()
        print(line, flush=True)

    def monitor(signum=None, frame=None):
        try:
            state = resources(torch_module)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        result['maximum_sampled_mps_driver_bytes'] = max(
            result['maximum_sampled_mps_driver_bytes'], state['mps_driver_bytes'])
        return state

    previous = signal.signal(signal.SIGALRM, monitor)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        assert torch.backends.mps.is_available()
        torch_module = torch
        torch.manual_seed(SEED)
        import model_v2 as model
        from joint_prefix_final_data_v1 import TRAIN, VALID, PKEYS, weight_path
        assert len(TRAIN) == 160 and len(VALID) == 40 and not set(TRAIN)&set(VALID)
        assert set(TRAIN)|set(VALID) == set(range(35, 235))
        assert tuple(PKEYS) == tuple(model.PKEYS)
        assert model.PAIR_CAP == 4096
        # Require qualified existing source files, not synthetic module filenames.
        for imported in tuple(sys.modules.values()):
            path = getattr(imported, '__file__', None)
            if path:
                path = Path(path).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    assert str(path) in pins and pins[str(path)] == sha(path), path
        assert 'threadpoolctl' not in sys.modules
        weight_pins, rows = weight_bindings(tuple(TRAIN)+tuple(VALID))
        assert all(Path(weight_path(r)).resolve() == Path(rows[r]['weights']).resolve() for r in rows)
        write(out/'weight_bindings.json', weight_pins)
        result['artifacts'][str(out/'weight_bindings.json')] = sha(out/'weight_bindings.json')
        result.update(train_ranks=list(TRAIN), validation_ranks=list(VALID),
            trainable_keys=list(PKEYS), Q_used=False, birth_intervention_used=False)
        manifest = json.loads((NEW/'joint_prefix_final_fit_v1/manifest.json').read_text())
        expected = {x['rank']: x['target_sha256'] for x in manifest['target_sources']}
        assert set(expected) == set(TRAIN)
        targets, target_pins = {}, {}
        for rank in TRAIN:
            path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
            assert sha(path) == expected[rank], rank
            target_pins[str(path)] = expected[rank]
            result['training_targets_accessed'] = True
            with np.load(path, allow_pickle=False) as source:
                values = source['mc_post'].copy()
            assert values.shape == (16, 1024) and np.isfinite(values).all(), rank
            targets[rank] = torch.tensor(values, device='mps', dtype=torch.float32)
        write(out/'training_targets.json', target_pins)
        result['artifacts'][str(out/'training_targets.json')] = sha(out/'training_targets.json')
        with np.load(NEW/'payload_response_rows_v1.npz', allow_pickle=False) as source:
            original = {key: source[key].copy() for key in source.files}
        assert set(PKEYS).issubset(original)
        P = {key: torch.tensor(value, device='mps', dtype=torch.float32,
                              requires_grad=key in PKEYS) for key, value in original.items()}
        parameters = [P[key] for key in PKEYS]
        assert sum(v.numel() for v in parameters) == 45200
        assert all(original[k].dtype == np.float32 for k in PKEYS)
        inactive = {key: P[key][0].detach().clone() for key in ('head', 'head_b', 'gain')}
        assert sum(v.numel() for v in inactive.values()) == 66
        optimizer = torch.optim.Adam(parameters, lr=.0005, betas=(.9, .999),
            eps=1e-8, weight_decay=0, foreach=False)
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
                    card = model.CappedCard(rank, device='mps', P=P)
                    prediction, _ = card.run(P=P, training=True)
                    check_domain(card)
                    prediction = torch.where(prediction < 0, torch.zeros_like(prediction), prediction)
                    error = prediction-targets[rank]
                    final, auxiliary = error[-1].square().mean(), error[:-1].square().mean()
                    loss = (final+.05*auxiliary)/2e-8
                    raw = float(final.detach().cpu())
                    assert math.isfinite(raw) and bool(torch.isfinite(loss)), rank
                    (loss/BATCH).backward()
                    check_domain(card)
                    assert all(card.replay_counts[k] >= 2 for k, v in card.retirement.items()
                               if v['retired']), 'Differentiable retirement checkpoint not replayed'
                    batch_raw.append(raw); metrics.append(raw)
                    monitor()
                    # No retained graph/weights from this net when the next is built.
                    del card, prediction, error, final, auxiliary, loss
                    gc.collect(); torch.mps.empty_cache()
                for key in PKEYS:
                    assert P[key].grad is not None and bool(torch.isfinite(P[key].grad).all()), key
                for key in inactive:
                    assert bool((P[key].grad[0] == 0).all()), key
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.,
                    error_if_nonfinite=True, foreach=False).detach().cpu())
                optimizer.step(); step += 1
                assert all(bool(torch.isfinite(v).all()) for v in parameters)
                assert all(torch.equal(P[k][0].detach(), v) for k, v in inactive.items())
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
        terminal_seal = dict(terminal_epoch=EPOCHS, updates=step, terminal_payload=terminal,
            source_bindings=pins, weight_bindings_sha256=sha(out/'weight_bindings.json'),
            training_targets_sha256=sha(out/'training_targets.json'),
            validation_targets_accessed=False, checkpoint_selection=False,
            utc=datetime.now(timezone.utc).isoformat())
        write(out/'terminal_seal.json', terminal_seal)
        result['artifacts'][str(out/'terminal_seal.json')] = sha(out/'terminal_seal.json')
        record(stage='terminal_sealed', payload_sha256=terminal['sha256'])
        # Reconstruct VALID inference ONLY from these exact sealed terminal bytes.
        del targets, optimizer, parameters, P
        gc.collect(); torch.mps.empty_cache()
        with np.load(terminal['path'], allow_pickle=False) as source:
            terminal_P = {key: torch.tensor(source[key].copy(), device='mps', dtype=torch.float32)
                          for key in source.files}
        assert sha(terminal['path']) == terminal['sha256']
        outputs = {}
        with torch.no_grad():
            for rank in VALID:
                card = model.CappedCard(rank, device='mps', P=terminal_P)
                prediction, _ = card.run(P=terminal_P)
                check_domain(card)
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
                gc.collect(); torch.mps.empty_cache()
                record(stage='validation_prediction_only', rank=rank, **monitor())
        assert set(map(int, outputs)) == set(VALID) and len(outputs) == 40
        assert sha(terminal['path']) == terminal['sha256']
        unchanged(pins); unchanged(weight_pins); unchanged(target_pins); unchanged(result['artifacts'])
        sealed = dict(terminal_payload=terminal, terminal_seal_sha256=sha(out/'terminal_seal.json'),
            validation_outputs=outputs, validation_targets_accessed=False,
            all_40_complete=True, native_score_measured=False, utc=datetime.now(timezone.utc).isoformat())
        write(out/'validation_predictions_sealed.json', sealed)
        result['artifacts'][str(out/'validation_predictions_sealed.json')] = sha(out/'validation_predictions_sealed.json')
        result.update(passed=True, completed=True, history=history, updates=step,
            audit_checkpoints=checkpoints, terminal_payload=terminal,
            terminal_seal_sha256=sha(out/'terminal_seal.json'), validation_outputs=outputs,
            validation_predictions_sealed_sha256=sha(out/'validation_predictions_sealed.json'),
            hashes_unchanged=True, unused_layer0_parameters_unchanged=True, **monitor())
    except BaseException:
        result['passed'] = result['completed'] = False
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        progress.close()
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-start)
    result['artifacts'][str(out/'progress.jsonl')] = sha(out/'progress.jsonl')
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
