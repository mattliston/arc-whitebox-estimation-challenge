"""Fixed35/100 native/Torch and gradient prerequisite. No training; inert by default.

--stage cpu/mps reads no truth. --stage judge requires the sealed native, CPU
and MPS completions, verifies target-free gates first, then opens only the two
authenticated pre-existing TRAIN truths to check the fixed 1% MSE agreement.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from common import HERE, NEW, P2, bindings, completed, sha, unchanged, weight_bindings, write

PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
TARGET_MANIFEST_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'


def resources(torch=None):
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    driver = (int(torch.mps.driver_allocated_memory())
              if torch is not None and torch.backends.mps.is_available() else 0)
    assert rss < 12e9 and driver < 16e9, (rss, driver)
    return dict(peak_rss_bytes=rss, mps_driver_bytes=driver)


def save_array(np, out, name, value, result):
    path = out/name
    with path.open('xb') as stream:
        np.save(stream, value)
    result['artifacts'][str(path)] = sha(path)
    return str(path)


def native_admission(digest):
    native = completed(HERE/'native', digest)
    build = json.loads((HERE/'build.json').read_text())
    assert native['candidate_sha256'] == build['candidate_sha256'] == sha(HERE/'estimator.py')
    assert native['truth_accessed'] is False and native['cost_gate_passed']
    assert [t['rank'] for t in native['trials']] == [35, 100, 35]
    assert all(t['flops'] <= 360000000000 for t in native['trials'][:2])
    assert native['trials'][0]['output_bytes_sha256'] == native['trials'][2]['output_bytes_sha256']
    return native


def same_cells(diagnostic, native_diagnostic):
    assert len(native_diagnostic['selections']) == 47
    offset = 0
    for layer in range(16):
        for slot in range(2 if layer == 15 else 3):
            observed = diagnostic['selections'][str((layer, slot))]
            reference = native_diagnostic['selections'][offset]
            offset += 1
            if slot < 2:
                # Alive/response row permutations do not change the selected
                # support. Birth order DOES define canonical pair slots/ties.
                assert sorted(observed) == sorted(reference), 'Native/Torch alive or response membership differs'
            else:
                assert observed == reference, 'Native/Torch ordered birth IDs differ'
    torch_retired = [diagnostic['retirement'][str(l)]['chosen'] for l in range(15)
                     if diagnostic['retirement'][str(l)]['retired']]
    assert torch_retired == [x['chosen'] for x in native_diagnostic['retirement']], 'Native/Torch retirement IDs differ'


def acquire(np, torch, model, device, native, out, result):
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    torch.set_num_threads(1)
    torch.manual_seed(9309092)
    if device == 'mps':
        assert torch.backends.mps.is_available()
    payload = NEW/'payload_response_rows_v1.npz'
    assert sha(payload) == PAYLOAD_SHA
    result['source_bindings'][str(payload)] = PAYLOAD_SHA
    # Bind imported source text before any card or numerical fixture acquisition.
    for imported in tuple(sys.modules.values()):
        path = getattr(imported, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(NEW):
            result['source_bindings'][str(Path(path).resolve())] = sha(path)
    wb, rows = weight_bindings()
    result['source_bindings'].update(wb)
    write(out/'acquisition_bindings.json', result['source_bindings'])
    result['entries'] = []
    for call, rank in enumerate((35, 100)):
        card = model.CappedCard(rank, device=device)
        with torch.no_grad():
            raw, trace = card.run(capture=True)
            prediction = raw.clamp(min=0).cpu().float().numpy().copy()
        assert np.isfinite(prediction).all() and prediction.shape == (16, 1024)
        forward_diag = card.diagnostics()
        output = save_array(np, out, f'{rank}_prediction.npy', prediction, result)
        reference = np.load(native['trials'][call]['output'], allow_pickle=False)
        error = np.abs(prediction.astype(float)-reference.astype(float))
        entry = dict(rank=rank, output=output, per_layer_max_difference=error.max(1).tolist(),
            max_mean_difference=float(error.max()), trace=trace, forward=forward_diag)
        # Save the attempted forward and diagnostics before enforcing any fidelity guard.
        result['entries'].append(entry)
        write(out/f'{rank}_forward.json', entry)
        result['artifacts'][str(out/f'{rank}_forward.json')] = sha(out/f'{rank}_forward.json')
        assert entry['max_mean_difference'] <= 1e-5
        domain = forward_diag['variance_domain']
        assert len(domain) == 16 and all(v['finite'] and v['below_floor_count'] == 0 for v in domain.values())
        native_diag = json.loads(Path(native['trials'][call]['state_diagnostic']).read_text())
        same_cells(forward_diag, native_diag)
        P = card.trainable_payload()
        differentiated, _ = card.run(P=P, training=True)
        cot = torch.tensor(np.sin(np.arange(1024, dtype=float)*.37).astype(np.float32)/1024, device=device)
        loss = (differentiated[-1].clamp(min=0)*cot).sum()
        loss.backward()
        values = []
        for key in model.PKEYS:
            assert P[key].grad is not None, key
            values.append(P[key].grad.detach().cpu().float().numpy().reshape(-1))
        gradient = np.concatenate(values)
        gradient_norm = float(np.linalg.norm(gradient.astype(float)))
        assert gradient.shape == (45200,) and np.isfinite(gradient).all() and gradient_norm > 1e-10
        assert np.count_nonzero(P['head'].grad[0].detach().cpu().numpy()) == 0
        assert float(P['head_b'].grad[0].detach().cpu()) == float(P['gain'].grad[0].detach().cpu()) == 0
        gd = card.diagnostics()
        assert gd['selections'] == forward_diag['selections']
        assert {k:v['chosen'] for k,v in gd['retirement'].items()} == {k:v['chosen'] for k,v in forward_diag['retirement'].items()}
        assert all(gd['checkpoint_retirement_visits'][k] >= 2 for k,v in gd['retirement'].items()
                   if v['retired']), 'Retirement replay was not exercised'
        # The differentiated forward at the unchanged current payload must agree.
        grad_prediction = differentiated.detach().clamp(min=0).cpu().float().numpy()
        assert np.max(np.abs(grad_prediction.astype(float)-prediction.astype(float))) <= 1e-6
        entry.update(gradient=save_array(np, out, f'{rank}_gradient.npy', gradient, result),
            gradient_norm=gradient_norm, differentiated=gd, **resources(torch))
        print(json.dumps(dict(rank=rank, device=device, max_mean_difference=entry['max_mean_difference'],
                              gradient_norm=gradient_norm, **resources(torch))), flush=True)
        del card, raw, differentiated, loss, P, prediction, grad_prediction, reference
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    result['truth_accessed'] = False


def judge(np, args, native, out, result):
    cpu = completed(HERE/'cpu', args.cpu_sha256)
    mps = completed(HERE/'mps', args.mps_sha256)
    for entry in (cpu, mps):
        assert entry['native_result_sha256'] == args.native_sha256
        for name in ('model.py', 'check_wide.py', 'PROTOCOL.md', 'IMPLEMENTATION.md', 'common.py'):
            assert entry['source_bindings'][str(HERE/name)] == sha(HERE/name)
    result['source_bindings'].update({str(HERE/'cpu/result.json'): args.cpu_sha256,
                                    str(HERE/'mps/result.json'): args.mps_sha256})
    manifest = NEW/'joint_prefix_final_fit_v1/manifest.json'
    assert sha(manifest) == TARGET_MANIFEST_SHA
    result['source_bindings'][str(manifest)] = TARGET_MANIFEST_SHA
    expected = {x['rank']: x['target_sha256'] for x in json.loads(manifest.read_text())['target_sources']}
    entries, arrays = [], {}
    for call, rank in enumerate((35, 100)):
        c, m = cpu['entries'][call], mps['entries'][call]
        assert c['rank'] == m['rank'] == rank
        gcpu, gmps = np.load(c['gradient']), np.load(m['gradient'])
        relative = float(np.linalg.norm(gmps.astype(float)-gcpu.astype(float))/np.linalg.norm(gcpu.astype(float)))
        entry = dict(rank=rank, gradient_relative_discrepancy=relative)
        entries.append(entry)
        assert relative <= .05, entry
        arrays[rank] = [np.load(native['trials'][call]['output']), np.load(c['output']), np.load(m['output'])]
    # This receipt seals ALL participating predictions, bills, derivatives and
    # source bindings before any truth-file hash or array access below.
    write(out/'pretruth_seal.json', dict(native_sha256=args.native_sha256,
        cpu_sha256=args.cpu_sha256, mps_sha256=args.mps_sha256,
        gradient_checks=entries, truth_accessed=False))
    result['pretruth_seal_sha256'] = sha(out/'pretruth_seal.json')
    result['entries'] = entries
    result['truth_bindings'] = {}
    for entry in entries:
        rank = entry['rank']
        path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        assert sha(path) == expected[rank]
        result['truth_bindings'][str(path)] = expected[rank]
        result['truth_accessed'] = True
        with np.load(path, allow_pickle=False) as src:
            truth = src['mc_post'].copy()
        assert truth.shape == (16, 1024) and np.isfinite(truth).all()
        errors = [float(np.mean((x[-1].astype(float)-truth[-1])**2)) for x in arrays[rank]]
        assert all(np.isfinite(x) and x > 0 for x in errors)
        entry.update(native_final_mse=errors[0], cpu_final_mse=errors[1], mps_final_mse=errors[2],
            cpu_native_mse_ratio=errors[1]/errors[0], mps_native_mse_ratio=errors[2]/errors[0])
        assert abs(entry['cpu_native_mse_ratio']-1) <= .01 and abs(entry['mps_native_mse_ratio']-1) <= .01
    unchanged(result['truth_bindings'])
    result['fidelity_and_cost_passed'] = True
    result['training_executed'] = False
    result['truth_frame'] = 'Existing 1e9 mc_post, development TRAIN35/100 only; no accuracy gain claim'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu', 'mps', 'judge'))
    parser.add_argument('--native-sha256')
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--mps-sha256')
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help(); return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    out = HERE/('parity' if args.stage == 'judge' else args.stage)
    assert not out.exists()
    pins = bindings((__file__, HERE/'common.py', HERE/'model.py', HERE/'PROTOCOL.md', HERE/'IMPLEMENTATION.md'))
    native = native_admission(args.native_sha256)
    pins[str(HERE/'native/result.json')] = args.native_sha256
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, schema='current_paircap4096_e2e_wide_v1', stage=args.stage,
        source_bindings=pins, artifacts={}, native_result_sha256=args.native_sha256,
        truth_accessed=False, training_executed=False, started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    try:
        import numpy as np
        if args.stage == 'judge':
            judge(np, args, native, out, result)
        else:
            import torch
            import model
            acquire(np, torch, model, args.stage, native, out, result)
        unchanged(result['source_bindings'])
        unchanged(result['artifacts'])
        result.update(passed=True, **resources())
    except BaseException:
        result['error'] = traceback.format_exc()
    result['wall_s'] = time.monotonic()-start
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(json.dumps(dict(passed=result['passed'], stage=args.stage, error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
