"""Inert fixed native-G0 CPU forward comparison; --run and reviewed hashes only.

No targets, gradients, updates, MPS, native calls, or fit. The previous bridge
FAIL is preserved: only its completed native-acquisition inventory is reused.
Every attempted returned output/scalar is saved before its acceptance gates.
"""
import argparse
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
NEW = HERE.parent
CO = NEW / 'current_bd_p_coadaptation_v1'
BR = NEW / 'current_d50_native_p_bridge_v1'
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
BR_JUDGE_SHA = 'b6d24664bc03a90d0129a767b35b64ee3530242d008b6cefb94bc55b9b87b4e4'
BR_RESULT_SHA = 'a8ff76e5d020a59da6626d52b79993c9e28da118abdc458ff9bbcc1c9372f4f5'
BUILD_SHA = '810a406585ce746f491ce9edcd27c2fa3d56d1d2881035b86fbe50470ea4126e'
PROTOCOL_SHA = '78a137940279a7b605d8671a35fc852d602fd1f37dc90ac4b7fb69b9afbeae07'
CASES = (('null', 0.), ('plus025', .025), ('minus025', -.025),
         ('plus050', .05), ('minus050', -.05))
RANKS = (35, 100)
SCHEMA = 'current_d50_native_g0_forward_v1'
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--model-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if any(not x or len(x) != 64 for x in (args.source_sha256, args.model_sha256, args.protocol_sha256)):
        parser.error('Execution requires all three reviewed SHA256 values')
    helper_path = CO / 'qualify.py'
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Frozen scalar/IO helper changed')
    sys.dont_write_bytecode = True
    h = load_module(helper_path, 'native_g0_original_scalar_helpers')
    folder = HERE / 'forward_v1'
    h.require(not folder.exists(), 'Exclusive fixed forward attempt')
    h.bind(Path(__file__), args.source_sha256)
    h.require(args.protocol_sha256 == PROTOCOL_SHA, 'Unchanged prospective protocol')
    folder.mkdir()
    started = time.monotonic()
    result = dict(schema=SCHEMA, passed=False, completed=False, sources_unchanged=False,
        hashes_unchanged=False, alltruth_accessed=False, training_targets_accessed=False,
        validation_targets_accessed=False, parameter_updates=0, no_fit=True,
        gradient_calls=0, native_card_calls=0, card_calls=0, MPS_used=False,
        old_native_bridge_remains_failed=True, fit_admitted=False, score_measured=False,
        source_bindings={}, source_metadata={}, input_bindings={}, dependencies={},
        weight_bindings={}, artifacts={}, artifact_bindings={}, resources={},
        disabled_nulls={}, outputs={}, secants=[], native_scalars={},
        sequence=[], started_utc=datetime.now(timezone.utc).isoformat())
    h.write(folder / 'started.json', result)
    progress = (folder / 'progress.jsonl').open('x')
    inside = False

    def record(**fields):
        line = json.dumps(dict(elapsed_s=time.monotonic() - started, **fields), allow_nan=False, sort_keys=True)
        progress.write(line + '\n'); progress.flush(); print(line, flush=True)

    def pin(path, expected=None, source=False):
        item = h.bind(path, expected)
        p = item['path']
        old = result['input_bindings'].get(p)
        h.require(old is None or old == item['sha256'], 'Conflicting input binding')
        result['input_bindings'][p] = item['sha256']
        result['dependencies'][p] = item
        if source:
            result['source_metadata'][p] = item
            result['source_bindings'][p] = item['sha256']
        return item['sha256']

    def guard(*_):
        nonlocal inside
        if inside:
            return
        inside = True
        try:
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform != 'darwin':
                rss *= 1024
            result['resources'] = dict(peak_rss_bytes=rss, elapsed_s=time.monotonic() - started)
            h.require(rss < 12_000_000_000 and time.monotonic() - started < 900, 'Fixed host/deadline guard')
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        finally:
            inside = False

    old_handler = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        h.require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'AP CPU interpreter')
        h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        h.require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        h.require(all(os.environ.get(k) == '1' for k in BLAS), 'BLAS1 required')
        h.require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No fallback')
        h.require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'Inert admission')
        for path, digest in ((Path(__file__), args.source_sha256), (HERE / 'model.py', args.model_sha256),
                             (HERE / 'PROTOCOL.md', PROTOCOL_SHA), (helper_path, HELPER_SHA),
                             (BR / 'judge.py', BR_JUDGE_SHA)):
            pin(path, digest, source=True)
        model = load_module(HERE / 'model.py', 'native_g0_cpu_model')
        for path, digest in model.authenticate().items():
            pin(path, digest, source=True)
        own = model.owner()
        cpu_path = CO / 'stage_cpu_forward_v1/result.json'
        pin(cpu_path, CPU_SHA)
        cpu = json.loads(cpu_path.read_text())
        cp = cpu_path.parent / 'completion.json'
        pin(cp)
        completion = json.loads(cp.read_text())
        h.require(cpu['passed'] and cpu['completed'] and cpu['sources_unchanged'] and cpu['no_fit']
                  and not cpu['alltruth_accessed'] and cpu['parameter_updates'] == 0
                  and completion['passed'] and completion['completed'] and completion['result_sha256'] == CPU_SHA,
                  'Original complete CPU probe source')
        h.require(cpu['versions'] == h.VERSIONS == {'numpy': '1.26.4', 'torch': '2.7.0'}
                  and cpu['environment']['versions'] == cpu['versions'], 'Original numerical frame')
        for group in ('source_metadata', 'weight_bindings', 'artifact_bindings'):
            for item in cpu[group].values():
                pin(item['path'], item['sha256'])
        result.update(probe_owner=str(cpu_path), probe_owner_sha256=CPU_SHA,
                      payloads=cpu['payloads'], cotangents=cpu['cotangents'], probe_seal_sha256=cpu['probe_seal_sha256'])
        judgment_path = BR / 'judgment_v1/result.json'
        pin(judgment_path, BR_RESULT_SHA)
        jc = judgment_path.parent / 'completion.json'
        pin(jc)
        judgment, seal = json.loads(judgment_path.read_text()), json.loads(jc.read_text())
        h.require(judgment['passed'] is False and judgment['completed'] is False
                  and seal == dict(passed=False, completed=False, result_sha256=BR_RESULT_SHA)
                  and judgment['null_bytes_and_bills_exact'] and not judgment['truth_accessed']
                  and set(judgment['native_records']) == {case for case, _ in CASES},
                  'Preserve failed bridge; complete five-case native inventory only')
        # Reauthenticate the fixed judge's sources and complete native artifacts;
        # no estimator or native verifier is executed in this CPU experiment.
        for group in ('source_bindings', 'input_bindings'):
            for path, digest in judgment[group].items():
                pin(path, digest)
        bp = BR / 'build_probes_v1/build.json'
        pin(bp, BUILD_SHA)
        build = json.loads(bp.read_text())
        h.require(build['passed'] and build['all_five_sources_complete'] and build['all_source_inverses_byte_exact']
                  and build['cpu_forward_result_sha256'] == CPU_SHA
                  and build['probe_seal_sha256'] == cpu['probe_seal_sha256'], 'Exact existing payload-only native builds')
        native = {}
        for case, amplitude in CASES:
            item = judgment['native_records'][case]
            h.require(item['complete_loop_verified'], 'Completed native loop, not scalar success')
            rp, wp = BR / f'run_{case}_v1/result.json', BR / f'watch_{case}_v1/result.json'
            pin(rp, item['result_sha256']); pin(wp, item['watch_sha256'])
            acq, watched = json.loads(rp.read_text()), json.loads(wp.read_text())
            candidate = build['candidates'][case]
            h.require(candidate['amplitude'] == amplitude and candidate['payload_path'] == cpu['payloads'][case]['path']
                      and candidate['payload_sha256'] == cpu['payloads'][case]['sha256'], 'Same sealed FP32 P case')
            h.require(acq['passed'] and acq['hashes_unchanged'] and acq['ranks'] == [35, 100]
                      and not acq['truth_accessed'] and acq['source'] == candidate['path']
                      and acq['candidate_sha256'] == candidate['sha256']
                      and [t['rank'] for t in acq['trials']] == [35, 100, 35]
                      and [t['flops'] for t in acq['trials']] == item['flops'], 'Existing native sequence/source/bills')
            h.require(watched['passed'] and watched['completed'] and watched['sources_unchanged']
                      and not watched['terminated'] and watched['child_exit_code'] == 0
                      and watched['child_result_sha256'] == item['result_sha256'], 'Exact native watchdog')
            for call, trial in enumerate(acq['trials']):
                expected = rp.parent / f'call_{call:02d}_rank{trial["rank"]}.npy'
                h.require(Path(trial['output']) == expected and acq['artifacts'][str(expected)] == trial['output_sha256'], 'Native output seal')
                pin(expected, trial['output_sha256'])
            native[case] = acq['trials'][:2]
        _, result['weight_bindings'] = h.weight_bindings()
        for path, item in result['weight_bindings'].items():
            pin(path, item['sha256'])
        h.require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'All bindings precede numerical imports')
        result['before_swap'] = h.swap_state()
        import numpy as np
        import torch
        h.require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Actual numerical frame')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        result['environment'] = dict(versions=dict(numpy=np.__version__, torch=torch.__version__), device='cpu', blas={k: os.environ[k] for k in BLAS})

        def save_attempt(path, value):
            # Unlike a passing-array helper, preserve a returned nonfinite array
            # too, before its finite/shape gate. No alternate forward is tried.
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            item = h.bind(path)
            result['artifacts'][str(path)] = item['sha256']
            result['artifact_bindings'][str(path)] = item
            return dict(item, **h.array_meta(value))

        for index, rank in enumerate(RANKS):
            cot = h.load_array(np, cpu['cotangents'][str(rank)])
            losses, native_losses = {}, {}
            result['outputs'][str(rank)], result['native_scalars'][str(rank)] = {}, {}
            for enabled, case, amplitude in ((False, 'null', 0.),) + tuple((True, c, a) for c, a in CASES):
                values = h.load_bundle(np, cpu['payloads'][case])
                P = h.torch_payload(torch, values, own.PKEYS, 'cpu')
                h.require(all(not x.requires_grad for x in P.values()), 'Forward-only P leaves')
                card = model.card(rank, P=P, enabled=enabled)
                begin = time.monotonic()
                result['card_calls'] += 1
                result['sequence'].append(dict(rank=rank, case=case, enabled=enabled))
                with torch.no_grad():
                    pred, _ = card.run()
                    value = own.served(pred).detach().cpu().numpy().copy()
                    loss = float(h.scalar(torch, own, pred, cot, torch.float64))
                label = case if enabled else 'disabled_null'
                prediction = save_attempt(folder / f'row_{rank:04d}_{label}.npy', value)
                diagnostics = card.diagnostics()
                entry = dict(rank=rank, case=case, amplitude=amplitude, enabled=enabled,
                    prediction=prediction, synthetic_loss=loss if math.isfinite(loss) else None,
                    nonfinite_scalar=None if math.isfinite(loss) else repr(loss),
                    diagnostics=diagnostics, elapsed_s=time.monotonic() - begin)
                destination = result['outputs'][str(rank)] if enabled else result['disabled_nulls']
                destination[case if enabled else str(rank)] = entry
                # This first call receipt is saved before any returned-output gate.
                h.write(folder / f'row_{rank:04d}_{label}_attempt.json', entry)
                attempt_path = folder / f'row_{rank:04d}_{label}_attempt.json'
                result['artifacts'][str(attempt_path)] = h.sha(attempt_path)
                result['artifact_bindings'][str(attempt_path)] = h.bind(attempt_path)
                record(kind='forward_saved', rank=rank, case=case, enabled=enabled, synthetic_loss=entry['synthetic_loss'])
                h.require(value.shape == (16, 1024) and value.dtype == np.float32
                          and np.isfinite(value).all() and (value >= 0).all() and math.isfinite(loss), 'Served output/scalar contract')
                own.check_replay(card)
                h.require(diagnostics['source_inverse_byte_exact'] and diagnostics['only_Cpre_Bpre_at_layer0_replaced']
                          and diagnostics['native_G0_enabled'] is enabled
                          and not diagnostics['native_G0_requires_grad'], 'Exact isolated source/constant-G0 convention')
                old = h.load_array(np, cpu['outputs'][str(rank)][case]['prediction'])
                entry['old_cpu_maximum_absolute_difference'] = float(np.max(np.abs(value.astype(np.float64) - old)))
                entry['old_cpu_selector_comparison'] = h.selector_difference(
                    cpu['outputs'][str(rank)][case]['diagnostics']['selection_cells'], diagnostics['selection_cells'])
                if not enabled:
                    entry['old_cpu_byte_exact'] = value.tobytes() == old.tobytes()
                    entry['ordered_selector_records_equal'] = diagnostics['selection_cells'] == cpu['outputs'][str(rank)]['null']['diagnostics']['selection_cells']
                    h.require(entry['old_cpu_byte_exact'] and entry['ordered_selector_records_equal'], 'Disabled exact CPU inverse')
                else:
                    trial = native[case][index]
                    native_value = np.load(trial['output'], allow_pickle=False)
                    h.require(native_value.shape == (16, 1024) and native_value.dtype == np.float32
                              and np.isfinite(native_value).all() and (native_value >= 0).all()
                              and hashlib.sha256(native_value.tobytes()).hexdigest() == trial['output_bytes_sha256'], 'Native served raw-byte seal')
                    native_loss = float(h.scalar(torch, own, torch.from_numpy(native_value), cot, torch.float64))
                    entry['native_synthetic_loss'] = native_loss
                    native_losses[case], losses[case] = native_loss, loss
                    result['native_scalars'][str(rank)][case] = native_loss
                    entry['native_layer0_byte_exact'] = value[0].tobytes() == native_value[0].tobytes()
                    entry['native_all16_maximum_absolute_difference'] = float(np.max(np.abs(value.astype(np.float64) - native_value)))
                    record(kind='native_comparison', rank=rank, case=case,
                           layer0_byte_exact=entry['native_layer0_byte_exact'], maximum=entry['native_all16_maximum_absolute_difference'])
                    h.require(math.isfinite(native_loss) and entry['native_layer0_byte_exact']
                              and entry['native_all16_maximum_absolute_difference'] <= 1e-5, 'Unchanged native L0/all16 forward gates')
                    del native_value
                h.require(all(P[key].detach().numpy().tobytes() == values[key].tobytes() for key in P), 'P not updated or mutated')
                guard()
                del pred, value, card, P, values, old
                gc.collect()
            for suffix, amplitude in (('025', .025), ('050', .05)):
                new = (losses['plus' + suffix] - losses['minus' + suffix]) / (2 * amplitude)
                reference = (native_losses['plus' + suffix] - native_losses['minus' + suffix]) / (2 * amplitude)
                relative = abs(new - reference) / max(abs(new), abs(reference), 1e-10)
                entry = dict(rank=rank, amplitude=amplitude, symmetric_secant=new,
                             native_secant=reference, relative_error=relative)
                result['secants'].append(entry)
                record(kind='secant', **entry)
                h.require(math.isfinite(new) and math.isfinite(reference) and relative <= .05, 'Unchanged native/CPU 5% scalar gate')
            del cot
            gc.collect()
        h.require(result['card_calls'] == 12 and len(result['secants']) == 4, 'Complete fixed sequence')
        h.require(not any(k in sys.modules for k in ('flopscope', 'threadpoolctl')), 'No native runtime or forbidden AP module')
        for mod in tuple(sys.modules.values()):
            path = getattr(mod, '__file__', None)
            if path and Path(path).suffix == '.py' and Path(path).is_file():
                path = Path(path).resolve()
                if path.is_relative_to(NEW):
                    h.require(str(path) in result['input_bindings'], 'Unbound local source: ' + str(path))
        for group in ('source_metadata', 'dependencies', 'weight_bindings', 'artifact_bindings'):
            h.recheck(result[group])
        result['after_swap'] = h.swap_state()
        result['swap_growth_bytes'] = result['after_swap']['used_bytes'] - result['before_swap']['used_bytes']
        h.require(result['swap_growth_bytes'] <= 2 * 1024**3, 'Swap-growth guard')
        guard()
        result.update(passed=True, completed=True, sources_unchanged=True, hashes_unchanged=True,
                      fixed_forward_bridge_passed=True, gradient_qualification_passed=False)
    except BaseException:
        result.update(passed=False, completed=False, sources_unchanged=False, hashes_unchanged=False,
                      error=traceback.format_exc())
        record(kind='failure', error=result['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        result.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        progress.close()
        result['artifacts'][str(folder / 'progress.jsonl')] = h.sha(folder / 'progress.jsonl')
        h.write(folder / 'result.json', result)
        h.write(folder / 'completion.json', dict(schema=SCHEMA, passed=result['passed'], completed=result['completed'],
                                               result_sha256=h.sha(folder / 'result.json')))
    print(json.dumps({key: result.get(key) for key in ('passed', 'completed', 'card_calls', 'elapsed_s', 'error')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
