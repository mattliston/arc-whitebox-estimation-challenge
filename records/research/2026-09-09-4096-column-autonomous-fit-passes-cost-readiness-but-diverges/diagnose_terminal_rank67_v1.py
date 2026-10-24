"""One fixed target-free native/MPS replay and physical-choice diagnosis.

Inert without --stage. No estimator arithmetic, payload, target or score edits.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
from common import BD, HERE, NEW, bindings, module, sha, unchanged, weight_bindings, write

OUT = HERE/'terminal_rank67_diagnostic_v1'
PROTOCOL = HERE/'TERMINAL_RANK67_DIAGNOSIS_V1.md'
FIXED = {
    BD/'native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    HERE/'model.py': 'cde8c3415324765e95a5a4237f796834af948c637976518ff641d18f6131693c',
    HERE/'model_v2.py': '24457708ce51dcffc87caad4bf01e2ee9200fba541df8f00b4c4caaa427ea9da',
    HERE/'training_v1/result.json': '5499433f54ed648c66716cfd9044fa7908c6024f9bb20012f6c091d56f8500d8',
    HERE/'training_v1/terminal_payload.npz': '95a98b87c952161031e85571dfa595905de79cf729af1fbc9ba92dfa24a6b628',
    HERE/'build_terminal_v1/estimator.py': '604d23a2118cfd7781ff3a51b4adc9d9125c586b051a9cc797c23a0573735a6a',
    HERE/'build_terminal_v1/build.json': '027186a362bb487a76f993b7c9b9a9c68c4be1157f7f05cd1aa8e403219ff4a1',
    HERE/'native_validation_terminal_v1/result.json': 'ad5107badc63f81630595db4f5d6058f1ee44cc2c6be44b3af38147308f1c598',
    HERE/'terminal_score_v1/result.json': '7f07b494eed79a512e716cfdd3a4134fb1845ec29bba242899056b843730abce',
}


def admission():
    pins = bindings((Path(__file__), PROTOCOL, HERE/'common.py', HERE/'native.py',
                     HERE/'check_wide_v3.py'))
    for path, digest in FIXED.items():
        assert sha(path) == digest, path
        pins[str(path)] = digest
    training = json.loads((HERE/'training_v1/result.json').read_text())
    # Authenticate the original transitive source closure, not training data.
    for path, digest in training['source_bindings'].items():
        if Path(path).suffix == '.py':
            assert sha(path) == digest, path
            pins[path] = digest
    native = json.loads((HERE/'native_validation_terminal_v1/result.json').read_text())
    failure = json.loads((HERE/'terminal_score_v1/result.json').read_text())
    assert training['passed'] and training['completed'] and native['passed']
    assert not failure['passed'] and not failure['truth_accessed']
    nr = [t for t in native['trials'] if t['rank'] == 67]
    assert len(nr) == 1 and not nr[0]['includes_initialization']
    nr = nr[0]
    mr = training['validation_outputs']['67']
    for path, digest in ((nr['output'], nr['output_sha256']), (mr['path'], mr['sha256'])):
        assert sha(path) == digest
        pins[str(path)] = digest
    weights, rows = weight_bindings((67,))
    pins.update(weights)
    return pins, nr, mr, native['initialization_flops'], rows


def once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def native_observer():
    original = module(HERE/'native.py', 'cap67_original_observer').Observer

    class Observer(original):
        def __init__(self, candidate):
            super().__init__(candidate)
            self.scores = []
            old_select = candidate.select

            def selected(score, floor):
                answer = old_select(score, floor)
                self.scores.append((score, floor))
                return answer
            candidate.select = selected

        def clear(self):
            super().clear()
            self.scores.clear()

        def materialize(self, np):
            value = super().materialize(np)
            value['scores'] = [dict(values=np.asarray(s).tolist(), floor=float(f)) for s, f in self.scores]
            value['retirement_energies'] = [np.asarray(e).tolist() for e in self.energies]
            return value
    return Observer


def run_native():
    pins, nr, mr, init, _ = admission()
    path = BD/'native_v2.py'
    old = module(path, 'cap67_preserved_native')
    text = path.read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    source = original
    a = source.index("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',")
    b = source.index('    manifest=json.loads(', a)
    replacement = ('    inputs=dict(DIAGNOSTIC_PINS)\n'
                   '    for path,digest in PINS.items():\n'
                   '        assert sha(path)==digest,path\n'
                   '        inputs[str(path)]=digest\n'
                   '    ranks=(67,)\n')
    edits = [(source[a:b], replacement),
        ('        estimator=None\n', '        observer=Observer(module)\n        estimator=None\n'),
        ('                started=time.monotonic()\n', '                observer.clear()\n                started=time.monotonic()\n'),
        ("                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
         "                diagnostic=args.out/f'call_{call:02d}_state.json'\n"
         "                write(diagnostic,observer.materialize(np))\n"
         "                result['artifacts'][str(diagnostic)]=sha(diagnostic)\n"
         "                trial['state_diagnostic']=str(diagnostic)\n"
         "                observer.clear()\n"
         "                trial['own_stored_bytes_equal']=values.tobytes()==np.load(REFERENCE['output'],allow_pickle=False).tobytes()\n"
         "                trial['own_stored_cost_equal']=trial['flops']==REFERENCE['flops']+(INIT_FLOPS if call==0 else 0)\n"
         "                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"),
        ('                guard(trial)\n',
         "                guard(trial)\n"
         "                assert trial['own_stored_bytes_equal'] and trial['own_stored_cost_equal'], 'Observed replay changed bytes or bill'\n"),
        ('        result.update(passed=True,hashes_unchanged=True,',
         "        result['schema']='cap67_target_free_native_diagnostic_v1'\n"
         "        result['score_measured']=False\n"
         '        result.update(passed=True,hashes_unchanged=True,'),
    ]
    for before, after in edits:
        source = once(source, before, after)
    inverse = source
    for before, after in reversed(edits):
        inverse = once(inverse, after, before)
    assert inverse == original
    env = dict(vars(old), __file__=str(Path(__file__).resolve()), PARENT=NEW,
        DIAGNOSTIC_PINS=pins, REFERENCE=nr, INIT_FLOPS=init, Observer=native_observer())
    saved_argv = sys.argv
    sys.argv = [str(__file__), '--source', str(HERE/'build_terminal_v1/estimator.py'),
                '--split', 'validation', '--out', str(OUT/'native')]
    try:
        exec(compile(source, str(__file__)+':preserved_native', 'exec'), env)
        return env['main']()
    finally:
        sys.argv = saved_argv


def run_mps():
    pins, _, mr, _, _ = admission()
    out = OUT/'mps'
    out.mkdir()
    result = dict(passed=False, completed=False, truth_accessed=False, score_measured=False,
        schema='cap67_target_free_mps_diagnostic_v1', source_bindings=pins, artifacts={},
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    stop, sampler = threading.Event(), None
    try:
        assert sys.executable == '/Users/matt/anaconda3/bin/python'
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        import numpy as np
        import torch
        torch.set_num_threads(1)
        assert torch.backends.mps.is_available()
        memory = dict(sampled_driver_peak_bytes=0, samples=0)

        def memory_exceeded(*_):
            raise MemoryError('Fixed MPS diagnostic memory guard')
        signal.signal(signal.SIGUSR1, memory_exceeded)

        def sample_memory():
            try:
                while not stop.is_set():
                    driver = int(torch.mps.driver_allocated_memory())
                    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    memory['sampled_driver_peak_bytes'] = max(memory['sampled_driver_peak_bytes'], driver)
                    memory['samples'] += 1
                    if driver >= 16e9 or rss >= 12e9:
                        memory['limit_exceeded'] = True
                        os.kill(os.getpid(), signal.SIGUSR1)
                        return
                    stop.wait(.05)
            except BaseException:
                memory['monitor_error'] = traceback.format_exc()
                os.kill(os.getpid(), signal.SIGUSR1)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        model = module(HERE/'model_v2.py', 'cap67_unchanged_terminal_shadow')
        with np.load(HERE/'training_v1/terminal_payload.npz', allow_pickle=False) as source:
            P = {key: torch.tensor(source[key].copy(), device='mps', dtype=torch.float32) for key in source.files}
        card = model.CappedCard(67, device='mps', P=P)
        scores = []
        old_select = card._select

        def selected(score):
            answer = old_select(score)
            scores.append(score.detach())
            return answer
        card._select = selected
        started = time.monotonic()
        with torch.no_grad():
            prediction, _ = card.run(P=P)
            served = torch.where(prediction < 0, torch.zeros_like(prediction), prediction).cpu().float().numpy().copy()
        elapsed = time.monotonic()-started
        assert served.shape == (16, 1024) and served.dtype == np.float32
        assert np.isfinite(served).all() and (served >= 0).all()
        diagnostic = card.diagnostics()
        diagnostic['scores'] = [s.cpu().numpy().tolist() for s in scores]
        write(out/'state.json', diagnostic)
        with (out/'output.npy').open('xb') as stream:
            np.save(stream, served)
        for name in ('state.json', 'output.npy'):
            result['artifacts'][str(out/name)] = sha(out/name)
        result.update(own_stored_bytes_equal=served.tobytes()==np.load(mr['path'],allow_pickle=False).tobytes(),
            raw_sha256=hashlib.sha256(served.tobytes()).hexdigest(), wall_s=elapsed,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            driver_bytes=int(torch.mps.driver_allocated_memory()))
        stop.set(); sampler.join(timeout=1)
        assert not sampler.is_alive()
        result['memory_monitor'] = dict(memory, interval_s=.05, driver_peak_is_sampled_not_continuous=True)
        assert memory['samples'] > 0 and not memory.get('limit_exceeded') and not memory.get('monitor_error')
        assert result['own_stored_bytes_equal'], 'Observed MPS replay changed stored terminal bytes'
        assert elapsed < 120 and result['peak_rss_bytes'] < 12e9 and result['driver_bytes'] < 16e9
        assert all(v['finite'] and v['below_floor_count']==0 for v in diagnostic['variance_domain'].values())
        unchanged(pins); unchanged(result['artifacts'])
        result.update(passed=True, completed=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings','artifacts')}), flush=True)
    return 0 if result['passed'] else 1


def run_judge():
    pins, _, _, _, _ = admission()
    results = {}
    for name in ('native', 'mps'):
        path = OUT/name/'result.json'
        item = json.loads(path.read_text())
        seal = json.loads((path.parent/'completion.json').read_text())
        assert item['passed'] and seal['passed'] and seal['result_sha256'] == sha(path)
        unchanged(item['source_bindings']); unchanged(item['artifacts'])
        pins[str(path)] = sha(path)
        results[name] = item
    npath = Path(results['native']['trials'][0]['state_diagnostic'])
    tpath = OUT/'mps/state.json'
    nd, td = json.loads(npath.read_text()), json.loads(tpath.read_text())
    comparer = module(HERE/'check_wide_v3.py', 'cap67_physical_identity_comparer')
    comparison = comparer.identity_comparison(td, nd)
    out = OUT/'judge'
    out.mkdir()
    write(out/'physical_identity.json', comparison)
    import numpy as np
    nv = np.load(results['native']['trials'][0]['output'], allow_pickle=False)
    tv = np.load(OUT/'mps/output.npy', allow_pickle=False)
    differences = np.abs(nv.astype(np.float64)-tv.astype(np.float64))
    result = dict(passed=True, completed=True, truth_accessed=False, score_measured=False,
        schema='cap67_target_free_diagnosis_v1', source_bindings=pins,
        artifacts={str(out/'physical_identity.json'):sha(out/'physical_identity.json')},
        own_backend_bytes_reproduced=True, native_own_complete_cost_reproduced=True,
        physical_choices_agree=comparison['passed'], first_mismatch=next(iter(comparison['mismatches']),None),
        mismatches=[dict(layer=m['layer'], type=m['type'], missing_count=len(m['missing']), extra_count=len(m['extra'])) for m in comparison['mismatches']],
        per_layer_max_difference=differences.max(1).tolist(),
        native_retirement_margins=[r['minimum_relative_margin'] for r in nd['retirement']],
        mps_retirement_margins={k:r['minimum_relative_margin'] for k,r in td['retirement'].items()},
        finished_utc=datetime.now(timezone.utc).isoformat())
    unchanged(pins)
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=True, result_sha256=sha(out/'result.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings','artifacts')}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('native','mps','judge'))
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help(); return 0
    assert sha(__file__) == args.source_sha256, 'Reviewed diagnostic source hash required'
    assert os.getpriority(os.PRIO_PROCESS,0) >= 10
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    os.chdir(NEW)
    OUT.mkdir(exist_ok=True)
    if args.stage == 'mps':
        def expired(*_):
            raise TimeoutError('Fixed MPS 120s diagnostic watchdog')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(120)
    try:
        return dict(native=run_native, mps=run_mps, judge=run_judge)[args.stage]()
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
