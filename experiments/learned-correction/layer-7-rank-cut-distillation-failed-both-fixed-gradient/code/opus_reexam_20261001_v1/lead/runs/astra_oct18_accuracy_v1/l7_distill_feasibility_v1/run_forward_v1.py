"""Fixed late-point no-grad forward only. Root owns launch. No fitting."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1')
RUNS = HERE.parents[2]
ACC = RUNS / 'astra_oct18_accuracy_v1'
PORT = ACC / 'current_torch_rtt_port_v1'
AUDIT = ACC / 'post336935_training_audit_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True,
                        choices=[*range(7100268, 7100272), *range(7103480, 7103488)])
    parser.add_argument('--mode', choices=['eval'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--w3module', type=Path, required=True)
    parser.add_argument('--late-arrays', type=Path)
    parser.add_argument('--full-teacher', action='store_true')
    parser.add_argument('--late-arrays-sha256')
    args = parser.parse_args()
    assert bool(args.late_arrays) == bool(args.late_arrays_sha256)
    assert not args.full_teacher or args.late_arrays is None
    if args.late_arrays is not None:
        assert sha(args.late_arrays) == args.late_arrays_sha256
    assert args.mode != 'grad' or args.rank >= 7100268, 'Training ranks only'
    binding = json.loads((AUDIT / 'BINDING_v1.json').read_text())
    for key in ('source', 'params', 'request', 'worker'):
        assert sha(binding[key]) == binding[key + '_sha256']
    plan = json.loads((HERE / 'SOURCE_PLAN_v1.json').read_text())
    for filename, expected in plan['pinned_sources'].items():
        assert sha(filename) == expected, filename
    out = args.output.resolve()
    out.mkdir()
    for name in ('queue', 'claimed', 'results'):
        (out / name).mkdir()
    for name in ('TRAINER_GENERATED_ALLOW_FILE', 'TRAINER_TRUTH_PLAN',
                 'TRAINER_TRUTH_PLAN_SHA256'):
        os.environ.pop(name, None)
    allow = out / 'ALLOW.txt'
    with allow.open('x') as stream:
        stream.write(str(args.rank) + '\n')
    os.environ['TRAINER_GENERATED_ALLOW_FILE'] = str(allow)
    os.environ['TRAINER_GENERATED_MIN_N'] = '7995392'
    worker = Path(binding['worker'])
    sys.path.insert(0, str(worker.parent))
    sys.argv = [str(worker), str(out), '0']
    text = worker.read_text()
    marker = "say(event='start', run=str(RUN)"
    assert text.count(marker) == 1
    w = {'__name__': '_late_encoder_worker', '__file__': str(worker)}
    exec(compile(text[:text.index(marker)], str(worker), 'exec'), w)
    E, torch = w['E'], w['torch']
    assert E.THREADS == torch.get_num_threads() == torch.get_num_interop_threads() == 1
    adapter = load('_late_current_configure', PORT / 'configure_private_v8.py')
    scope_module = load('_late_encoder_scope', HERE / 'late_encoder_scope_v1.py')
    W3 = load('_late_current_w3', args.w3module)
    W3.install_worker_module(E.M, torch)
    original_install = E.M.install
    holder = {}

    def current_install(cd):
        runtime = E._otr_cut_runtime_v1
        module = runtime['module']
        assert module is not None, 'Current private physics is mandatory'
        module.install(cd, opts=runtime['spec'].get('opts'))
        answer = module.install_m(cd, original_install)
        # This is outside all native-M/HEADF and current RTT install hooks.
        holder['scope'].install_card(cd)
        return answer

    E.M.install = current_install
    original_configure = w['configure_cutsim']

    def configure(spec):
        original_configure(spec)
        children = adapter.configure(E._otr_cut_runtime_v1['module'])
        assert children[0].RANKS == {7: 384, 10: 384, 13: 384}
        if not args.full_teacher:
            children[0].RANKS = {7: 320, 10: 384, 13: 384}

    w['configure_cutsim'] = configure
    original_card = E.LC.card

    def scalar_card(*positional, **keywords):
        cd = original_card(*positional, **keywords)
        cd.knobs['bfix_gamma'] = .119
        return cd

    E.LC.card = scalar_card
    source = inspect.getsource(E.run_network)
    changes = {
        "record['hu_grads'] = M.HEADU.grad.clone()":
            "record['hu_grads'] = torch.zeros_like(M.HEADU)",
        "grads = {key: T[key].grad.clone() for key in TRAINABLE}":
            "grads = _late_scope.gradients()",
        "dial_grads = {key: DL[key].grad.clone() for key in DIAL_LR}":
            "dial_grads = {key: torch.zeros_like(DL[key]) for key in DIAL_LR}",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(E.run_network.__globals__)
    exec(compile(source, '<late-only gradient reporting>', 'exec'), namespace)
    run_late = namespace['run_network']

    def scoped_run(*positional, **keywords):
        assert not holder, 'Exactly one network per process'
        arrays = None
        if args.late_arrays is not None:
            import numpy as np
            with np.load(args.late_arrays, allow_pickle=False) as saved:
                arrays = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
        scope = scope_module.LateEncoderScope(E, torch, arrays)
        holder['scope'] = scope
        namespace.update(vars(E))
        namespace['_late_scope'] = scope
        return run_late(*positional, **keywords)

    E.run_network = scoped_run
    request = json.loads(Path(binding['request']).read_text())
    request.update(id=f'late7_{args.mode}_{args.rank}', rank=args.rank,
                   mode=args.mode, scale=1e8, keep_final=args.mode == 'eval',
                   keep_all=args.mode == 'eval')
    assert not request.get('unfreeze') and not request.get('lossw')
    assert not request.get('dump15')
    claim = out / 'claimed' / (request['id'] + '.w0.json')
    write(claim, request)
    write(out / 'SCOPE_v1.json', dict(
        layers=list(range(7, 16)), trainable=['late.' + key for key in scope_module.SHAPES],
        current_private_configure=str(PORT / 'configure_private_v8.py'),
        original_parameters_frozen=True, original_W3_metric_frozen=True,
        prefix_no_graph=True, prefix_grad_mode_preserved_for_replay=True,
        native_parent=plan['native_branch'], native_parent_sha256=plan['native_branch_sha256'],
        late_arrays=str(args.late_arrays) if args.late_arrays else None,
        no_optimizer=True, full_teacher=args.full_teacher,
        rank_rule={7:384 if args.full_teacher else 320,10:384,13:384}))
    w['serve'](claim)
    result, error = w['result_path'](out, request), w['error_path'](out, request)
    assert result.exists() and not error.exists(), str(error)
    scope = holder['scope']
    assert len(scope.cards) == 1 and hasattr(scope.cards[0], '_o18_rtt_port')
    assert scope.cards[0]._o18_rtt_port['pending'] is None
    if args.mode == 'grad':
        record = torch.load(result, map_location='cpu', weights_only=False)
        assert set(record['grads']) == {'late.' + key for key in scope_module.SHAPES}
    else:
        record = json.loads(result.read_text())
    write(out / 'COMPLETE_v1.json', dict(
        rank=args.rank, mode=args.mode, result=str(result), result_sha256=sha(result),
        mse=record['mse'], seconds=record['seconds'], peak_rss=record['peak_rss'],
        current_private_sources=plan['pinned_sources'], no_optimizer=True,
        original_encoder_hashes=scope.original_hashes,
        parameters_updated=False, late_arrays=str(args.late_arrays) if args.late_arrays else None, late_arrays_sha256=args.late_arrays_sha256, forward_harness_sha256=sha(__file__)))
    print(json.dumps({'completed': args.rank, 'mode': args.mode,
                      'seconds': record['seconds'], 'mse': record['mse']}))


if __name__ == '__main__':
    main()
