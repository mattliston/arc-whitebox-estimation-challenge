"""One frozen TRAIN268 teacher or student gradient; no optimizer or truth edits."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC = HERE.parent
RUNS = ACC.parent
PORT = ACC / 'current_torch_rtt_port_v1'
AUDIT = ACC / 'post336935_training_audit_v1'
BRANCH = RUNS / 'astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1'


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


def distillation_loss(torch, prediction, teacher):
    assert teacher.shape == (1024,) and not teacher.requires_grad
    return ((prediction.to(torch.float64) - teacher) ** 2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=list(range(7100268, 7100272)), required=True)
    parser.add_argument('--role', choices=['teacher', 'student-grad'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-receipt', type=Path)
    args = parser.parse_args()
    student = args.role == 'student-grad'
    assert student == (args.teacher_receipt is not None)
    plan = json.loads((HERE / 'PLAN_v3.json').read_text())
    for filename, expected in plan['source_pins'].items():
        assert sha(filename) == expected, filename
    binding = json.loads((AUDIT / 'BINDING_v1.json').read_text())
    for key in ('source', 'params', 'request', 'worker'):
        assert sha(binding[key]) == binding[key + '_sha256']
    out = args.output.resolve()
    out.mkdir()
    for name in ('queue', 'claimed', 'results'):
        (out / name).mkdir()
    for name in ('TRAINER_GENERATED_ALLOW_FILE', 'TRAINER_TRUTH_PLAN',
                 'TRAINER_TRUTH_PLAN_SHA256'):
        os.environ.pop(name, None)
    allow = out / 'ALLOW.txt'
    allow.write_text(str(args.rank) + '\n')
    os.environ['TRAINER_GENERATED_ALLOW_FILE'] = str(allow)
    os.environ['TRAINER_GENERATED_MIN_N'] = '7995392'
    worker = Path(binding['worker'])
    sys.path.insert(0, str(worker.parent))
    sys.argv = [str(worker), str(out), '0']
    text = worker.read_text()
    marker = "say(event='start', run=str(RUN)"
    assert text.count(marker) == 1
    w = {'__name__': '_l7_distill_worker', '__file__': str(worker)}
    exec(compile(text[:text.index(marker)], str(worker), 'exec'), w)
    E, torch = w['E'], w['torch']
    assert E.THREADS == torch.get_num_threads() == torch.get_num_interop_threads() == 1
    adapter = load('_l7_current_configure', PORT / 'configure_private_v8.py')
    scope_module = load('_l7_encoder_scope', BRANCH / 'late_encoder_scope_v1.py')
    W3 = load('_l7_current_w3', Path(plan['w3module']))
    W3.install_worker_module(E.M, torch)
    original_install = E.M.install
    holder = {}
    cards = []

    def current_install(cd):
        runtime = E._otr_cut_runtime_v1
        module = runtime['module']
        assert module is not None
        module.install(cd, opts=runtime['spec'].get('opts'))
        answer = module.install_m(cd, original_install)
        if student:
            holder['scope'].install_card(cd)
        cards.append(cd)
        return answer

    E.M.install = current_install
    original_configure = w['configure_cutsim']

    def configure(spec):
        original_configure(spec)
        children = adapter.configure(E._otr_cut_runtime_v1['module'])
        assert children[0].RANKS == {7: 384, 10: 384, 13: 384}
        if student:
            # Fresh private module, before card installation; no live source change.
            children[0].RANKS = {7: 320, 10: 384, 13: 384}
        holder['ranks'] = dict(children[0].RANKS)

    w['configure_cutsim'] = configure
    original_card = E.LC.card

    def scalar_card(*positional, **keywords):
        cd = original_card(*positional, **keywords)
        cd.knobs['bfix_gamma'] = .119
        return cd

    E.LC.card = scalar_card
    # Generated weights are an in-memory .npy stream, not a filesystem path.
    # Pin its checked hash from the existing truth metadata; the unchanged
    # generated loader verifies actual weight bytes at the real forward.
    from otr_generated_v2 import load_analyst
    truth_path = load_analyst().truth_path(args.rank)
    with E.np.load(truth_path, allow_pickle=False) as saved:
        assert int(saved['seed']) == args.rank and int(saved['N']) >= 7995392
        weights_sha = str(saved['weights_sha256'])
    data_pins = {str(truth_path): sha(truth_path),
                 'generated_weights_seed_' + str(args.rank): weights_sha}
    teacher = None
    if student:
        teacher_record = json.loads(args.teacher_receipt.read_text())
        assert teacher_record['role'] == 'teacher' and teacher_record['rank'] == args.rank
        assert teacher_record['rank_rule'] == {'7': 384, '10': 384, '13': 384}
        assert teacher_record['data_pins'] == data_pins
        assert teacher_record['source_pins'] == plan['source_pins']
        assert teacher_record['params_sha256'] == binding['params_sha256']
        target_path = Path(teacher_record['target'])
        assert sha(target_path) == teacher_record['target_sha256']
        import numpy as np
        with np.load(target_path, allow_pickle=False) as saved:
            assert set(saved.files) == {'final_prediction', 'prediction'}
            target = saved['final_prediction'].copy()
        assert target.dtype == np.float64 and target.shape == (1024,)
        assert np.isfinite(target).all()
        teacher = torch.from_numpy(target).detach()

    source = inspect.getsource(E.run_network)
    if student:
        changes = {
            "record['hu_grads'] = M.HEADU.grad.clone()":
                "record['hu_grads'] = torch.zeros_like(M.HEADU)",
            "grads = {key: T[key].grad.clone() for key in TRAINABLE}":
                "grads = _late_scope.gradients()",
            "dial_grads = {key: DL[key].grad.clone() for key in DIAL_LR}":
                "dial_grads = {key: torch.zeros_like(DL[key]) for key in DIAL_LR}",
            "train = loss if not LOSS_W else sum(float(w) * ((prediction[int(l)].to(torch.float64) - truth[int(l)]) ** 2).mean() for l, w in LOSS_W.items())":
                "assert not LOSS_W\n            train = _distillation_loss(torch, prediction[15], _teacher)\n            record['distillation_mse'] = float(train.detach())\n            record['training_target_kind'] = 'frozen_full384_current_port_prediction'\n            record['prediction'] = prediction.detach().to(torch.float64).tolist()",
        }
        for old, new in changes.items():
            assert source.count(old) == 1, old
            source = source.replace(old, new)
    namespace = dict(E.run_network.__globals__)
    exec(compile(source, '<private L7 distillation loss>', 'exec'), namespace)
    run_private = namespace['run_network']

    def scoped_run(*positional, **keywords):
        assert not holder.get('run_started'), 'Exactly one network per process'
        holder['run_started'] = True
        namespace.update(vars(E))
        if student:
            scope = scope_module.LateEncoderScope(E, torch)
            holder['scope'] = scope
            namespace['_late_scope'] = scope
            namespace['_teacher'] = teacher
            namespace['_distillation_loss'] = distillation_loss
        return run_private(*positional, **keywords)

    E.run_network = scoped_run
    request = json.loads(Path(binding['request']).read_text())
    mode = 'grad' if student else 'eval'
    request.update(id=f'l7_distill_{args.role}_{args.rank}', rank=args.rank,
                   mode=mode, scale=1e8, keep_final=not student, keep_all=not student)
    assert not request.get('unfreeze') and not request.get('lossw') and not request.get('dump15')
    claim = out / 'claimed' / (request['id'] + '.w0.json')
    write(claim, request)
    write(out / 'SCOPE_v1.json', dict(role=args.role, rank=args.rank,
          student_layers=list(range(7, 16)) if student else [],
          original_parameters_frozen=student, original_W3_metric_frozen=True,
          training_target='frozen teacher only' if student else None,
          truth_use='Original Monte Carlo truth, reporting only for student',
          source_pins=plan['source_pins'], data_pins=data_pins,
          teacher_receipt=str(args.teacher_receipt) if student else None,
          no_optimizer=True))
    w['serve'](claim)
    result, error = w['result_path'](out, request), w['error_path'](out, request)
    assert result.exists() and not error.exists(), str(error)
    assert len(cards) == 1 and cards[0]._o18_rtt_port['pending'] is None
    if student:
        record = torch.load(result, map_location='cpu', weights_only=False)
        assert set(record['grads']) == {'late.' + key for key in scope_module.SHAPES}
        assert record['training_target_kind'] == 'frozen_full384_current_port_prediction'
        assert teacher.grad is None and not teacher.requires_grad
        summary = {key: dict(shape=list(value.shape),
                    unscaled_norm=float(torch.linalg.vector_norm(value.to(torch.float64))) / 1e8)
                   for key, value in record['grads'].items()}
        write(out / 'GRADIENT_SUMMARY_v1.json', dict(
            distillation_mse=record['distillation_mse'], true_mc_mse=record['mse'],
            gradient_scale=1e8, gradients=summary, no_optimizer=True))
    else:
        record = json.loads(result.read_text())
        import numpy as np
        target_path = out / 'TEACHER_TARGET_v1.npz'
        np.savez(target_path,
                 final_prediction=np.asarray(record['final_prediction'], dtype=np.float64),
                 prediction=np.asarray(record['prediction'], dtype=np.float64))
        write(out / 'TEACHER_v1.json', dict(role='teacher', rank=args.rank,
            rank_rule=holder['ranks'], target=str(target_path),
            target_sha256=sha(target_path), data_pins=data_pins,
            source_pins=plan['source_pins'], params_sha256=binding['params_sha256'],
            original_mc_truth_preserved=True, true_mc_mse=record['mse'],
            result=str(result), result_sha256=sha(result)))
    write(out / 'COMPLETE_v1.json', dict(role=args.role, rank=args.rank,
        rank_rule=holder['ranks'], result=str(result), result_sha256=sha(result),
        true_mc_mse=record['mse'], distillation_mse=record.get('distillation_mse'),
        seconds=record['seconds'], peak_rss=record['peak_rss'],
        parameters_updated=False, no_optimizer=True, source_pins=plan['source_pins']))
    print(json.dumps(dict(completed=args.role, true_mc_mse=record['mse'],
          distillation_mse=record.get('distillation_mse'), seconds=record['seconds'])))


if __name__ == '__main__':
    main()
