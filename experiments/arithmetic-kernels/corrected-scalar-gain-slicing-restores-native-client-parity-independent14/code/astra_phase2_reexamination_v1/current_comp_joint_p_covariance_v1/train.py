"""One fixed current-COMP P plus covariance fit; inert without --run."""
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
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
P1 = NEW/'current_d50_native_objective_p1_v1'
TL = NEW/'current_d50_terminal_linear_v2'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
SCHEMA = 'current_comp_joint_p_covariance_training_v1'
TRAIN = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
VALID = (37, 39, 42, 48)
SEED, LIMIT, BUDGET = 9309093, 600, 2**41
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
SHA_ARGS = ('source', 'model', 'model_protocol', 'protocol', 'watcher',
    'qualification_source', 'qualification_protocol', 'qualification_watcher',
    'qualification_watch_spec', 'qualification_result', 'qualification_watch')
PINS = {
    HERE/'DECISION.md': '814ac1f31121db92f3165ecfa0cb3253e8b66617e3e62d17316f2f288e71be98',
    HERE/'TRAIN_WATCH_SPEC.md': 'f8dcaffd8dad66934a29c4d49b771d99ab84b7e013fce944b6721594f6c963ca',
    JOINT/'train.py': '539841276180eccd44ae0da399455eb74ea1b2f3dc8627a7bcc917821dee3822',
    JOINT/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',
    P1/'worker.py': 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c',
    P1/'training_v1/result.json': 'aaaef3d5f5a3c60583af779895b8db4f769276c64cb0cc032e93755169dbab91',
    P1/'training_v1/completion.json': '6964011b2262a70e56fafd74b040432065513706c3fde5586341a09b85fd0253',
    INDEX: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def inherited_helpers():
    """Exact matched JOINT loss/inputs/lifecycle with explicit Q plumbing."""
    path = JOINT/'train.py'
    require(sha(path) == PINS[path], 'Frozen completed matched trainer source')
    source = path.read_text(); tree = ast.parse(source)
    names = ('require', 'sha', 'read', 'write', 'module', 'payload_helpers',
             'Stage', 'array_meta', 'inputs', 'adam_checks', 'run_card')
    nodes = [node for name in names for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    require(tuple(node.name for node in nodes) == names, 'Unique exact helper inventory')
    lines = source.splitlines(keepends=True)
    original = '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in nodes)
    changes = (
        ("self.folder = HERE/('train_'+args.arm+'_v1')", "self.folder = HERE/'train_v1'"),
        ('P, A, original, original_A, spec, weights, target=None, weight=1.):',
         'P, A, Q, original, original_A, spec, weights, target=None, weight=1.):'),
        ('before, before_A = check_parameters(stage, np, P, A, original, original_A)',
         'before, before_A, before_Q = check_parameters(stage, np, P, A, Q, original, original_A)'),
        ("A_before_sha256=array_meta(before_A)['raw_sha256'])",
         "A_before_sha256=array_meta(before_A)['raw_sha256'], Q_before_sha256=array_meta(before_Q)['raw_sha256'])"),
        ("stage.model.card(rank, P=P, A=A, device='mps', weights=w)",
         "stage.model.card(rank, P=P, A=A, Q=Q, device='mps', weights=w)"),
        ("require(card.A is A and all(card.P0[k] is P[k] for k in P), 'Supplied P/A identities')",
         "require(card.A is A and card.Q is Q and all(card.P0[k] is P[k] for k in P), 'Supplied P/A/Q identities')"),
        ("require(set(forward['node_head_calls']) == {str(l) for l in range(1, 15)}, 'All14 existing NODE birth calls')",
         "require(set(forward['node_head_calls']) == set(forward['covariance_head_calls']) == {str(l) for l in range(1, 15)}, 'All14 original NODE and covariance calls')"),
        ('gradient_checks(stage, np, P, A)', 'gradient_checks(stage, np, P, A, Q)'),
        ('after, after_A = check_parameters(stage, np, P, A, original, original_A)',
         'after, after_A, after_Q = check_parameters(stage, np, P, A, Q, original, original_A)'),
        ("stage.helpers['exact_values'](after, before) and after_A.tobytes() == before_A.tobytes(),",
         "stage.helpers['exact_values'](after, before) and after_A.tobytes() == before_A.tobytes() and after_Q.tobytes() == before_Q.tobytes(),"),
    )
    derived = original
    for old, new in changes:
        require(derived.count(old) == 1, 'Unique covariance trainer span: '+old)
        derived = derived.replace(old, new)
    inverse = derived
    for old, new in reversed(changes):
        require(inverse.count(new) == 1, 'Unique covariance inverse span')
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Byte-exact helper/source inverse')
    exec(compile(derived, str(path)+':covariance-training-adapter', 'exec'), globals())
    return dict(owner=str(path), owner_sha256=PINS[path], functions=list(names),
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(derived.encode()).hexdigest(), inverse_source_byte_exact=True,
        loss_and_cost_input_arithmetic_unchanged=True, original_card_retirement_unchanged=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in SHA_ARGS:
        parser.add_argument('--'+name.replace('_', '-')+'-sha256')
    return parser


def qualification(stage):
    args = stage.args
    owners = {
        HERE/'model.py': args.model_sha256,
        HERE/'MODEL_PROTOCOL.md': args.model_protocol_sha256,
        HERE/'qualification.py': args.qualification_source_sha256,
        HERE/'QUALIFICATION_PROTOCOL.md': args.qualification_protocol_sha256,
        HERE/'watch_qualification.py': args.qualification_watcher_sha256,
        HERE/'WATCH_SPEC.md': args.qualification_watch_spec_sha256,
    }
    for path, digest in owners.items():
        stage.pin(path, digest)
    records = []
    for folder, digest in ((HERE/'qualification_v1', args.qualification_result_sha256),
                           (HERE/'qualification_watch_v1', args.qualification_watch_sha256)):
        stage.pin(folder/'result.json', digest)
        record = read(folder/'result.json')
        end, final = read(folder/'completion.json'), read(folder/'finalization.json')
        end_sha, final_sha = sha(folder/'completion.json'), sha(folder/'finalization.json')
        stage.pin(folder/'completion.json', end_sha); stage.pin(folder/'finalization.json', final_sha)
        require(record['passed'] and end['passed'] and final['passed']
            and end['result_sha256'] == digest and final['completion_sha256'] == end_sha,
            'Successful fresh qualification three-seal chain')
        if folder == HERE/'qualification_v1':
            require(end['completed'] and final['completed'] and final['result_sha256'] == digest
                and final['elapsed_s'] < 180, 'Qualification worker completed within180s')
        else:
            require(final['elapsed_through_completion_io_s'] < 180 and record['sources_unchanged'],
                    'Qualification watcher completed within180s')
        for path, expected in owners.items():
            require(record['source_bindings'].get(str(path)) == expected,
                    'Exact qualification source owner: '+str(path))
        stage.inventory(record); records.append((record, end_sha, final_sha))
    (q, qe, qf), (w, _, _) = records
    require(q['schema'] == 'current_comp_joint_p_covariance_qualification_v1'
        and q['completed'] and q['hashes_unchanged'] and q['sources_unchanged']
        and q['execution_integrity_passed'] and q['full_model_calls'] == q['retired_call_count'] == 5
        and q['native_gradient_qualified'] is False and q['training_targets_accessed'] is False
        and q['validation_targets_accessed'] is False, 'Exactly five target-free qualified covariance calls')
    require(w['retired_calls'] == 5 and w['whole_owned_tree_retired'] and w['cleanup_passed']
        and w['child_result_sha256'] == args.qualification_result_sha256
        and w['child_completion_sha256'] == qe and w['child_finalization_sha256'] == qf,
        'Independent qualification retirement and exact child seals')
    stage.record['prerequisites'] = dict(qualification_result_sha256=args.qualification_result_sha256,
        qualification_watch_sha256=args.qualification_watch_sha256,
        qualification_source_sha256=args.qualification_source_sha256,
        qualification_protocol_sha256=args.qualification_protocol_sha256,
        qualification_watcher_sha256=args.qualification_watcher_sha256,
        qualification_watch_spec_sha256=args.qualification_watch_spec_sha256)


def check_parameters(stage, np, P, A, Q, original, original_A):
    current = stage.helpers['frozen'](np, P, original, stage.model)
    av, qv = (value.detach().cpu().numpy().copy() for value in (A, Q))
    require(av.dtype == qv.dtype == np.float32 and av.shape == (64, 4) and qv.shape == (14, 65)
        and np.isfinite(av).all() and np.isfinite(qv).all(), 'Exact finite FP32 A/Q surfaces')
    require(not A.requires_grad and A.grad is None and av.tobytes() == original_A.tobytes(),
            'Retained NODE A byte exact and no gradient')
    require(Q.is_leaf and Q.grad_fn is None, 'One supplied covariance leaf, never reconstructed inside a card')
    return current, av, qv


def gradient_checks(stage, np, P, A, Q):
    grads = stage.helpers['gradients'](np, P, stage.model)
    require(A.grad is None and not A.requires_grad, 'No frozen NODE A gradient')
    grad = Q.grad
    require(Q.requires_grad and grad is not None and not grad.requires_grad and grad.grad_fn is None,
            'Owned detached accumulated covariance gradient')
    value = grad.detach().cpu().numpy().copy()
    require(value.dtype == np.float32 and value.shape == (14, 65) and np.isfinite(value).all(),
            'Finite all910 covariance gradient coordinates')
    grads['covariance_Q'] = value
    return {k: array_meta(v)['raw_sha256'] for k, v in grads.items()}


def train(stage, np, torch, weights, targets, costs):
    model = stage.model
    P, A, Q = model.frozen('mps', trainable_P=True, trainable_Q=True)
    original = stage.helpers['payload_values'](np, P)
    original_A = A.detach().cpu().numpy().copy()
    original_Q = Q.detach().cpu().numpy().copy()
    require(sum(P[k].numel() for k in model.PKEYS) == 45200
        and sum(P[k][0].numel() for k in model.INACTIVE_KEYS) == 66
        and all(P[k].requires_grad and P[k].is_leaf for k in model.PKEYS), 'Exact active/inactive P surface')
    require(Q.requires_grad and Q.is_leaf and Q.numel() == 910
        and original_Q.tobytes() == np.zeros((14, 65), dtype=np.float32).tobytes(), 'Exact zero covariance initialization')
    check_parameters(stage, np, P, A, Q, original, original_A)
    stage.cstar_bytes = model._CO.coefficient_bytes(model._CO.frozen_coefficients())
    stage.record.update(initial_P={k: array_meta(v) for k, v in original.items()},
        initial_A=array_meta(original_A), initial_Q=array_meta(original_Q),
        cstar_raw_sha256=hashlib.sha256(stage.cstar_bytes).hexdigest(), seed=SEED, batch_nets=4,
        trainable_keys=list(model.PKEYS)+['covariance_Q'], trainable_parameter_count=46110,
        active_trainable_parameter_count=46044,
        loss='mean20-normalized cold d50 multiplier*(final served MSE+.05*earlier MSE)/2e-8',
        optimizer=dict(name='Adam', P_lr=[5e-4, 5e-5], Q_lr=[.01, .001], steps=15,
            betas=[.9, .999], eps=1e-8, weight_decay=0, foreach=False, group_gradient_norm_cap=5.),
        frozen_A_cstar=True, checkpoint_selection=False, covariance_H_path_not_detached=True)
    rng = np.random.Generator(np.random.PCG64DXSM(SEED))
    orders = [rng.permutation(TRAIN).tolist() for _ in range(3)]
    require(all(len(order) == 20 and set(order) == set(TRAIN) for order in orders), 'Three fixed complete permutations')
    entries = [dict(graph_index=i, rank=rank, kind='train_backward', update=i//4, epoch=i//20+1)
               for i, rank in enumerate(rank for order in orders for rank in order)]
    entries += [dict(graph_index=60+i, rank=rank, kind='validation_forward', update=None, epoch=None)
                for i, rank in enumerate(VALID)]
    stage.record['model_call_roster'] = stage.json('model_call_roster.json', dict(schema=SCHEMA,
        arm=stage.args.arm, seed=SEED, generator='PCG64DXSM', orders=orders, entries=entries))
    groups = [[P[k] for k in model.PKEYS], [Q]]
    param_groups = [dict(params=groups[0], lr=5e-4), dict(params=groups[1], lr=.01)]
    optimizer = torch.optim.Adam(param_groups, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
    require(not optimizer.state, 'Fresh empty Adam state, no resume')
    metrics = []
    for update in range(15):
        cosine = .5*(1+math.cos(math.pi*update/14))
        lrs = [5e-5+(5e-4-5e-5)*cosine, .001+(.01-.001)*cosine]
        for group, lr in zip(optimizer.param_groups, lrs):
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        for spec in entries[4*update:4*update+4]:
            rank = spec['rank']; info = targets[rank]
            stage.pin(info['path'], info['sha256'], 'target_bindings')
            stage.record['training_targets_accessed'] = True
            with np.load(info['path'], allow_pickle=False) as source:
                values = source['mc_post'].copy()
            require(np.issubdtype(values.dtype, np.floating) and values.shape == (16, 1024)
                and np.isfinite(values).all(), 'Original finite floating TRAIN mean frame')
            stage.record.setdefault('target_original_dtypes', {})[str(rank)] = str(values.dtype)
            target = torch.tensor(values, dtype=torch.float32, device='mps')
            packet = run_card(stage, np, torch, P, A, Q, original, original_A, spec,
                              weights[rank], target, costs[rank]['normalized_weight'])
            metrics.append(dict(rank=rank, update=update, raw_final_mse=packet['raw_final_mse'],
                                earlier_mean_mse=packet['earlier_mean_mse']))
            del target, values, packet
        gradient_checks(stage, np, P, A, Q)
        norms = [float(torch.nn.utils.clip_grad_norm_(group, 5., error_if_nonfinite=True,
                       foreach=False).detach().cpu()) for group in groups]
        optimizer.step(); adam_checks(torch, optimizer, groups, update+1)
        check_parameters(stage, np, P, A, Q, original, original_A)
        stage.record['updates'] = update+1
        stage.record['update_records'].append(stage.json(f'update_{update+1:02d}.json',
            dict(update=update+1, ranks=[s['rank'] for s in entries[4*update:4*update+4]],
                 learning_rates=lrs, pre_clip_gradient_norms=norms, resources=stage.guard())))
    terminal_P, _, terminal_Q = check_parameters(stage, np, P, A, Q, original, original_A)
    pm = stage.bundle(np, 'terminal_P.npz', terminal_P)
    qm = stage.save(np, 'terminal_Q.npy', terminal_Q)
    seal = stage.json('terminal_seal.json', dict(schema=SCHEMA, arm=stage.args.arm, terminal_epoch=3,
        updates=15, training_graphs=60, terminal_payload=pm, terminal_Q=qm,
        validation_predictions_started=False, validation_targets_accessed=False, no_native_gradient_claim=True,
        cstar_raw_sha256=stage.record['cstar_raw_sha256'], initial_A=stage.record['initial_A'],
        source_bindings=stage.record['source_bindings'], frozen_A_cstar=True,
        frozen_nonP_and_inactive66_byte_exact=True))
    stage.record.update(terminal_payload=pm, terminal_Q=qm, terminal_seal=seal,
        frozen_A_cstar=True, frozen_nonP_and_inactive66_byte_exact=True,
        P_optimizer_steps=15, Q_optimizer_steps=15, online_training_metrics=metrics,
        online_raw_mean=math.fsum(x['raw_final_mse'] for x in metrics)/60)
    parameter_refs = [weakref.ref(v) for v in P.values()]+[weakref.ref(A), weakref.ref(Q)]
    optimizer.zero_grad(set_to_none=True); optimizer.state.clear()
    groups.clear(); param_groups.clear(); optimizer.param_groups.clear(); P.clear()
    del optimizer, P, A, Q, group, terminal_P, terminal_Q
    stage.release()
    require(all(ref() is None for ref in parameter_refs), 'Live training leaves retired before terminal restore')
    require(sha(pm['path']) == pm['sha256'] and sha(qm['path']) == qm['sha256'], 'Sealed terminal payloads unchanged')
    with np.load(pm['path'], allow_pickle=False) as source:
        values = {k: source[k].copy() for k in source.files}
    qv = np.load(qm['path'], allow_pickle=False)
    require(set(values) == set(original) and all(array_meta(v) == pm['arrays'][k] for k, v in values.items())
        and array_meta(qv) == {k: qm[k] for k in ('dtype', 'shape', 'raw_sha256')}, 'Exact sealed payload restore')
    P = {k: torch.tensor(v, dtype=torch.float32, device='mps') for k, v in values.items()}
    A = torch.tensor(original_A, dtype=torch.float32, device='mps')
    Q = torch.tensor(qv, dtype=torch.float32, device='mps'); del values, qv
    check_parameters(stage, np, P, A, Q, original, original_A)
    outputs = {}
    for spec in entries[60:]:
        packet = run_card(stage, np, torch, P, A, Q, original, original_A, spec, weights[spec['rank']])
        outputs[str(spec['rank'])] = packet['prediction']; del packet
    parameter_refs = [weakref.ref(v) for v in P.values()]+[weakref.ref(A), weakref.ref(Q)]
    P.clear(); del P, A, Q; stage.release()
    require(all(ref() is None for ref in parameter_refs), 'Terminal inference leaves retired')
    stage.record['validation_outputs'] = outputs
    stage.record['validation_seal'] = stage.json('validation_predictions_sealed.json', dict(schema=SCHEMA,
        arm=stage.args.arm, terminal_seal_sha256=seal['sha256'], outputs=outputs,
        all_four_complete=True, validation_targets_accessed=False, native_score_measured=False))


def main():
    parser = arguments(); args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    for key in SHA_ARGS:
        value = getattr(args, key+'_sha256')
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Explicit reviewed SHA: '+key)
    args.arm = 'p_covariance'
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Whole600s P/covariance training deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    proof = inherited_helpers()
    stage = Stage(args, started)
    try:
        for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
            HERE/'model.py': args.model_sha256, HERE/'MODEL_PROTOCOL.md': args.model_protocol_sha256,
            HERE/'TRAIN_PROTOCOL.md': args.protocol_sha256, HERE/'watch_train.py': args.watcher_sha256}.items():
            stage.pin(path, digest)
        stage.record['trainer_source_proof'] = proof
        qualification(stage)
        stage.model = module(HERE/'model.py', 'joint_p_covariance_training_model')
        for path, digest in stage.model.authenticate().items():
            stage.pin(path, digest)
        stage.helpers, helper_proof = payload_helpers()
        stage.record['parameter_helper_proof'] = helper_proof
        weights, targets, costs = inputs(stage)
        require(sys.executable == '/Users/matt/anaconda3/bin/python' and sys.flags.optimize == 0
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Qualified AP/nice/BLAS/MPS launch')
        require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'All source/qualification/input admission before numerical imports')
        stage.json('admission.json', dict(source_bindings=stage.record['source_bindings'],
            input_bindings=stage.record['input_bindings'], weight_bindings=stage.record['weight_bindings'],
            target_bindings=stage.record['target_bindings'], training_targets_accessed=False,
            validation_targets_accessed=False, prerequisites=stage.record['prerequisites']))
        import numpy as np
        import torch
        stage.torch = torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified numeric runtime')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory()); require(recommended > 0, 'MPS allowance')
        fraction = min(1., 16_000_000_000/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        stage.record['mps_allocator'] = dict(configured=True, recommended_max_memory_bytes=recommended,
            fraction=fraction, effective_limit_bytes=fraction*recommended, hard_limit_bytes=16_000_000_000)
        stage.record['environment'] = dict(executable=sys.executable,
            versions=dict(numpy=np.__version__, torch=torch.__version__),
            launch={k: os.environ.get(k) for k in BLAS}, torch_threads=1, torch_interop_threads=1,
            nice=os.getpriority(os.PRIO_PROCESS, 0))
        stage.model.runtime(); train(stage, np, torch, weights, targets, costs)
        stage.finish()
    except BaseException as exc:
        stage.record.update(passed=False, completed=False, error=repr(exc), traceback=traceback.format_exc())
    finally:
        stage.close(); signal.alarm(0)
    return 0 if stage.finalized_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
