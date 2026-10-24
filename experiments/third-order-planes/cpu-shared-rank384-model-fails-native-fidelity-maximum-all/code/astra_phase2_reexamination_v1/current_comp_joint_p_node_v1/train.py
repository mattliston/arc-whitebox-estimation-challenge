"""One fixed 3-epoch current-COMP arm; inert without explicit --run."""
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
P1 = NEW/'current_d50_native_objective_p1_v1'
TL = NEW/'current_d50_terminal_linear_v2'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
SCHEMA = 'current_comp_joint_p_node_training_v1'
TRAIN = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
VALID = (37, 39, 42, 48)
SEED, LIMIT, BUDGET = 9309093, 600, 2**41
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
PINS = {
    HERE/'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'TRAIN_WATCH_SPEC.md': 'a891abf17620c273de433298e67cfb6e64712d2209eeca85e1803774354cbac6',
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


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.link(temporary, path)
    temporary.unlink()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--arm', choices=('p_only', 'joint'))
    for name in ('source', 'model', 'protocol', 'watcher',
                 'qualification-result', 'qualification-watch'):
        parser.add_argument('--'+name+'-sha256')
    return parser


def payload_helpers():
    """Compile ONLY four unchanged, pinned P1 parameter-checking functions."""
    path = P1/'worker.py'
    require(sha(path) == PINS[path], 'Original P1 parameter-check source')
    source = path.read_text(); tree = ast.parse(source)
    names = ('payload_values', 'exact_values', 'frozen', 'gradients')
    chosen = [node for name in names for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == name]
    require(tuple(node.name for node in chosen) == names, 'Exactly four original P1 functions')
    lines = source.splitlines(keepends=True)
    text = '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in chosen)
    env = {'c': type('Checks', (), {'require': staticmethod(require)})}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path)+':exact_checks', 'exec'), env)
    return env, dict(owner_sha256=PINS[path], functions=list(names),
        exact_source_sha256=hashlib.sha256(text.encode()).hexdigest(), unchanged_source=True)


class Stage:
    def __init__(self, args, started):
        self.args, self.start, self.torch = args, started, None
        self.folder = HERE/('train_'+args.arm+'_v1')
        self.folder.mkdir(exist_ok=False)
        self.record = dict(schema=SCHEMA, arm=args.arm, passed=False, completed=False,
            hashes_unchanged=False, sources_unchanged=False, source_bindings={}, input_bindings={},
            weight_bindings={}, target_bindings={}, artifacts={}, graph_records=[], update_records=[],
            train_ranks=list(TRAIN), validation_ranks=list(VALID), epochs=3, planned_updates=15,
            updates=0, full_model_calls=0, retired_call_count=0, training_graphs=0, validation_graphs=0,
            training_targets_accessed=False, validation_targets_accessed=False,
            no_native_gradient_claim=True, native_gradient_qualified=False, native_score_measured=False,
            information_promising=None, scientific_score_computed=False, checkpoint_selection=False,
            maximum_sampled_mps_driver_bytes=0, all_graphs_retired=False)

    def pin(self, path, expected, group=None):
        path = Path(path).resolve()
        require(path.is_relative_to(P2), 'Project-local binding')
        if group is None:
            group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
        if path.is_relative_to(TRUTH):
            require(group == 'target_bindings' and path in {TRUTH/f'truthpost_rank{r:04d}.npz' for r in TRAIN},
                    'Only fixed TRAIN truth files may be hashed or read')
        require(isinstance(expected, str) and len(expected) == 64, 'Explicit expected binding')
        old = self.record[group].get(str(path))
        require(old is None or old == expected, 'Conflicting binding')
        require(sha(path) == expected, 'Pinned bytes changed: '+str(path))
        self.record[group][str(path)] = expected
        return expected

    def inventory(self, record):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
            for path, digest in record.get(group, {}).items():
                self.pin(path, digest, 'source_bindings' if group == 'source_bindings' else 'input_bindings')

    def json(self, name, value):
        path = self.folder/name; write(path, value)
        digest = sha(path); self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def save(self, np, name, value):
        path = self.folder/name
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        digest = sha(path); self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest, **array_meta(value))

    def bundle(self, np, name, values):
        path = self.folder/name
        with path.open('xb') as stream:
            np.savez(stream, **values)
        digest = sha(path); self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest,
                    arrays={key: array_meta(value) for key, value in values.items()})

    def guard(self):
        elapsed = time.monotonic()-self.start
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        driver = int(self.torch.mps.driver_allocated_memory()) if self.torch is not None else 0
        self.record['maximum_sampled_mps_driver_bytes'] = max(self.record['maximum_sampled_mps_driver_bytes'], driver)
        require(elapsed < LIMIT and rss < 12_000_000_000 and driver < 16_000_000_000,
                'Whole600s/host12GB/MPS16GB child bounds')
        require('threadpoolctl' not in sys.modules, 'AP must never import threadpoolctl')
        return dict(elapsed_s=elapsed, peak_rss_bytes=rss, mps_driver_bytes=driver)

    def release(self):
        if self.torch is not None:
            self.torch.mps.synchronize(); gc.collect(); self.torch.mps.empty_cache()
            self.torch.mps.synchronize()
        else:
            gc.collect()
        return self.guard()

    def retired(self, spec, packet):
        index = spec['graph_index']
        require(index == self.record['retired_call_count'], 'Exactly ordered completed model call')
        item = self.json(f'graph_{index:04d}_complete.json', dict(packet, **spec,
            passed=True, completed=True, graph_retired=True, resources=self.guard()))
        self.record['graph_records'].append(item)
        self.record['retired_call_count'] += 1
        self.record['training_graphs' if spec['kind'] == 'train_backward' else 'validation_graphs'] += 1
        print(json.dumps(dict(event='retired', call=index, rank=spec['rank'], arm=self.args.arm,
            elapsed_s=time.monotonic()-self.start)), flush=True)
        self.guard()

    def finish(self):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'target_bindings', 'artifacts'):
            for path, digest in self.record[group].items():
                require(sha(path) == digest, 'Final changed bytes: '+path)
                self.guard()
        for imported in tuple(sys.modules.values()):
            name = getattr(imported, '__file__', None)
            if name:
                path = Path(name).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    require(self.record['source_bindings'].get(str(path)) == sha(path), 'Unbound local import: '+str(path))
        require(self.record['full_model_calls'] == self.record['retired_call_count'] == 64
            and self.record['training_graphs'] == 60 and self.record['validation_graphs'] == 4
            and self.record['updates'] == 15, 'Complete fixed64-call/15-update arm')
        self.record.update(resources=self.guard(), hashes_unchanged=True, sources_unchanged=True,
            passed=True, completed=True, all_graphs_retired=True, execution_integrity_passed=True)

    def close(self):
        self.record['wall_s'] = time.monotonic()-self.start
        write(self.folder/'result.json', self.record)
        result_sha = sha(self.folder/'result.json')
        write(self.folder/'completion.json', dict(schema=SCHEMA, arm=self.args.arm,
            passed=self.record['passed'], completed=self.record['completed'], result_sha256=result_sha))
        completion_sha = sha(self.folder/'completion.json')
        elapsed = time.monotonic()-self.start
        good = bool(self.record['passed'] and elapsed < LIMIT)
        write(self.folder/'finalization.json', dict(schema=SCHEMA, arm=self.args.arm,
            passed=good, completed=good, result_sha256=result_sha,
            completion_sha256=completion_sha, elapsed_s=elapsed))
        self.finalized_passed = good


def array_meta(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def qualification(stage):
    args = stage.args; records = []
    for folder, digest in ((HERE/'qualification_v1', args.qualification_result_sha256),
                           (HERE/'qualification_watch_v1', args.qualification_watch_sha256)):
        stage.pin(folder/'result.json', digest)
        record = read(folder/'result.json')
        end, final = read(folder/'completion.json'), read(folder/'finalization.json')
        end_sha = sha(folder/'completion.json'); final_sha = sha(folder/'finalization.json')
        stage.pin(folder/'completion.json', end_sha); stage.pin(folder/'finalization.json', final_sha)
        require(record['passed'] and end['passed'] and final['passed']
            and end['result_sha256'] == digest and final['completion_sha256'] == end_sha,
            'Successful qualification result/completion/finalization chain')
        if folder == HERE/'qualification_v1':
            require(end['completed'] and final['completed'] and final['result_sha256'] == digest
                and final['elapsed_s'] < 180, 'Worker qualification complete under its original deadline')
        else:
            require(final['elapsed_through_completion_io_s'] < 180 and record['sources_unchanged'],
                    'Watcher qualification complete under its original deadline')
        require(record['source_bindings'].get(str(HERE/'model.py')) == args.model_sha256,
                'Exact currently qualified joint model source')
        stage.inventory(record); records.append((record, end_sha, final_sha))
    (q, qe, qf), (w, _, _) = records
    require(q['completed'] and q['hashes_unchanged'] and q['sources_unchanged']
        and q['full_model_calls'] == q['retired_call_count'] == 5 and q['execution_integrity_passed']
        and q['native_gradient_qualified'] is False and q['training_targets_accessed'] is False
        and q['validation_targets_accessed'] is False, 'Exactly five target-free qualified calls')
    require(w['retired_calls'] == 5 and w['whole_owned_tree_retired'] and w['cleanup_passed']
        and w['child_result_sha256'] == args.qualification_result_sha256
        and w['child_completion_sha256'] == qe and w['child_finalization_sha256'] == qf,
        'Independent qualification cleanup and child seals')
    stage.record['prerequisites'] = dict(qualification_result_sha256=args.qualification_result_sha256,
        qualification_watch_sha256=args.qualification_watch_sha256)


def inputs(stage):
    p1 = read(P1/'training_v1/result.json'); end = read(P1/'training_v1/completion.json')
    require(p1['passed'] and p1['completed'] and p1['hashes_unchanged'] and p1['sources_unchanged']
        and end['passed'] and end['completed'] and end['result_sha256'] == PINS[P1/'training_v1/result.json'],
        'Original completed P1 label/cost owner')
    index = {int(x['rank']): x for x in read(INDEX)['target_sources']}
    rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    weights, targets = {}, {}
    for rank in TRAIN+VALID:
        found = [x for x in rows if x['role_rank'] == rank]
        require(len(found) == 1, 'Unique canonical weight manifest row')
        rp = (STORE/found[0]['path']).resolve()
        require(rp.is_relative_to(STORE), 'Owned STORE row path')
        stage.pin(rp, found[0]['sha256'], 'weight_bindings'); row = read(rp)
        require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Weight row identity')
        wp = rp.parent/'weights.npy'; wh = row['files']['weights.npy']['sha256']
        stage.pin(wp, wh, 'weight_bindings'); weights[rank] = dict(path=str(wp), sha256=wh)
        if rank in TRAIN:
            tp = TRUTH/f'truthpost_rank{rank:04d}.npz'; entry = index[rank]
            require(entry['weight_file'] == str(wp) and entry['target_file'] == str(tp)
                and p1['target_bindings'][str(tp)] == entry['target_sha256'], 'Same authenticated TRAIN labels')
            stage.pin(tp, entry['target_sha256'], 'target_bindings')
            targets[rank] = dict(path=str(tp), sha256=entry['target_sha256'])
    cost_meta = p1['cost_weights']; cp = P1/'training_v1/cost_weights.json'
    require(cost_meta['path'] == str(cp) and p1['artifacts'][str(cp)] == cost_meta['sha256'], 'Sealed original loss-cost table')
    stage.pin(cp, cost_meta['sha256']); source_costs = read(cp)['weights']; costs = {}
    for rank in TRAIN:
        item = source_costs[str(rank)]; rp = Path(item['capture_result']['path']).resolve()
        require(rp.is_relative_to(TL) and rp.name == 'result.json', 'Original cold capture owner')
        stage.pin(rp, item['capture_result']['sha256']); rec = read(rp)
        ep = rp.with_name('completion.json'); stage.pin(ep, sha(ep)); done = read(ep)
        bp = Path(item['bill']['path']).resolve()
        require(bp == rp.parent/'bill.json', 'Original complete bill path')
        stage.pin(bp, item['bill']['sha256']); bill = read(bp)
        require(rec['passed'] and rec['completed'] and rec['hashes_unchanged'] and rec['sources_unchanged']
            and done['passed'] and done['completed'] and done['result_sha256'] == item['capture_result']['sha256']
            and rec['rank'] == rank and rec['weights']['path'] == weights[rank]['path']
            and rec['weights']['sha256'] == weights[rank]['sha256'] and rec['truth_accessed'] is False
            and rec['cold'] and rec['includes_initialization'] and rec['initialization_subtracted'] is False
            and rec['network_calls'] == 1 and rec['complete_bill'] == bill
            and bill['cold'] and bill['includes_initialization'] and bill['observed_predict_calls'] == 1
            and type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET
            and bill['flops'] == item['flops'] and 0 < bill['peak_rss_bytes'] < 8_000_000_000
            and bill['wall_s'] < 120, 'Authenticated own cold d50 multiplier')
        multiplier = max(.1, bill['flops']/BUDGET)
        require(multiplier == item['multiplier'], 'Original multiplier arithmetic')
        costs[rank] = dict(flops=bill['flops'], multiplier=multiplier, capture_result=item['capture_result'], bill=item['bill'])
    mean = math.fsum(costs[r]['multiplier'] for r in TRAIN)/20
    for value in costs.values():
        value['normalized_weight'] = value['multiplier']/mean
    stage.record['cost_weights'] = stage.json('cost_weights.json', dict(training_mean_multiplier=mean,
        normalization_ranks=list(TRAIN), weights={str(r): x for r, x in costs.items()},
        changed_candidate_cost_predicted=False, capture_features_decoded=False))
    return weights, targets, costs


def check_parameters(stage, np, P, A, original, original_A):
    current = stage.helpers['frozen'](np, P, original, stage.model)
    a = A.detach().cpu().numpy().copy()
    require(a.dtype == np.float32 and a.shape == (64, 4) and np.isfinite(a).all(), 'Finite existing A surface')
    if stage.args.arm == 'p_only':
        require(not A.requires_grad and A.grad is None and a.tobytes() == original_A.tobytes(), 'P-only retains fixed A and no A gradient')
    return current, a


def gradient_checks(stage, np, P, A):
    grads = stage.helpers['gradients'](np, P, stage.model)
    if stage.args.arm == 'joint':
        require(A.grad is not None and not A.grad.requires_grad and A.grad.grad_fn is None, 'Detached accumulated A gradient')
        value = A.grad.detach().cpu().numpy().copy()
        require(value.dtype == np.float32 and value.shape == (64, 4) and np.isfinite(value).all(), 'Finite full A gradient')
        grads['NODE_A'] = value
    else:
        require(A.grad is None, 'No P-only A gradient')
    return {k: array_meta(v)['raw_sha256'] for k, v in grads.items()}


def run_card(stage, np, torch, P, A, original, original_A, spec, weights, target=None, weight=1.):
    before, before_A = check_parameters(stage, np, P, A, original, original_A)
    index, rank = spec['graph_index'], spec['rank']; training = spec['kind'] == 'train_backward'
    packet = dict(parameter_before={k: array_meta(v)['raw_sha256'] for k, v in before.items()},
                  A_before_sha256=array_meta(before_A)['raw_sha256'])
    stage.json(f'graph_{index:04d}_attempt.json', dict(spec, resources=stage.guard()))
    card = pred = trace = served = error = final = earlier = unweighted = loss = w = None
    refs = []; began = time.monotonic()
    try:
        stage.pin(weights['path'], weights['sha256'], 'weight_bindings')
        w = np.load(weights['path'], mmap_mode='r', allow_pickle=False)
        require(w.dtype == np.float32 and w.shape == (16, 1024, 1024), 'Original FP32 physical MLP weights')
        card = stage.model.card(rank, P=P, A=A, device='mps', weights=w)
        refs.extend((weakref.ref(card), weakref.ref(w)))
        require(card.A is A and all(card.P0[k] is P[k] for k in P), 'Supplied P/A identities')
        require(not card.coefficients.requires_grad and card.coefficients.grad is None
            and card.coefficients.detach().cpu().numpy().tobytes() == stage.cstar_bytes, 'Exact frozen cstar')
        stage.record['full_model_calls'] += 1
        with torch.set_grad_enabled(training):
            pred, trace = card.run(training=training); torch.mps.synchronize()
            served = stage.model._CO.served(pred)
            refs.extend((weakref.ref(pred), weakref.ref(served)))
            value = served.detach().cpu().numpy().copy()
            require(value.dtype == np.float32 and value.shape == (16, 1024)
                and np.isfinite(value).all() and (value >= 0).all(), 'Finite all16 served prediction')
            packet['prediction'] = stage.save(np, f'graph_{index:04d}_served.npy', value)
            forward = card.diagnostics(); stage.model._CO.check_domain(card)
            require(set(forward['node_head_calls']) == {str(l) for l in range(1, 15)}, 'All14 existing NODE birth calls')
            packet['forward_diagnostics'] = stage.json(f'graph_{index:04d}_forward.json', forward)
            if training:
                require(target is not None, 'Explicit TRAIN target')
                error = served-target
                final, earlier = error[-1].square().mean(), error[:-1].square().mean()
                unweighted = (final+.05*earlier)/2e-8
                loss = weight*unweighted/4.
                require(bool(torch.isfinite(loss).detach().cpu()), 'Finite prescribed loss')
                packet.update(raw_final_mse=float(final.detach().cpu()), earlier_mean_mse=float(earlier.detach().cpu()),
                    normalized_d50_cost_weight=weight, batch_scaled_loss=float(loss.detach().cpu()))
                loss.backward(); torch.mps.synchronize()
                packet['accumulated_gradient_raw_sha256'] = gradient_checks(stage, np, P, A)
            diagnostic = stage.model._CO.check_replay(card)
            require(diagnostic['selection_cells'] == forward['selection_cells']
                and diagnostic['variance_domain'] == forward['variance_domain'], 'All16/all47 replay exact')
            packet['diagnostics'] = stage.json(f'graph_{index:04d}_diagnostics.json', diagnostic)
            after, after_A = check_parameters(stage, np, P, A, original, original_A)
            require(stage.helpers['exact_values'](after, before) and after_A.tobytes() == before_A.tobytes(),
                    'Parameters cannot change during forward/backward')
            packet.update(all16_domains_present=True, all47_selectors_present=True, frozen_cstar=True,
                parameter_bytes_unchanged_during_call=True, observed_checkpoint_replay_byte_exact=True,
                node_H_path_not_detached=True, graph_wall_s=time.monotonic()-began)
    finally:
        card = pred = trace = served = error = final = earlier = unweighted = loss = w = None
        stage.release()
    require(len(refs) == 4 and all(ref() is None for ref in refs), 'Actual card/weight/output owners retired')
    stage.retired(spec, packet)
    return packet


def adam_checks(torch, optimizer, groups, expected_step):
    parameters = [p for group in groups for p in group]
    require(len(optimizer.state) == len(parameters), 'Every expected Adam state and no extras')
    for p in parameters:
        state = optimizer.state[p]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and float(state['step']) == expected_step
            and all(bool(torch.isfinite(x).all().detach().cpu()) for x in state.values())
            and state['exp_avg'].shape == p.shape and state['exp_avg_sq'].shape == p.shape,
            'Complete finite Adam state at exact step')


def train(stage, np, torch, weights, targets, costs):
    model = stage.model
    P, A = model.frozen('mps', trainable_P=True, trainable_A=stage.args.arm == 'joint')
    original = stage.helpers['payload_values'](np, P); original_A = A.detach().cpu().numpy().copy()
    require(sum(P[k].numel() for k in model.PKEYS) == 45200
        and sum(P[k][0].numel() for k in model.INACTIVE_KEYS) == 66
        and all(P[k].requires_grad for k in model.PKEYS), 'Exact active/inactive parameter surface')
    stage.cstar_bytes = model._CO.coefficient_bytes(model._CO.frozen_coefficients())
    stage.record.update(initial_P={k: array_meta(v) for k, v in original.items()}, initial_A=array_meta(original_A),
        cstar_raw_sha256=hashlib.sha256(stage.cstar_bytes).hexdigest(), seed=SEED, batch_nets=4,
        trainable_keys=list(model.PKEYS)+(['NODE_A'] if stage.args.arm == 'joint' else []),
        loss='mean20-normalized cold d50 multiplier*(final served MSE+.05*earlier MSE)/2e-8',
        optimizer=dict(name='Adam', P_lr=[5e-4, 5e-5], A_lr=[.01, .001], steps=15,
            betas=[.9, .999], eps=1e-8, weight_decay=0, foreach=False, group_gradient_norm_cap=5.))
    rng = np.random.Generator(np.random.PCG64DXSM(SEED))
    orders = [rng.permutation(TRAIN).tolist() for _ in range(3)]
    require(all(len(x) == 20 and set(x) == set(TRAIN) for x in orders), 'Exactly three complete fixed permutations')
    entries = [dict(graph_index=i, rank=r, kind='train_backward', update=i//4, epoch=i//20+1)
               for i, r in enumerate(r for order in orders for r in order)]
    entries += [dict(graph_index=60+i, rank=r, kind='validation_forward', update=None, epoch=None)
                for i, r in enumerate(VALID)]
    stage.record['model_call_roster'] = stage.json('model_call_roster.json', dict(schema=SCHEMA,
        arm=stage.args.arm, seed=SEED, generator='PCG64DXSM', orders=orders, entries=entries))
    groups = [[P[k] for k in model.PKEYS]] + ([[A]] if stage.args.arm == 'joint' else [])
    param_groups = [dict(params=groups[0], lr=5e-4)]
    if stage.args.arm == 'joint':
        param_groups.append(dict(params=groups[1], lr=.01))
    optimizer = torch.optim.Adam(param_groups, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
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
            packet = run_card(stage, np, torch, P, A, original, original_A, spec,
                              weights[rank], target, costs[rank]['normalized_weight'])
            metrics.append(dict(rank=rank, update=update, raw_final_mse=packet['raw_final_mse'],
                                earlier_mean_mse=packet['earlier_mean_mse']))
            del target, values, packet
        gradient_checks(stage, np, P, A)
        norms = [float(torch.nn.utils.clip_grad_norm_(group, 5., error_if_nonfinite=True,
                       foreach=False).detach().cpu()) for group in groups]
        optimizer.step(); adam_checks(torch, optimizer, groups, update+1)
        check_parameters(stage, np, P, A, original, original_A)
        stage.record['updates'] = update+1
        stage.record['update_records'].append(stage.json(f'update_{update+1:02d}.json',
            dict(update=update+1, ranks=[s['rank'] for s in entries[4*update:4*update+4]],
                 learning_rates=lrs[:len(groups)], pre_clip_gradient_norms=norms, resources=stage.guard())))
    terminal_P, terminal_A = check_parameters(stage, np, P, A, original, original_A)
    pm = stage.bundle(np, 'terminal_P.npz', terminal_P)
    am = stage.save(np, 'terminal_A.npy', terminal_A)
    seal = stage.json('terminal_seal.json', dict(schema=SCHEMA, arm=stage.args.arm,
        terminal_epoch=3, updates=15, training_graphs=60, terminal_payload=pm, terminal_A=am,
        validation_predictions_started=False, validation_targets_accessed=False, no_native_gradient_claim=True,
        cstar_raw_sha256=stage.record['cstar_raw_sha256'], source_bindings=stage.record['source_bindings'],
        frozen_nonP_and_inactive66_byte_exact=True, P_only_A_byte_exact=stage.args.arm == 'p_only'))
    stage.record.update(terminal_payload=pm, terminal_A=am, terminal_seal=seal,
        online_training_metrics=metrics, online_raw_mean=math.fsum(x['raw_final_mse'] for x in metrics)/60)
    parameter_refs = [weakref.ref(v) for v in P.values()]+[weakref.ref(A)]
    optimizer.zero_grad(set_to_none=True); optimizer.state.clear()
    groups.clear(); param_groups.clear(); optimizer.param_groups.clear(); P.clear()
    del optimizer, P, A, group, terminal_P, terminal_A
    stage.release()
    require(all(ref() is None for ref in parameter_refs), 'Live training leaves retired before terminal restore')
    require(sha(pm['path']) == pm['sha256'] and sha(am['path']) == am['sha256'], 'Sealed terminal payloads unchanged')
    with np.load(pm['path'], allow_pickle=False) as source:
        values = {k: source[k].copy() for k in source.files}
    av = np.load(am['path'], allow_pickle=False)
    require(set(values) == set(original) and all(array_meta(v) == pm['arrays'][k] for k, v in values.items())
        and array_meta(av) == {k: am[k] for k in ('dtype', 'shape', 'raw_sha256')}, 'Exact sealed payload restore')
    P = {k: torch.tensor(v, dtype=torch.float32, device='mps') for k, v in values.items()}
    A = torch.tensor(av, dtype=torch.float32, device='mps'); del values, av
    outputs = {}
    for spec in entries[60:]:
        packet = run_card(stage, np, torch, P, A, original, original_A, spec, weights[spec['rank']])
        outputs[str(spec['rank'])] = packet['prediction']; del packet
    parameter_refs = [weakref.ref(v) for v in P.values()]+[weakref.ref(A)]
    P.clear(); del P, A; stage.release()
    require(all(ref() is None for ref in parameter_refs), 'Terminal inference leaves retired')
    stage.record['validation_outputs'] = outputs
    stage.record['validation_seal'] = stage.json('validation_predictions_sealed.json', dict(schema=SCHEMA,
        arm=stage.args.arm, terminal_seal_sha256=seal['sha256'], outputs=outputs,
        all_four_complete=True, validation_targets_accessed=False, native_score_measured=False))


def main():
    parser = argument_parser(); args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.arm in ('p_only', 'joint'), 'Explicit one-arm process')
    for key in ('source', 'model', 'protocol', 'watcher', 'qualification_result', 'qualification_watch'):
        value = getattr(args, key+'_sha256')
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Explicit reviewed SHA: '+key)
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Whole600s training deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    stage = Stage(args, started)
    try:
        require(args.model_sha256 == PINS[HERE/'model.py'], 'Fixed reviewed joint model')
        for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
            HERE/'model.py': args.model_sha256, HERE/'TRAIN_PROTOCOL.md': args.protocol_sha256,
            HERE/'watch_train.py': args.watcher_sha256}.items():
            stage.pin(path, digest)
        qualification(stage)
        stage.model = module(HERE/'model.py', 'matched_joint_model')
        for path, digest in stage.model.authenticate().items():
            stage.pin(path, digest)
        stage.helpers, proof = payload_helpers(); stage.record['parameter_helper_proof'] = proof
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
        stage.record['environment'] = dict(executable=sys.executable, versions=dict(numpy=np.__version__, torch=torch.__version__),
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
