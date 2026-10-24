"""Fresh five-call proposal integrity or one fixed terminal160+40 pass.

Inert without --run. CS remains FAILED; this never qualifies a derivative.
"""
import ast
import gc
import hashlib
import math
from pathlib import Path
import signal
import sys
import time
import traceback
import weakref

import common as c


def source_helpers():
    """Keep P1 I/O, retirement, roster and exact proposal-loss arithmetic."""
    path = c.P1/'worker.py'
    c.require(c.sha(path) == c.P1_WORKER_SHA, 'Exact P1 numerical convention owner')
    source = path.read_text(); lines = source.splitlines(keepends=True)
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    names = ('payload_values', 'exact_values', 'release', 'roster')
    original = '\n\n'.join(''.join(lines[functions[k].lineno-1:functions[k].end_lineno]) for k in names)
    changes = [
        ("(('parent_forward', 100),\n                       ('synthetic_backward', 35)",
         "(('parent_forward', 35), ('parent_forward', 100),\n                       ('synthetic_backward', 35)"),
        ('current_d50_native_objective_p1_roster_v1', 'current_comp_cstar_proposal_roster_v1')]
    derived = original
    for old, new in changes:
        c.require(derived.count(old) == 1, 'Unique exact roster splice')
        derived = derived.replace(old, new, 1)
    inverse = derived
    for old, new in reversed(changes):
        c.require(inverse.count(new) == 1, 'Unique roster inverse')
        inverse = inverse.replace(new, old, 1)
    c.require(inverse == original, 'P1 roster/helpers source byte inverse')
    first = '                served = stage.model.served(prediction)\n'
    last = '                del unweighted\n'
    c.require(source.count(first) == source.count(last) == 1, 'Unique exact P1 loss block')
    block = source[source.index(first):source.index(last)+len(last)]
    revised = block.replace('stage.model.served(prediction)', 'stage.model._CO.served(prediction)', 1)
    body = ''.join('    '+line[16:] for line in revised.splitlines(keepends=True))
    recovered = ''.join(' '*16+line[4:] for line in body.splitlines(keepends=True))
    c.require(recovered.replace('stage.model._CO.served(prediction)', 'stage.model.served(prediction)', 1) == block,
              'Unchanged P1 loss operation order; only CO facade/indentation')
    loss = 'def proposal_loss(stage, prediction, target, weight, packet):\n'+body+'    return loss\n'
    env = dict(globals())
    exec(compile(derived+'\n\n'+loss, str(path)+':exact-helper-subset', 'exec'), env)
    proof = dict(owner_sha256=c.P1_WORKER_SHA, methods=list(names),
        original_subset_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_subset_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        loss_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        inverse_source_byte_exact=True, exact_loss_arithmetic=True,
        roster_changes=['fresh35 reference added', 'schema owner'], no_old_worker_stage_executed=True)
    return {key: env[key] for key in (*names, 'proposal_loss')}, proof


def array(tensor):
    return tensor.detach().cpu().contiguous().numpy().copy()


def rawsha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def frozen(stage, np, state):
    current = stage.helpers['payload_values'](np, state['P'])
    c.require(stage.helpers['exact_values'](current, stage.initial_P)
        and all(not v.requires_grad and v.grad is None for v in state['P'].values()), 'Every P/normalizer/inactive byte frozen')
    c.require(not state['A'].requires_grad and state['A'].grad is None
        and array(state['A']).tobytes() == stage.initial_A.tobytes(), 'Exact frozen NODE A bytes/gradient')
    c0 = stage.model.initial_coefficients('mps')
    c.require(array(c0).tobytes() == stage.initial_cstar.tobytes(), 'Original exported c0 remains frozen')
    del c0
    if 'delta' in state:
        delta = state['delta']
        c.require(delta.is_leaf and delta.requires_grad and delta.dtype == stage.torch.float32
            and delta.shape == (14, 4) and bool(stage.torch.isfinite(delta).all().detach().cpu()), 'Only actual delta56 leaf')
        exported = stage.model.coefficients(delta.detach())
    else:
        exported = state['cstar']
        c.require(not exported.requires_grad and exported.grad is None, 'Reloaded nontrainable terminal c')
    value = array(exported)
    c.require(value.dtype == np.float32 and value.shape == (15, 4) and np.isfinite(value).all()
        and (np.abs(value) < 1).all() and value[0].tobytes() == stage.initial_cstar[0].tobytes(), 'Exact row0 and strict stored coefficient domain')
    if 'cstar' in state:
        c.require(value.tobytes() == stage.terminal_cstar.tobytes(), 'Every terminal forward uses the sealed actual c')
    del exported
    return value


def gradient(np, delta, nontrivial=False):
    grad = delta.grad
    c.require(grad is not None and not grad.requires_grad and grad.grad_fn is None, 'Owned detached delta gradient')
    result = array(grad)
    c.require(result.dtype == np.float32 and result.shape == (14, 4) and np.isfinite(result).all(), 'All56 finite gradient coordinates')
    if nontrivial:
        c.require(float(np.linalg.norm(result.astype(np.float64))) > 0., 'Nonzero full56 proposal gradient')
    return result


def physical_hashes(card):
    fields = dict(weights=card.w, physical_weights=card.ws, fixed_features=card.fixed, coefficients=card.coefficients)
    for name in ('c0', 'headroom'):
        if hasattr(card, name):
            fields[name] = getattr(card, name)
    return {name: rawsha(array(value)) for name, value in fields.items()}


def run_card(stage, np, torch, state, spec, *, cotangent=None, zero_lr=False, reference=None, repeat=None):
    """One actual card; only detached copies and metadata survive retirement."""
    index, rank, kind = spec['graph_index'], spec['rank'], spec['kind']
    backward = kind in ('synthetic_backward', 'train_backward')
    started = time.monotonic(); exported = frozen(stage, np, state)
    before_delta = array(state['delta']) if 'delta' in state else None
    stage.json(f'graph_{index:04d}_attempt.json', dict(spec, elapsed_s=started-stage.start,
        coefficients_raw_sha256=rawsha(exported), delta_raw_sha256=None if before_delta is None else rawsha(before_delta)))
    card = raw = served = loss = trace = weights = target = None
    refs, packet, grad_np = [], dict(spec), None
    try:
        weights = np.load(stage.weights[rank]['path'], mmap_mode='r', allow_pickle=False)
        c.require(weights.dtype == np.float32 and weights.shape == (16, 1024, 1024)
            and np.isfinite(weights).all(), 'Canonical complete FP32 MLP weights')
        if kind == 'train_backward':
            target_path = c.TRUTH/f'truthpost_rank{rank:04d}.npz'
            stage.pin(target_path, stage.targets[rank], 'target_bindings')
            stage.record['training_targets_accessed'] = True
            with np.load(target_path, allow_pickle=False) as source:
                values = source['mc_post'].copy()
            c.require(values.shape == (16, 1024) and np.issubdtype(values.dtype, np.floating)
                and np.isfinite(values).all(), 'Original finite floating TRAIN1e9 post means')
            packet.update(target_original_dtype=str(values.dtype), target_loss_dtype='float32',
                target_sha256=stage.targets[rank], metrics_are_FP32_target_proposal_only=True)
            target = torch.tensor(values, device='mps', dtype=torch.float32)
            del values
        with torch.set_grad_enabled(backward):
            if kind == 'parent_forward':
                card = stage.model.reference_card(rank, P=state['P'], A=state['A'], device='mps', weights=weights)
            elif backward:
                card = stage.model.card(rank, P=state['P'], A=state['A'], delta=state['delta'], device='mps', weights=weights)
                c.require(card.delta is state['delta'] and card.coefficients.requires_grad
                    and not card.coefficients.is_leaf and card.coefficients.grad_fn is not None,
                    'Actual delta and supplied nonleaf coefficient graph identity')
            else:
                card = stage.model.card_from_coefficients(rank, P=state['P'], A=state['A'], cstar=state['cstar'], device='mps', weights=weights)
                c.require(card.coefficients is state['cstar'], 'Supplied sealed terminal c identity')
            c.require(card.dtype == torch.float32 and card.n == 1024 and card.depth == 16
                and card.A is state['A'] and all(card.P0[k] is v for k, v in state['P'].items()), 'Original MPS physical frame and frozen parent identities')
            before = physical_hashes(card)
            c.require(before['weights'] == rawsha(weights) and before['coefficients'] == rawsha(exported), 'Exact input and actual exported coefficient bytes')
            raw, trace = card.run(training=backward)
            c.require(trace == [], 'No retained capture or extra model call')
            torch.mps.synchronize()
            packet['forward_elapsed_s'] = time.monotonic()-started
            served = stage.model._CO.served(raw)
            raw_np, served_np = array(raw), array(served)
            c.require(raw_np.dtype == served_np.dtype == np.float32 and raw_np.shape == served_np.shape == (16, 1024)
                and np.isfinite(raw_np).all() and np.isfinite(served_np).all() and (served_np >= 0).all(), 'Finite all16 raw/served outputs')
            forward = card.diagnostics()
            stage.model._CO.check_domain(card)
            if reference is not None:
                c.require(raw_np.tobytes() == reference['raw'].tobytes()
                    and served_np.tobytes() == reference['served'].tobytes()
                    and all(forward[k] == reference['diagnostics'][k] for k in ('variance_domain', 'selection_cells')),
                    'Fresh unmodified same-backend null raw/served/domain/selector bytes')
                packet['same_backend_null_byte_exact'] = True
            if kind in ('parent_forward', 'synthetic_backward'):
                loss = stage.h.scalar(torch, stage.model._CO, raw, cotangent, card.dtype)
                packet.update(synthetic_scalar=float(loss.detach().cpu()), synthetic_scalar_raw_sha256=rawsha(array(loss)))
            elif kind == 'train_backward':
                loss = stage.helpers['proposal_loss'](stage, raw, target, stage.costs[rank]['normalized_weight'], packet)
            if backward:
                c.require(bool(torch.isfinite(loss).detach().cpu()), 'Finite original proposal scalar')
                loss.backward(); torch.mps.synchronize()
                grad_np = gradient(np, state['delta'], nontrivial=kind == 'synthetic_backward')
                packet['gradient'] = stage.save(np, f'graph_{index:04d}_gradient.npy', grad_np)
                packet['gradient_is_batch_accumulated'] = kind == 'train_backward'
            diagnostic = stage.model._CO.check_replay(card)
            c.require(set(diagnostic['node_head_calls']) == {str(l) for l in range(1, 15)}
                and all(v > 0 for v in diagnostic['node_head_calls'].values())
                and all(diagnostic[k] == forward[k] for k in ('variance_domain', 'selection_cells')), 'All14 NODE layers and every observed replay unchanged')
            c.require(before == physical_hashes(card) and frozen(stage, np, state).tobytes() == exported.tobytes(), 'Card/backward changed no input, constant or coefficient')
            if reference is not None:
                c.require(packet['synthetic_scalar_raw_sha256'] == reference['scalar_raw_sha256'], 'Original reference scalar bytes')
            if repeat is not None:
                c.require(raw_np.tobytes() == repeat['raw'].tobytes() and served_np.tobytes() == repeat['served'].tobytes()
                    and grad_np.tobytes() == repeat['gradient'].tobytes() and diagnostic == repeat['diagnostics']
                    and packet['synthetic_scalar_raw_sha256'] == repeat['scalar_raw_sha256'], 'Repeat35 raw/served/scalar/full56 gradient/diagnostic byte exact')
                packet.update(repeat_prediction_byte_exact=True, repeat_gradient_byte_exact=True)
            packet.update(raw=stage.save(np, f'graph_{index:04d}_raw.npy', raw_np),
                prediction=stage.save(np, f'graph_{index:04d}_served.npy', served_np),
                diagnostics=stage.json(f'graph_{index:04d}_diagnostics.json', diagnostic),
                forward_diagnostics=stage.json(f'graph_{index:04d}_forward.json', forward),
                frozen_hashes=before, all16_domains_qualified=True, all47_selectors_qualified=True,
                observed_checkpoint_replay_byte_exact=True, frozen_P_A_c0=True, live_resources=stage.guard())
            if zero_lr:
                optimizer = state['optimizer']
                c.require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]['lr'] == 0.
                    and len(optimizer.param_groups[0]['params']) == 1
                    and optimizer.param_groups[0]['params'][0] is state['delta'], 'Actual sole56 zero-LR Adam owner')
                optimizer.step()
                c.require(len(optimizer.state) == 1 and state['delta'] in optimizer.state, 'Actual one-leaf Adam state allocated')
                allocation = optimizer.state[state['delta']]
                c.require(set(allocation) == {'step', 'exp_avg', 'exp_avg_sq'}
                    and all(bool(torch.isfinite(v).all().detach().cpu()) for v in allocation.values())
                    and allocation['exp_avg'].shape == allocation['exp_avg_sq'].shape == (14, 4), 'Finite actual step and112 moment entries')
                actual_step = float(allocation['step'].detach().cpu())
                c.require(actual_step == index-1, 'Exactly three fresh zero-LR state updates in roster order')
                packet['zero_lr_Adam'] = dict(step=actual_step,
                    state={k: stage.h.array_meta(array(v)) for k, v in allocation.items()}, actual_optimizer_step_executed=True)
                c.require(array(state['delta']).tobytes() == before_delta.tobytes()
                    and frozen(stage, np, state).tobytes() == stage.initial_cstar.tobytes(), 'Zero-LR delta/c0 byte exact')
                optimizer.zero_grad(set_to_none=True)
                packet['zero_lr_delta_byte_exact'] = True
                del allocation, optimizer
            refs = [weakref.ref(v) for v in (card, raw, served, card.w, card.ws, card.fixed, weights)]
            if loss is not None:
                refs.append(weakref.ref(loss))
            if target is not None:
                refs.append(weakref.ref(target))
            if kind != 'validation_forward':
                refs.append(weakref.ref(card.coefficients))
            for name in ('c0', 'headroom'):
                if hasattr(card, name):
                    refs.append(weakref.ref(getattr(card, name)))
            answer = dict(raw=raw_np, served=served_np, gradient=grad_np, diagnostics=diagnostic,
                scalar_raw_sha256=packet.get('synthetic_scalar_raw_sha256'))
            stage.json(f'graph_{index:04d}_checked.json', packet)
    finally:
        card = raw = served = loss = trace = weights = target = None
        stage.helpers['release'](stage)
    c.require(all(ref() is None for ref in refs), 'Per-call card/output/nonleaf/weight/target owners retired; persistent parents/delta excluded')
    packet.update(graph_retired=True, retired_elapsed_s=time.monotonic()-started,
        persistent_owner_scope='Frozen P/A and optimizer delta until their stage-end retirement')
    stage.graph_complete(index, packet)
    return packet, answer


def qualify(stage, np, torch, state, entries):
    cotangents = {rank: np.random.Generator(np.random.PCG64DXSM(9309098+rank)).standard_normal((16, 1024)).astype(np.float32)
                  for rank in (35, 100)}
    stage.record['cotangents'] = {str(rank): stage.save(np, f'cotangent_{rank:04d}.npy', value) for rank, value in cotangents.items()}
    refs, first, checks = {}, None, []
    for spec in entries:
        state['optimizer'].zero_grad(set_to_none=True)
        packet, answer = run_card(stage, np, torch, state, spec, cotangent=cotangents[spec['rank']],
            zero_lr=spec['kind'] == 'synthetic_backward', reference=refs.get(spec['rank']),
            repeat=first if spec['graph_index'] == 4 else None)
        if spec['kind'] == 'parent_forward':
            refs[spec['rank']] = answer
        else:
            if first is None:
                first = answer
            checks.append(packet)
    c.require(array(state['delta']).tobytes() == np.zeros((14, 4), np.float32).tobytes()
        and frozen(stage, np, state).tobytes() == stage.initial_cstar.tobytes(), 'All five calls preserve computed exact null')
    stage.record.update(execution_integrity_passed=True, checks=checks, reference_forwards=2,
        gradient_calls=3, zero_lr_Adam_calls=3, parameter_updates=0, optimizer_updates=0,
        same_backend_null_byte_exact=True, first_repeat35_prediction_byte_exact=True,
        first_repeat35_gradient_byte_exact=True, zero_lr_delta_byte_exact=True,
        all16_domains_qualified=True, all47_selectors_qualified=True, frozen_P_A_c0=True)


def retire_optimizer(stage, state):
    optimizer = state.pop('optimizer'); delta = state.pop('delta')
    refs = [weakref.ref(optimizer), weakref.ref(delta)]
    if delta.grad is not None:
        refs.append(weakref.ref(delta.grad))
    refs.extend(weakref.ref(v) for allocation in optimizer.state.values() for v in allocation.values())
    optimizer.zero_grad(set_to_none=True); optimizer.state.clear()
    for group in optimizer.param_groups:
        group['params'].clear()
    del group, optimizer, delta
    stage.helpers['release'](stage)
    c.require(all(ref() is None for ref in refs), 'Actual optimizer/delta/gradient/moment owners retired')


def train(stage, np, torch, state, entries):
    metrics, updates = [], []
    stage.record.update(epochs=1, planned_updates=40, batch_nets=4, seed=c.SEED,
        optimizer=dict(name='Adam', coordinates='relative_headroom_delta', lr_start=.025, lr_terminal=.0025,
            betas=[.9, .999], eps=1e-8, weight_decay=0, foreach=False, gradient_norm_cap=5.),
        loss='normalized old cold d50 weight*(FP32-target served final MSE+.05*earlier mean MSE)/2e-8',
        old_d50_weights_are_proposal_constants=True, native_cost_derivative_claimed=False)
    for update in range(40):
        lr = .0025+.5*(.025-.0025)*(1+math.cos(math.pi*update/39))
        state['optimizer'].param_groups[0]['lr'] = lr
        state['optimizer'].zero_grad(set_to_none=True)
        for spec in entries[4*update:4*update+4]:
            packet, answer = run_card(stage, np, torch, state, spec)
            metrics.append({k: packet[k] for k in ('rank', 'raw_final_mse', 'earlier_mean_mse',
                'unweighted_surrogate_loss', 'weighted_surrogate_loss', 'normalized_d50_cost_weight', 'target_original_dtype')})
            del answer
        gradient(np, state['delta'])
        norm = float(torch.nn.utils.clip_grad_norm_([state['delta']], 5., error_if_nonfinite=True, foreach=False).detach().cpu())
        state['optimizer'].step()
        exported = frozen(stage, np, state)
        stage.record['parameter_updates'] = stage.record['optimizer_updates'] = update+1
        updates.append(stage.json(f'update_{update+1:02d}.json', dict(update=update+1, learning_rate=lr,
            ranks=[x['rank'] for x in entries[4*update:4*update+4]], pre_clip_gradient_norm=norm,
            delta_raw_sha256=rawsha(array(state['delta'])), coefficient_raw_sha256=rawsha(exported),
            coefficient_abs_max=float(np.abs(exported).max()), resources=stage.guard())))
    c.require(len(metrics) == 160 and stage.record['optimizer_updates'] == 40
        and len(stage.record['graph_records']) == 160, 'Exactly fixed160 TRAIN/40 optimizer updates')
    terminal_delta = stage.save(np, 'terminal_delta.npy', array(state['delta']))
    terminal_c = stage.save(np, 'terminal_coefficients.npy', frozen(stage, np, state))
    seal = stage.json('terminal_seal.json', dict(terminal_epoch=1, updates=40, training_graphs=160,
        terminal_delta=terminal_delta, terminal_coefficients=terminal_c,
        parent_candidate_sha256=stage.model.COMP_SHA, model_source_proof=stage.model.source_proof(),
        initial_P=stage.record['initial_P'], initial_A=stage.record['initial_A'], initial_cstar=stage.record['initial_cstar'],
        frozen_parent_parameter_updates=0, validation_predictions_started=False,
        validation_targets_accessed=False, checkpoint_selection=False, source_bindings=stage.record['source_bindings'],
        input_bindings=stage.record['input_bindings'], qualification=stage.record['prerequisites'],
        native_score_measured=False, native_gradient_qualified=False, no_native_gradient_claim=True))
    retire_optimizer(stage, state)
    stage.terminal_cstar = stage.h.load_array(np, terminal_c)
    c.require(stage.terminal_cstar.dtype == np.float32 and stage.terminal_cstar.shape == (15, 4), 'Reload the actual sealed exported c only')
    state['cstar'] = torch.tensor(stage.terminal_cstar, device='mps', dtype=torch.float32, requires_grad=False)
    outputs = {}
    for spec in entries[160:]:
        packet, answer = run_card(stage, np, torch, state, spec)
        outputs[str(spec['rank'])] = packet['prediction']
        del answer
    c.require(list(map(int, outputs)) == list(stage.valid), 'All40 original VALID outputs without truth')
    prediction_seal = stage.json('validation_predictions_sealed.json', dict(terminal_coefficients=terminal_c,
        terminal_seal_sha256=seal['sha256'], validation_outputs=outputs, all_40_complete=True,
        validation_targets_accessed=False, native_score_measured=False, no_native_gradient_claim=True))
    stage.record.update(updates=40, training_graphs=160, validation_graphs=40,
        gradient_calls=160, terminal_delta=terminal_delta, terminal_coefficients=terminal_c,
        terminal_seal_sha256=seal['sha256'], validation_predictions_sealed_sha256=prediction_seal['sha256'],
        validation_outputs=outputs, update_records=updates, online_training_metrics=metrics,
        online_proposal_raw_mean=math.fsum(x['raw_final_mse'] for x in metrics)/160,
        online_unweighted_surrogate_loss=math.fsum(x['unweighted_surrogate_loss'] for x in metrics)/160,
        online_weighted_surrogate_loss=math.fsum(x['weighted_surrogate_loss'] for x in metrics)/160,
        all16_domains_qualified=True, all47_selectors_qualified=True, frozen_P_A_c0=True,
        terminal_export_reloaded_without_inverse=True, training_optimizer_retired_before_validation=True,
        proposal_complete=True)


def main():
    args = c.arguments()
    if args is None:
        return 0
    stage = c.Stage(args, Path(__file__))
    state, timer_started, old_handler, ok = {}, False, None, False
    try:
        stage.admit()
        stage.helpers, helper_proof = source_helpers()
        stage.record['worker_source_proof'] = helper_proof
        c.require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules), 'No numerical import before all stage/source admission')
        import numpy as np
        import torch
        c.require(dict(numpy=np.__version__, torch=torch.__version__) == c.VERSIONS
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Exact AP MPS library frame')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory())
        c.require(recommended > 0, 'Positive recommended MPS allocation capacity')
        fraction = min(1., 16_000_000_000/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        stage.torch = torch
        stage.record['mps_allocator'] = dict(configured=True, recommended_max_memory_bytes=recommended,
            configured_fraction=fraction, api='torch.mps.set_per_process_memory_fraction', low_watermark_ratio='0.1')
        stage.record['environment']['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        old_handler = signal.signal(signal.SIGALRM, lambda *_: stage.guard())
        signal.setitimer(signal.ITIMER_REAL, .5, .5); timer_started = True
        torch.manual_seed(c.SEED)
        stage.model.runtime()
        P, A, delta = stage.model.frozen('mps', trainable_delta=True)
        c.require(sum(P[k][0].numel() for k in stage.model.INACTIVE_KEYS) == 66, 'Original66 inactive P coordinates')
        state.update(P=P, A=A, delta=delta, optimizer=torch.optim.Adam([delta],
            lr=0. if args.stage == 'qualify' else .025, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False))
        stage.initial_P = stage.helpers['payload_values'](np, P)
        stage.initial_A = array(A)
        initial = stage.model.initial_coefficients('mps')
        stage.initial_cstar = array(initial)
        del initial, P, A, delta
        stage.record.update(initial_P=stage.bundle(np, 'initial_P.npz', stage.initial_P),
            initial_A=stage.save(np, 'initial_A.npy', stage.initial_A),
            initial_cstar=stage.save(np, 'initial_cstar.npy', stage.initial_cstar),
            initial_delta=stage.save(np, 'initial_delta.npy', array(state['delta'])),
            allocated_parameters=56, active_parameters=56, trainable_keys=['delta'],
            cstar_raw_sha256=rawsha(stage.initial_cstar), initial_P_A_cstar_frozen=True)
        frozen(stage, np, state)
        entries = stage.helpers['roster'](stage, np)
        (qualify if args.stage == 'qualify' else train)(stage, np, torch, state, entries)
        c.require(len(stage.record['graph_records']) == len(entries) == (5 if args.stage == 'qualify' else 200), 'Entire prospectively fixed roster completed')
        stage.record['model_source_proof'] = stage.model.source_proof()
        frozen(stage, np, state)
        if 'optimizer' in state:
            retire_optimizer(stage, state)
        refs = [weakref.ref(v) for v in (*state['P'].values(), state['A'])]
        if 'cstar' in state:
            refs.append(weakref.ref(state['cstar']))
        state['P'].clear(); state.clear()
        stage.helpers['release'](stage)
        c.require(all(ref() is None for ref in refs), 'All persistent parameter/parent/terminal owners retired before final PASS')
        stage.record.update(all_graphs_retired=True, all_persistent_owners_retired=True,
            frozen_parent_parameter_updates=0, native_gradient_qualified=False, no_native_gradient_claim=True,
            CS_failure_preserved=True, tiny_model_calls=0)
        stage.finish()
    except BaseException:
        stage.record.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False, error=traceback.format_exc())
    finally:
        state.clear()
        if timer_started:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)
        ok = stage.close()
    print(c.json.dumps({k: stage.record.get(k) for k in ('schema', 'stage', 'passed', 'completed',
        'graphs_completed', 'updates', 'wall_s', 'maximum_sampled_mps_driver_bytes', 'error')}), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
