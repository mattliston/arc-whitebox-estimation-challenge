"""One target-free MPS lifecycle check or one fixed surrogate proposal pass.

Inert without --run. This never claims or qualifies a native derivative.
Only a later actual-native objective may accept the sole terminal proposal.
"""
import gc
import hashlib
import math
from pathlib import Path
import signal
import time
import traceback

import common as c


def payload_values(np, P):
    return {key: value.detach().cpu().numpy().copy() for key, value in P.items()}


def exact_values(left, right):
    return set(left) == set(right) and all(left[k].dtype == right[k].dtype
        and left[k].shape == right[k].shape and left[k].tobytes() == right[k].tobytes() for k in left)


def frozen(np, P, original, model, all_unchanged=False):
    current = payload_values(np, P)
    c.require(all(np.isfinite(v).all() for v in current.values()), 'Finite full P payload')
    for key, value in current.items():
        c.require(value.dtype == np.float32 and value.shape == original[key].shape, 'Original payload shape/precision')
        if key not in model.PKEYS:
            c.require(not P[key].requires_grad and P[key].grad is None
                      and value.tobytes() == original[key].tobytes(), 'Frozen non-P bytes/gradient')
        if key in model.INACTIVE_KEYS:
            c.require(value[0].tobytes() == original[key][0].tobytes(), 'Inactive layer0 P bytes')
    if all_unchanged:
        c.require(exact_values(current, original), 'Zero-LR qualification changed P bytes')
    return current


def gradients(np, P, model):
    result = {}
    for key in model.PKEYS:
        grad = P[key].grad
        c.require(grad is not None and not grad.requires_grad and grad.grad_fn is None, 'Owned detached gradient: '+key)
        value = grad.detach().cpu().numpy().copy()
        c.require(value.dtype == np.float32 and value.shape == tuple(P[key].shape)
                  and np.isfinite(value).all(), 'Finite full P gradient: '+key)
        if key in model.INACTIVE_KEYS:
            c.require(bool((value[0] == 0).all()), 'All66 inactive coordinates have zero gradients')
        result[key] = value
    c.require(sum(v.size for v in result.values()) == 45200, 'All45200 gradient coordinates')
    return result


def release(stage):
    stage.torch.mps.synchronize()
    gc.collect()
    stage.torch.mps.empty_cache()
    stage.torch.mps.synchronize()
    return stage.guard()


def roster(stage, np):
    if stage.args.stage == 'qualify':
        order, validation = [], []
        entries = [dict(graph_index=i, kind=kind, rank=rank, update=None)
                   for i, (kind, rank) in enumerate((('parent_forward', 100),
                       ('synthetic_backward', 35), ('synthetic_backward', 100), ('synthetic_backward', 35)))]
    else:
        order = np.random.Generator(np.random.PCG64DXSM(c.SEED)).permutation(stage.train).tolist()
        validation = list(stage.valid)
        c.require(len(order) == 160 and set(order) == set(stage.train), 'One complete deterministic TRAIN pass')
        entries = [dict(graph_index=i, kind='train_backward', rank=rank, update=i//4)
                   for i, rank in enumerate(order)]
        entries += [dict(graph_index=160+i, kind='validation_forward', rank=rank, update=None)
                    for i, rank in enumerate(validation)]
    value = dict(schema='current_d50_native_objective_p1_roster_v1', stage=stage.args.stage,
                 seed=c.SEED, generator='PCG64DXSM' if stage.args.stage == 'train' else None,
                 train_order=order, validation_order=validation, entries=entries)
    stage.record['model_call_roster'] = stage.json('model_call_roster.json', value)
    stage.record['training_order'] = order
    return entries


def run_card(stage, np, torch, P, original, spec, *, cotangent=None, target=None,
             weight=1., optimizer_zero=None, reference=None, repeat_gradient=None,
             reference_diagnostics=None, parent_selector_orders=None):
    """One unchanged CO card; only detached existing values enter receipts."""
    index, rank, kind = spec['graph_index'], spec['rank'], spec['kind']
    start = time.monotonic()
    before_P = frozen(np, P, original, stage.model)
    stage.json(f'graph_{index:04d}_attempt.json', dict(spec, started_utc=c.datetime.now(c.timezone.utc).isoformat(),
        elapsed_s=start-stage.start, P_raw_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in before_P.items()}))
    card = prediction = trace = loss = served = error = final = auxiliary = weights = None
    packet, grads = dict(spec), None
    try:
        weights = np.load(stage.weights[rank]['path'], mmap_mode='r', allow_pickle=False)
        c.require(weights.dtype == np.float32 and weights.shape == (16, 1024, 1024), 'Canonical MLP weight frame')
        card = stage.model.PCard(rank, P=P, device='mps', weights=weights, audit_selections=True)
        c.require(not card.coefficients.requires_grad and card.coefficients.grad is None
                  and card.coefficients.detach().cpu().numpy().tobytes() == stage.cstar_bytes, 'Exact frozen d50 cstar')
        backward = kind in ('synthetic_backward', 'train_backward')
        with torch.set_grad_enabled(backward):
            prediction, trace = card.run(training=backward)
            torch.mps.synchronize()
            packet['forward_elapsed_s'] = time.monotonic()-start
            value = stage.h.prediction_array(np, stage.model, prediction)
            packet['prediction'] = stage.save(np, f'graph_{index:04d}_served.npy', value)
            forward = card.diagnostics()
            packet['forward_diagnostics'] = stage.json(f'graph_{index:04d}_forward.json', forward)
            stage.model.check_domain(card)
            if reference_diagnostics is not None:
                c.require(forward['selection_cells'] == reference_diagnostics['selection_cells']
                          and forward['variance_domain'] == reference_diagnostics['variance_domain'],
                          'Original MPS35 chart/value bytes')
            if parent_selector_orders is not None:
                own = [forward['selection_cells'][f'{l}:{s}']['ordered_ids'] for l in range(16)
                       for s in range(2 if l == 15 else 3)]
                c.require(own == parent_selector_orders, 'Unmodified parent100 ordered selections')
            if reference is not None:
                packet['parent_byte_exact'] = value.tobytes() == reference.tobytes()
                c.require(packet['parent_byte_exact'], 'Same-backend parent output bytes')
            if kind == 'synthetic_backward':
                loss = stage.h.scalar(torch, stage.model, prediction, cotangent, card.dtype)
                packet['synthetic_loss'] = float(loss.detach().cpu())
            elif kind == 'train_backward':
                served = stage.model.served(prediction)
                error = served-target
                final, auxiliary = error[-1].square().mean(), error[:-1].square().mean()
                unweighted = (final+.05*auxiliary)/2e-8
                loss = weight*unweighted/4.
                packet.update(raw_final_mse=float(final.detach().cpu()),
                    earlier_mean_mse=float(auxiliary.detach().cpu()),
                    unweighted_surrogate_loss=float(unweighted.detach().cpu()),
                    weighted_surrogate_loss=float((weight*unweighted).detach().cpu()),
                    normalized_d50_cost_weight=weight)
                del unweighted
            if backward:
                c.require(bool(torch.isfinite(loss).detach().cpu()), 'Finite surrogate scalar')
                loss.backward()
                torch.mps.synchronize()
                packet['forward_backward_elapsed_s'] = time.monotonic()-start
                grads = gradients(np, P, stage.model)
                if kind == 'synthetic_backward':
                    packet['gradient'] = stage.bundle(np, f'graph_{index:04d}_gradient.npz', grads)
                    if repeat_gradient is not None:
                        packet['repeat_gradient_byte_exact'] = exact_values(grads, repeat_gradient)
                        c.require(packet['repeat_gradient_byte_exact'], 'Repeat35 full-gradient bytes')
                packet['gradient_raw_sha256'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in grads.items()}
            diagnostic = stage.model.check_replay(card)
            c.require(diagnostic['selection_cells'] == forward['selection_cells']
                      and diagnostic['variance_domain'] == forward['variance_domain'], 'Observed replay changed values')
            packet['diagnostics'] = stage.json(f'graph_{index:04d}_diagnostics.json', diagnostic)
            c.require(exact_values(frozen(np, P, original, stage.model), before_P), 'Card/backward mutated P')
            if optimizer_zero is not None:
                c.require(optimizer_zero.param_groups[0]['lr'] == 0., 'Only zero-LR qualification Adam')
                optimizer_zero.step()
                c.require(len(optimizer_zero.state) == len(stage.model.PKEYS), 'All11 Adam states allocated')
                for state in optimizer_zero.state.values():
                    c.require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                              and all(bool(torch.isfinite(v).all().detach().cpu()) for v in state.values()),
                              'Finite persistent Adam allocation')
                frozen(np, P, original, stage.model, all_unchanged=True)
                optimizer_zero.zero_grad(set_to_none=True)
                packet['zero_lr_Adam_P_byte_exact'] = True
            packet.update(all16_domains_present=True, all47_selectors_present=True,
                          observed_checkpoint_replay_byte_exact=True, frozen_cstar=True)
            stage.json(f'graph_{index:04d}_checked.json', packet)
    finally:
        card = prediction = trace = loss = served = error = final = auxiliary = weights = None
        release(stage)
    packet['retired_elapsed_s'] = time.monotonic()-start
    stage.graph_complete(index, packet)
    return packet, grads


def qualify(stage, np, torch, P, original, optimizer, entries):
    parent = pred = trace = weights = None
    started = time.monotonic()
    stage.json('graph_0000_attempt.json', entries[0])
    try:
        weights = np.load(stage.weights[100]['path'], mmap_mode='r', allow_pickle=False)
        c.require(weights.dtype == np.float32 and weights.shape == (16, 1024, 1024), 'Parent100 weights')
        parent = stage.model.parent_card(100, device='mps', weights=weights)
        with torch.no_grad():
            pred, trace = parent.run()
            value100 = stage.h.prediction_array(np, stage.model, pred)
        selectors = [list(map(int, item[0])) for item in parent.selections]
        c.require(len(selectors) == 47 and all(len(set(x)) == len(x) for x in selectors), 'Original parent47 selectors')
        parent_item = stage.save(np, 'parent_0100.npy', value100)
        stage.json('parent_0100_selectors.json', selectors)
    finally:
        parent = pred = trace = weights = None
        release(stage)
    frozen(np, P, original, stage.model, all_unchanged=True)
    stage.graph_complete(0, dict(entries[0], prediction=parent_item,
                                retired_elapsed_s=time.monotonic()-started, original_parent=True))
    value35 = stage.h.load_array(np, stage.parent35)
    refs, first_gradient, checked = {35: value35, 100: value100}, None, []
    for item in entries[1:]:
        optimizer.zero_grad(set_to_none=True)
        cot = stage.h.load_array(np, stage.cotangents[str(item['rank'])])
        packet, grads = run_card(stage, np, torch, P, original, item, cotangent=cot,
            optimizer_zero=optimizer, reference=refs[item['rank']],
            repeat_gradient=first_gradient if item['graph_index'] == 3 else None,
            reference_diagnostics=stage.reference35['diagnostics'] if item['rank'] == 35 else None,
            parent_selector_orders=selectors if item['rank'] == 100 else None)
        if first_gradient is None:
            first_gradient = grads
        checked.append(packet)
        del cot, grads
    frozen(np, P, original, stage.model, all_unchanged=True)
    stage.record.update(execution_integrity_passed=True, parent100=parent_item, checks=checked,
        gradient_calls=3, zero_lr_Adam_calls=3, parameter_updates=0,
        first_repeat35_prediction_byte_exact=True, first_repeat35_gradient_byte_exact=True,
        full_P_unchanged=True, unused_layer0_parameters_unchanged=True)


def train(stage, np, torch, P, original, optimizer, entries):
    metrics, update_records = [], []
    stage.record.update(epochs=1, planned_updates=40, batch_nets=4, seed=c.SEED,
        loss='normalized cold d50 multiplier*(served final MSE+.05*earlier mean MSE)/2e-8',
        optimizer=dict(name='Adam', lr_start=5e-4, lr_terminal=5e-5, betas=[.9, .999],
                       eps=1e-8, weight_decay=0, foreach=False, gradient_norm_cap=5.))
    parameters = [P[k] for k in stage.model.PKEYS]
    for update in range(40):
        lr = 5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*update/39))
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        for item in entries[4*update:4*update+4]:
            rank = item['rank']
            path = c.TRUTH/f'truthpost_rank{rank:04d}.npz'
            stage.pin(path, stage.targets[rank], 'target_bindings')
            stage.record['training_targets_accessed'] = True
            with np.load(path, allow_pickle=False) as source:
                values = source['mc_post'].copy()
            c.require(values.shape == (16, 1024) and np.isfinite(values).all(), 'Pinned TRAIN1e9 mean frame')
            target = torch.tensor(values, device='mps', dtype=torch.float32)
            packet, _grads = run_card(stage, np, torch, P, original, item, target=target,
                                      weight=stage.costs[rank]['normalized_weight'])
            metrics.append({k: packet[k] for k in ('rank', 'raw_final_mse', 'earlier_mean_mse',
                'unweighted_surrogate_loss', 'weighted_surrogate_loss', 'normalized_d50_cost_weight')})
            del target, values, _grads
        gradients(np, P, stage.model)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True,
                                                    foreach=False).detach().cpu())
        optimizer.step()
        frozen(np, P, original, stage.model)
        stage.record['parameter_updates'] = update+1
        update_records.append(stage.json(f'update_{update+1:02d}.json', dict(update=update+1,
            ranks=[x['rank'] for x in entries[4*update:4*update+4]], learning_rate=lr,
            pre_clip_gradient_norm=norm, resources=stage.guard())))
    c.require(len(metrics) == 160 and stage.record['parameter_updates'] == 40, 'Exactly160 graphs/40updates')
    terminal_values = frozen(np, P, original, stage.model)
    terminal = stage.bundle(np, 'terminal_payload.npz', terminal_values)
    seal = stage.json('terminal_seal.json', dict(terminal_epoch=1, updates=40, terminal_payload=terminal,
        cstar_raw_sha256=stage.record['cstar_raw_sha256'], trainable_keys=list(stage.model.PKEYS),
        validation_targets_accessed=False, checkpoint_selection=False,
        source_bindings=stage.record['source_bindings'], training_graphs=160,
        native_score_measured=False, no_native_gradient_claim=True))
    # Inference consumes the sole sealed NPZ, not the live training leaves.
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    parameters.clear()
    P.clear()
    release(stage)
    values = stage.h.load_bundle(np, terminal)
    terminal_P = stage.h.torch_payload(torch, values, stage.model.PKEYS, 'mps', gradients=False)
    outputs = {}
    for item in entries[160:]:
        packet, _ = run_card(stage, np, torch, terminal_P, original, item)
        outputs[str(item['rank'])] = packet['prediction']
    c.require(list(map(int, outputs)) == list(stage.valid), 'All40 fixed VALID outputs and no labels')
    prediction_seal = stage.json('validation_predictions_sealed.json', dict(terminal_payload=terminal,
        terminal_seal_sha256=seal['sha256'], validation_outputs=outputs, all_40_complete=True,
        validation_targets_accessed=False, no_native_gradient_claim=True, native_score_measured=False))
    stage.record.update(epochs=1, updates=40, training_graphs=160, validation_graphs=40,
        update_records=update_records, terminal_payload=terminal, terminal_seal_sha256=seal['sha256'],
        validation_predictions_sealed_sha256=prediction_seal['sha256'], validation_outputs=outputs,
        online_training_metrics=metrics,
        online_raw_mean=math.fsum(x['raw_final_mse'] for x in metrics)/160,
        online_unweighted_surrogate_loss=math.fsum(x['unweighted_surrogate_loss'] for x in metrics)/160,
        online_weighted_surrogate_loss=math.fsum(x['weighted_surrogate_loss'] for x in metrics)/160,
        unused_layer0_parameters_unchanged=True, proposal_complete=True)


def main():
    args = c.arguments()
    if args is None:
        return 0
    stage = c.Stage(args, Path(__file__))
    old_handler, timer_started = None, False
    try:
        stage.admit()
        import numpy as np
        import torch
        c.require(dict(numpy=np.__version__, torch=torch.__version__) == c.VERSIONS, 'Exact installed library frame')
        c.require(torch.backends.mps.is_available(), 'MPS only; no fallback')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory())
        c.require(recommended > 0, 'Positive recommended MPS memory')
        fraction = min(1., 16_000_000_000/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        stage.torch = torch
        stage.record['mps_allocator'] = dict(configured=True, recommended_max_memory_bytes=recommended,
            configured_fraction=fraction, api='torch.mps.set_per_process_memory_fraction', low_watermark_ratio='0.1')
        stage.record['environment']['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        old_handler = signal.signal(signal.SIGALRM, lambda *_: stage.guard())
        signal.setitimer(signal.ITIMER_REAL, .5, .5)
        timer_started = True
        torch.manual_seed(c.SEED)
        original = stage.h.load_bundle(np, stage.payload)
        c.require(exact_values(original, stage.model.payload_numpy()), 'Exact d50 initialization, no TL payload')
        P = stage.h.torch_payload(torch, original, stage.model.PKEYS, 'mps', gradients=True)
        c.require(sum(P[k].numel() for k in stage.model.PKEYS) == 45200
                  and all(P[k].is_leaf and P[k].requires_grad for k in stage.model.PKEYS), 'All existing P leaves')
        c.require(sum(P[k][0].numel() for k in stage.model.INACTIVE_KEYS) == 66,
                  'Exactly66 structurally inactive coordinates')
        stage.cstar_bytes = np.asarray(stage.model.frozen_coefficients(), dtype=np.float32).tobytes()
        stage.record.update(allocated_parameters=45200, active_parameters=45134,
            trainable_keys=list(stage.model.PKEYS), cstar_raw_sha256=hashlib.sha256(stage.cstar_bytes).hexdigest(),
            cstar_frozen=True, payload_initial_sha256=c.PAYLOAD_SHA)
        optimizer = torch.optim.Adam([P[k] for k in stage.model.PKEYS],
            lr=0. if args.stage == 'qualify' else 5e-4, betas=(.9, .999), eps=1e-8,
            weight_decay=0, foreach=False)
        entries = roster(stage, np)
        (qualify if args.stage == 'qualify' else train)(stage, np, torch, P, original, optimizer, entries)
        c.require(len(stage.record['graph_records']) == len(entries), 'Entire fixed model-call roster completed')
        del P, optimizer
        release(stage)
        stage.finish()
    except BaseException:
        stage.record.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                            error=traceback.format_exc())
    finally:
        if timer_started:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        stage.close()
    print(c.json.dumps({k: stage.record.get(k) for k in ('schema', 'stage', 'passed', 'completed',
        'graphs_completed', 'updates', 'wall_s', 'maximum_sampled_mps_driver_bytes', 'error')}), flush=True)
    return 0 if stage.record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
