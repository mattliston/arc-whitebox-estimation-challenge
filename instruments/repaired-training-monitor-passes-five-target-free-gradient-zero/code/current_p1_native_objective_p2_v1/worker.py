"""SOURCE-PREPARED ONLY: renewed P1 integrity or two fixed additional passes.

Unchanged P1 card/backward/loss/retirement owners are inverse-preserved.
Only explicit --run constructs a stage or imports numerical libraries.
"""
from pathlib import Path
import common as c


def verify_initial(stage, np, original):
    """Sealed P1 full payload/native literal, with original d50 frozen fields."""
    parent = stage.model.payload_numpy()  # Shape/frozen-field control, NOT initialization.
    nodes = [n for n in c.ast.parse((c.P1/'build_v1/estimator.py').read_text()).body
             if isinstance(n, c.ast.Assign) and any(isinstance(t, c.ast.Name)
             and t.id == '_GNN_LITERAL' for t in n.targets)]
    c.require(len(nodes) == 1, 'Unique sealed P1 native P literal')
    literal = c.ast.literal_eval(nodes[0].value)
    c.require(set(original) == set(parent) == set(literal) == set(stage.payload['arrays']), 'Full P1 payload keys')
    for key, value in original.items():
        decoded = np.asarray(literal[key], dtype=np.float32)
        c.require(value.dtype == np.float32 and value.shape == parent[key].shape
                  and np.isfinite(value).all() and stage.h.array_meta(value) == stage.payload['arrays'][key]
                  and decoded.shape == value.shape and decoded.tobytes() == value.tobytes(),
                  'P1 payload/native-literal shape, precision and bytes: '+key)
        if key not in stage.model.PKEYS:
            c.require(value.tobytes() == parent[key].tobytes(), 'Frozen d50 non-P fields')
        if key in stage.model.INACTIVE_KEYS:
            c.require(value[0].tobytes() == parent[key][0].tobytes(), 'Frozen66 inactive coordinates')
    stage.record.update(initial_native_literal_byte_exact=True, initial_payload_is_P1=True,
        worker_adapter=WORKER_ADAPTER_PROOF, fresh_Adam=True, optimizer_resume=False)


def roster(stage, np):
    order, validation, orders, skipped = [], [], [], []
    if stage.args.stage == 'qualify':
        entries = [dict(graph_index=i, kind=kind, rank=rank, update=None)
                   for i, (kind, rank) in enumerate((('parent_forward', 35), ('parent_forward', 100),
                       ('synthetic_backward', 35), ('synthetic_backward', 100), ('synthetic_backward', 35)))]
    else:
        rng = np.random.Generator(np.random.PCG64DXSM(c.SEED))
        skipped = rng.permutation(stage.train).tolist()
        c.require(skipped == stage.parent_lineage['training_order'], 'Reproduce the sealed P1 first permutation')
        orders = [rng.permutation(stage.train).tolist() for _ in range(2)]
        c.require(all(len(x) == 160 and set(x) == set(stage.train) for x in orders), 'Two complete next TRAIN passes')
        order, validation = orders[0]+orders[1], list(stage.valid)
        entries = [dict(graph_index=i, kind='train_backward', rank=rank, update=i//4)
                   for i, rank in enumerate(order)]
        entries += [dict(graph_index=320+i, kind='validation_forward', rank=rank, update=None)
                    for i, rank in enumerate(validation)]
    value = dict(schema='current_p1_native_objective_p2_roster_v1', stage=stage.args.stage,
        seed=c.SEED, generator='PCG64DXSM' if stage.args.stage == 'train' else None,
        train_order=order, train_orders=orders, skipped_P1_order=skipped,
        stream_permutations=[2, 3] if stage.args.stage == 'train' else [],
        parent_roster_sha256=c.P1_ROSTER_SHA, validation_order=validation, entries=entries)
    stage.record['model_call_roster'] = stage.json('model_call_roster.json', value)
    stage.record.update(training_order=order, training_orders=orders,
        additional_epochs=2 if stage.args.stage == 'train' else 0,
        cumulative_train_passes=3 if stage.args.stage == 'train' else 1)
    return entries


def check_zero_adam(stage, torch, P, optimizer, step_number):
    c.require(len(optimizer.state) == len(stage.model.PKEYS), 'All11 fresh Adam states allocated')
    for key in stage.model.PKEYS:
        state = optimizer.state[P[key]]
        c.require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                  and all(bool(torch.isfinite(v).all().detach().cpu()) for v in state.values())
                  and state['step'].numel() == 1 and float(state['step'].detach().cpu()) == step_number,
                  'Actual zero-LR Adam step count/finite state')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            c.require(value.shape == P[key].shape and value.dtype == P[key].dtype and value.device == P[key].device,
                      'Adam moment allocation matches the P leaf')
            if key in stage.model.INACTIVE_KEYS:
                c.require(bool((value[0] == 0).all().detach().cpu()), 'Inactive Adam moments remain zero')


def qualify(stage, np, torch, P, original, optimizer, entries):
    refs, selector_orders, controls = {}, {}, {}
    for item in entries[:2]:
        index, rank = item['graph_index'], item['rank']
        parent = pred = trace = weights = coefficients = None
        started = time.monotonic()
        stage.json(f'graph_{index:04d}_attempt.json', item)
        try:
            weights = np.load(stage.weights[rank]['path'], mmap_mode='r', allow_pickle=False)
            c.require(weights.dtype == np.float32 and weights.shape == (16, 1024, 1024), 'P1 parent canonical weights')
            base = stage.model._base_module()
            coefficients = torch.tensor(stage.model.frozen_coefficients(), device='mps', dtype=torch.float32)
            c.require(coefficients.detach().cpu().numpy().tobytes() == stage.cstar_bytes, 'Parent frozen cstar')
            parent = base.BirthDirectionCard(rank, coefficients=coefficients, device='mps',
                                            weights=weights, P=original, audit_selections=True)
            c.require(exact_values(payload_values(np, parent.P0), original), 'Explicit separate P1 parent leaves')
            with torch.no_grad():
                pred, trace = parent.run()
                torch.mps.synchronize()
                refs[rank] = stage.h.prediction_array(np, stage.model, pred)
            orders = [list(map(int, selection[0])) for selection in parent.selections]
            c.require(len(orders) == 47 and all(len(x) > 0 and len(set(x)) == len(x)
                      and all(0 <= i < 1024 for i in x) for x in orders), 'P1 parent47 ordered selector cells')
            c.require(exact_values(payload_values(np, parent.P0), original), 'Parent forward preserves all P1 bytes')
            selector_orders[rank] = orders
            controls[str(rank)] = dict(prediction=stage.save(np, f'parent_{rank:04d}.npy', refs[rank]),
                selectors=stage.json(f'parent_{rank:04d}_selectors.json', orders))
        finally:
            parent = pred = trace = weights = coefficients = None
            release(stage)
        frozen(np, P, original, stage.model, all_unchanged=True)
        stage.graph_complete(index, dict(item, **controls[str(rank)],
            retired_elapsed_s=time.monotonic()-started, explicit_P1_parent=True))
    first_gradient, checked = None, []
    for item in entries[2:]:
        optimizer.zero_grad(set_to_none=True)
        cot = stage.h.load_array(np, stage.cotangents[str(item['rank'])])
        packet, grads = run_card(stage, np, torch, P, original, item, cotangent=cot,
            optimizer_zero=optimizer, reference=refs[item['rank']],
            repeat_gradient=first_gradient if item['graph_index'] == 4 else None,
            parent_selector_orders=selector_orders[item['rank']])
        if first_gradient is None:
            first_gradient = grads
        checked.append(packet)
        del cot, grads
    frozen(np, P, original, stage.model, all_unchanged=True)
    stage.record.update(execution_integrity_passed=True, parent_controls=controls, checks=checked,
        parent_forward_calls=2, gradient_calls=3, zero_lr_Adam_calls=3, zero_lr_Adam_steps=[1, 2, 3],
        parameter_updates=0, first_repeat35_prediction_byte_exact=True,
        first_repeat35_gradient_byte_exact=True, full_P_unchanged=True,
        unused_layer0_parameters_unchanged=True, qualification_optimizer_discarded=True)


def _function(source, name):
    nodes = [n for n in c.ast.parse(source).body if isinstance(n, c.ast.FunctionDef) and n.name == name]
    c.require(len(nodes) == 1, 'Unique inherited function: '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


_owner = (c.P1/'worker.py').read_text()
_old_adam = """                c.require(len(optimizer_zero.state) == len(stage.model.PKEYS), 'All11 Adam states allocated')
                for state in optimizer_zero.state.values():
                    c.require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                              and all(bool(torch.isfinite(v).all().detach().cpu()) for v in state.values()),
                              'Finite persistent Adam allocation')"""
_changes = [
    (_function(_owner, 'roster'), '# P2 roster is explicitly defined in this versioned adapter.\n', 1),
    (_function(_owner, 'qualify'), '# P2 qualification is explicitly defined in this versioned adapter.\n', 1),
    (_old_adam, '                check_zero_adam(stage, torch, P, optimizer_zero, index-1)', 1),
    ("'Unmodified parent100 ordered selections'", "'Unmodified explicit P1 parent ordered selections'", 1),
    ('epochs=1, planned_updates=40', 'epochs=2, planned_updates=80', 1),
    ('for update in range(40):', 'for update in range(80):', 1),
    ('math.pi*update/39', 'math.pi*update/79', 1),
    ("len(metrics) == 160 and stage.record['parameter_updates'] == 40, 'Exactly160 graphs/40updates'",
     "len(metrics) == 320 and stage.record['parameter_updates'] == 80, 'Exactly320 graphs/80updates'", 1),
    ('terminal_epoch=1, updates=40, terminal_payload=terminal',
     'terminal_epoch=2, updates=80, terminal_payload=terminal,\n        parent_lineage=stage.parent_lineage, fresh_Adam=True, optimizer_resume=False,\n        additional_epochs=2, cumulative_train_passes=3', 1),
    ("source_bindings=stage.record['source_bindings'], training_graphs=160",
     "source_bindings=stage.record['source_bindings'], training_graphs=320", 1),
    ('for item in entries[160:]:', 'for item in entries[320:]:', 1),
    ('epochs=1, updates=40, training_graphs=160, validation_graphs=40',
     'epochs=2, updates=80, training_graphs=320, validation_graphs=40', 1),
    ("for x in metrics)/160", "for x in metrics)/320", 3),
    ("        c.require(exact_values(original, stage.model.payload_numpy()), 'Exact d50 initialization, no TL payload')",
     '        verify_initial(stage, np, original)', 1),
    ("        optimizer = torch.optim.Adam([P[k] for k in stage.model.PKEYS],",
     "        c.require(stage.record['cstar_raw_sha256'] == c.P1_CSTAR_SHA, 'P1/d50 cstar bytes unchanged')\n"
     "        optimizer = torch.optim.Adam([P[k] for k in stage.model.PKEYS],", 1),
    ('        entries = roster(stage, np)',
     "        c.require(len(optimizer.state) == 0, 'New process, fresh Adam: no optimizer resume')\n"
     '        entries = roster(stage, np)', 1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Inherited entrypoint suppressed; the adapter owns the sole explicit entrypoint.\n', 1),
]
_derived, WORKER_ADAPTER_PROOF = c.splice_source(c.P1/'worker.py', c.P1_WORKER_SHA, _changes)
exec(compile(_derived, str(Path(__file__).resolve())+':inherited', 'exec'), globals())


if __name__ == '__main__':
    raise SystemExit(main())
