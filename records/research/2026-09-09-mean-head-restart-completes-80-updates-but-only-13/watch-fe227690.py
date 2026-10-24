"""Versioned stdlib P1 supervisor: five renewed calls or320+40 model calls.

All original process-group cleanup, progress, memory, swap and source sealing
are inverse-preserved. Explicit launch attestations are not scientific gates.
"""
from pathlib import Path
import common as c


def roster(path, stage):
    value = read(path)
    require(value['schema'] == 'current_p1_native_objective_p2_roster_v1'
            and value['stage'] == stage and value['seed'] == 9309096
            and value['parent_roster_sha256'] == c.P1_ROSTER_SHA, 'Fixed P1-continuation roster lineage')
    if stage == 'qualify':
        require(value['generator'] is None and not value['train_order'] and not value['train_orders']
                and not value['validation_order'] and not value['skipped_P1_order']
                and not value['stream_permutations'], 'Qualification contains no training permutation')
        expected = [dict(graph_index=i, rank=r, kind=k, update=None) for i, (r, k) in enumerate(QUALIFY)]
    else:
        order, orders = value['train_order'], value['train_orders']
        parent_path = c.P1/'training_v1/model_call_roster.json'
        require(sha(parent_path) == c.P1_ROSTER_SHA, 'Immutable skipped P1 stream receipt')
        require(value['generator'] == 'PCG64DXSM' and value['stream_permutations'] == [2, 3]
                and value['skipped_P1_order'] == read(parent_path)['train_order']
                and len(orders) == 2 and all(len(x) == 160 and all(type(r) is int for r in x)
                and set(x) == set(TRAIN) for x in orders)
                and order == orders[0]+orders[1] and len(order) == 320
                and value['validation_order'] == list(VALID), 'Two fixed TRAIN memberships and exact VALID order')
        # The pinned worker owns PCG64DXSM arithmetic; this stdlib supervisor
        # independently checks lineage, per-pass membership and every call/update.
        expected = [dict(graph_index=i, rank=r, kind='train_backward', update=i//4) for i, r in enumerate(order)]
        expected += [dict(graph_index=320+i, rank=r, kind='validation_forward', update=None)
                     for i, r in enumerate(VALID)]
    require(value['entries'] == expected, 'Exact ordered call identities and80-update ownership')
    return value


def admit_parent(sources, inputs, tick):
    def pin(path, digest, source=False):
        path = str(Path(path).resolve())
        selected = sources if source else inputs
        require(path not in selected or selected[path] == digest, 'Consistent continuation lineage')
        # These are bytes/hashes, never numerical arrays or target inventories.
        if path not in sources and path not in inputs:
            require(sha(path, tick) == digest, 'Preserved P1 lineage hash: '+path)
        else:
            require(sources.get(path, inputs.get(path)) == digest, 'Same preserved P1 binding')
        selected[path] = digest
    return c.parent_lineage(pin)


def check_continuation(value, stage):
    require(value['parent_payload_sha256'] == c.P1_PAYLOAD_SHA
            and value['parent_candidate_sha256'] == c.P1_CANDIDATE_SHA
            and value['payload_initial_sha256'] == c.P1_PAYLOAD_SHA
            and value['cstar_raw_sha256'] == c.P1_CSTAR_SHA
            and value['initial_native_literal_byte_exact'] and value['initial_payload_is_P1']
            and value['fresh_Adam'] and value['optimizer_resume'] is False, 'Exact P1 start; fresh, not resumed Adam')
    if stage == 'qualify':
        require(value['parent_forward_calls'] == 2 and value['gradient_calls'] == value['zero_lr_Adam_calls'] == 3
                and value['zero_lr_Adam_steps'] == [1, 2, 3] and value['parameter_updates'] == 0
                and value['full_P_unchanged'] and value['first_repeat35_prediction_byte_exact']
                and value['first_repeat35_gradient_byte_exact'] and value['qualification_optimizer_discarded']
                and set(value['parent_controls']) == {'35', '100'}, 'Five-call P1 integrity and actual LR0 Adam')
    else:
        require(value['epochs'] == value['additional_epochs'] == 2 and value['cumulative_train_passes'] == 3
                and value['updates'] == value['planned_updates'] == value['parameter_updates'] == 80
                and value['training_graphs'] == 320 and value['validation_graphs'] == 40
                and value['proposal_complete'] and not value['checkpoint_selection']
                and value['unused_layer0_parameters_unchanged'], 'Sole fixed80-update terminal')


_owner = (c.P1/'watch.py').read_text()
_nodes = [n for n in c.ast.parse(_owner).body if isinstance(n, c.ast.FunctionDef) and n.name == 'roster']
c.require(len(_nodes) == 1, 'Unique inherited roster verifier')
_node = _nodes[0]
_old_roster = ''.join(_owner.splitlines(keepends=True)[_node.lineno-1:_node.end_lineno])
_changes = [
    (c.P1_PROTOCOL_SHA, c.P2_PROTOCOL_SHA, 1),
    ("SCHEMA = 'current_d50_native_objective_p1_v1'", "SCHEMA = 'current_p1_native_objective_p2_v1'", 1),
    ("WATCH_SCHEMA = 'current_d50_native_objective_p1_watch_v1'", "WATCH_SCHEMA = 'current_p1_native_objective_p2_watch_v1'", 1),
    ("QUALIFY = ((100,'parent_forward'),(35,'synthetic_backward'),\n           (100,'synthetic_backward'),(35,'synthetic_backward'))",
     "QUALIFY = ((35,'parent_forward'),(100,'parent_forward'),(35,'synthetic_backward'),\n           (100,'synthetic_backward'),(35,'synthetic_backward'))", 1),
    (_old_roster, '# P2 roster verification is explicitly defined in this adapter.\n', 1),
    ("    parser.add_argument('--mps-slot-clear',action='store_true')",
     "    parser.add_argument('--mps-slot-clear',action='store_true')\n"
     "    parser.add_argument('--prior-verdict-reviewed',action='store_true')", 1),
    ("require(args.protocol_sha256 == PROTOCOL_SHA and args.mps_slot_clear,'Fixed protocol and sole-MPS attestation')",
     "require(args.protocol_sha256 == PROTOCOL_SHA and args.mps_slot_clear and args.prior_verdict_reviewed,\n"
     "            'Fixed protocol, sole-MPS and retired prior-verdict launch attestations')", 1),
    ('    bind(sources,time_guard); bind(inputs,time_guard)',
     '    bind(sources,time_guard); bind(inputs,time_guard)\n'
     '    lineage = admit_parent(sources,inputs,time_guard)', 1),
    ("old['completed_graph_count'] == 4", "old['completed_graph_count'] == 5", 1),
    ("4 if args.stage == 'qualify' else 200", "5 if args.stage == 'qualify' else 360", 2),
    ('    return value,end', '    check_continuation(value,stage)\n    return value,end', 1),
    ('mps_slot_clear_attested=True,numerical_imports=False,truth_arrays_opened=False,',
     'mps_slot_clear_attested=True,prior_verdict_reviewed_attested=True,\n'
     '        parent_lineage=lineage,watch_adapter=WATCH_ADAPTER_PROOF,\n'
     '        numerical_imports=False,truth_arrays_opened=False,', 1),
    ('supervisor independently checks unique TRAIN/exact VALID and marker order.',
     'supervisor independently checks each TRAIN pass/exact VALID, skipped P1 order and marker order.', 1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Inherited entrypoint suppressed; this adapter owns the sole explicit entrypoint.\n', 1),
]
_derived, WATCH_ADAPTER_PROOF = c.splice_source(c.P1/'watch.py', c.P1_WATCH_SHA, _changes)
exec(compile(_derived, str(Path(__file__).resolve())+':inherited', 'exec'), globals())


if __name__ == '__main__':
    raise SystemExit(main())
