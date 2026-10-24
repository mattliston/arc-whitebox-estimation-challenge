"""Inert same-shaped-P builder for the sole completed P1 continuation.

Only --build admits receipts and then decodes the two parameter payloads.
No estimator, Torch, target or prediction-array decoding is performed.
The frozen P1 literal construction is preserved by a source inverse.
"""
from pathlib import Path
import common as c

P1_BUILDER_SHA = '57f87081cdcc742266bc8487ef5ba11358bdf25e8341b9b1f4dd4458f06da54c'
WATCH_SCHEMA = 'current_p1_native_objective_p2_watch_v1'


def _put(mapping, path, digest):
    path = str(Path(path).resolve())
    require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef'),
            'Explicit immutable digest')
    require(path not in mapping or mapping[path] == digest, 'Conflicting source/receipt owner')
    mapping[path] = digest


def _same_sources(record, sources):
    for path in (HERE/'PROTOCOL.md', HERE/'worker.py', HERE/'common.py', HERE/'watch.py'):
        require(record['source_bindings'].get(str(path)) == sources[str(path)], 'Same reviewed P2 sources')
    # sealed() already authenticates these bytes, and inherited main rehashes
    # the complete merged map immediately before producing either candidate.
    for path, digest in record['source_bindings'].items():
        _put(sources, path, digest)


def _watch(record, stage, child_sha, child_completion_sha, count):
    require(record['schema'] == WATCH_SCHEMA and record['stage'] == stage
            and record['child_result_sha256'] == child_sha
            and record['child_completion_sha256'] == child_completion_sha
            and record['whole_owned_tree_retired'] and record['child_execution_completed']
            and record['completed_graph_count'] == count
            and record['mps_slot_clear_attested'] and record['prior_verdict_reviewed_attested']
            and record['auto_launch_next_stage'] is False and record['scientific_advancement'] is False,
            'Completed independent continuation supervision, not native admission')
    require(0 < record['maximum_sampled_rss_bytes'] < 12_000_000_000
            and 0 < record['maximum_sampled_mps_driver_bytes'] < 16_000_000_000
            and 0 <= record['maximum_swap_growth_bytes'] <= 512*1024**2
            and 0 < record['maximum_sample_interval_s'] <= 1
            and record['elapsed_s'] < (300 if stage == 'qualify' else 1800),
            'Original external resource limits passed')


def _calls(record, folder, entries):
    require(record['graphs_completed'] == len(record['graph_records']) == len(entries), 'Complete exact model-call count')
    item = record['model_call_roster']
    roster_path = folder/'model_call_roster.json'
    require(Path(item['path']).resolve() == roster_path
            and sha(roster_path) == item['sha256'] == record['artifacts'][str(roster_path)], 'Sealed exact roster')
    roster = read(roster_path)
    require(roster['schema'] == 'current_p1_native_objective_p2_roster_v1'
            and roster['stage'] == record['stage'] and roster['seed'] == c.SEED
            and roster['entries'] == entries and roster['parent_roster_sha256'] == c.P1_ROSTER_SHA,
            'Exact call/update identities and original PCG lineage')
    for entry, marker in zip(entries, record['graph_records']):
        path = folder/f"graph_{entry['graph_index']:04d}_complete.json"
        require(Path(marker['path']).resolve() == path
                and sha(path) == marker['sha256'] == record['artifacts'][str(path)], 'Every retired call is sealed')
        value = read(path)
        require(all(value.get(key) is True for key in ('passed', 'completed', 'graph_retired'))
                and all(value[key] == entry[key] for key in ('graph_index', 'kind', 'rank', 'update')),
                'No omitted, reordered or unretired call')
    return roster


def admission(args):
    require(args.protocol_sha256 == PROTO_SHA, 'Unchanged fixed two-pass continuation protocol')
    helper, record, keys, sources, inputs = _inherited_admission(args)
    require(sha(c.P1/'build.py') == P1_BUILDER_SHA, 'Frozen same-shaped-P construction owner')
    _put(sources, c.P1/'build.py', P1_BUILDER_SHA)
    qual = sealed(HERE/'qualification_v1', args.qualification_sha256)
    qw = sealed(HERE/'qualification_watch_v1', args.qualification_watch_sha256)
    tw = sealed(HERE/'training_watch_v1', args.training_watch_sha256)
    for value in (record, qual, qw, tw):
        _same_sources(value, sources)

    def pin_parent(path, digest, source=False):
        mapping = sources if source else inputs
        path = str(Path(path).resolve())
        require(sha(path) == digest, 'Exact retained P1 lineage')
        _put(mapping, path, digest)

    lineage = c.parent_lineage(pin_parent)
    for value in (record, qual, qw, tw):
        require(value['parent_lineage'] == lineage, 'All new stages use the same retained P1 parent')
    for value in (record, qual):
        require(value['parent_payload_sha256'] == value['payload_initial_sha256'] == c.P1_PAYLOAD_SHA
                and value['parent_candidate_sha256'] == c.P1_CANDIDATE_SHA
                and value['initial_native_literal_byte_exact'] and value['initial_payload_is_P1']
                and value['fresh_Adam'] and value['optimizer_resume'] is False
                and value['cstar_raw_sha256'] == c.P1_CSTAR_SHA and value['cstar_frozen']
                and value['common_adapter'] == c.COMMON_ADAPTER_PROOF,
                'Exact P1 initial state, fixed d50 non-P/cstar and fresh Adam')
    require(qual['schema'] == SCHEMA and qual['stage'] == 'qualify'
            and qual['execution_integrity_passed'] and not qual['training_targets_accessed']
            and not qual['validation_targets_accessed'] and qual['no_native_gradient_claim']
            and qual['parent_forward_calls'] == 2 and qual['gradient_calls'] == qual['zero_lr_Adam_calls'] == 3
            and qual['zero_lr_Adam_steps'] == [1, 2, 3] and qual['parameter_updates'] == 0
            and qual['full_P_unchanged'] and qual['first_repeat35_prediction_byte_exact']
            and qual['first_repeat35_gradient_byte_exact'] and qual['qualification_optimizer_discarded']
            and set(qual['parent_controls']) == {'35', '100'} and len(qual['checks']) == 3,
            'Completed renewed target-free P1 qualification')
    qentries = [dict(graph_index=i, kind=kind, rank=rank, update=None)
        for i, (kind, rank) in enumerate((('parent_forward', 35), ('parent_forward', 100),
            ('synthetic_backward', 35), ('synthetic_backward', 100), ('synthetic_backward', 35)))]
    qr = _calls(qual, HERE/'qualification_v1', qentries)
    require(qr['generator'] is None and not qr['train_order'] and not qr['train_orders']
            and not qr['validation_order'] and not qr['skipped_P1_order'] and not qr['stream_permutations'],
            'Qualification has no training permutation or labels')
    _watch(qw, 'qualify', args.qualification_sha256, sha(HERE/'qualification_v1/completion.json'), 5)
    _watch(tw, 'train', args.training_sha256, sha(HERE/'training_v1/completion.json'), 360)
    prerequisites = dict(qualification_result_sha256=args.qualification_sha256,
                         qualification_watch_sha256=args.qualification_watch_sha256)
    require(record['prerequisites'] == tw['prerequisites'] == prerequisites,
            'Both new qualification seals admitted this exact training process')
    train, valid, _ = helper.definitions()
    orders = record['training_orders']
    require(record['train_ranks'] == list(train) and record['validation_ranks'] == list(valid)
            and len(orders) == 2 and all(len(order) == 160 and set(order) == set(train) for order in orders)
            and record['training_order'] == orders[0]+orders[1]
            and record['epochs'] == record['additional_epochs'] == 2 and record['cumulative_train_passes'] == 3
            and record['parameter_updates'] == record['updates'] == record['planned_updates'] == 80
            and record['training_graphs'] == 320 and record['validation_graphs'] == 40
            and len(record['update_records']) == 80 and record['proposal_complete'],
            'Only the complete fixed80-update terminal, with no intermediate selection')
    entries = [dict(graph_index=i, kind='train_backward', rank=rank, update=i//4)
               for i, rank in enumerate(record['training_order'])]
    entries += [dict(graph_index=320+i, kind='validation_forward', rank=rank, update=None)
                for i, rank in enumerate(valid)]
    tr = _calls(record, HERE/'training_v1', entries)
    require(tr['generator'] == 'PCG64DXSM' and tr['stream_permutations'] == [2, 3]
            and tr['skipped_P1_order'] == lineage['training_order'] and tr['train_orders'] == orders
            and tr['train_order'] == record['training_order'] and tr['validation_order'] == list(valid),
            'Pinned worker owns exactly the next two PCG permutations')
    for stage_record, watch_record in ((qual, qw), (record, tw)):
        require(stage_record['environment']['versions'] == c.VERSIONS
                and watch_record['model_call_roster_sha256'] == stage_record['model_call_roster']['sha256'],
                'Same qualified library and watched complete call roster')
    sp, vp = HERE/'training_v1/terminal_seal.json', HERE/'training_v1/validation_predictions_sealed.json'
    seal, predictions = read(sp), read(vp)
    terminal = record['terminal_payload']
    require(terminal['sha256'] == args.terminal_payload_sha256
            and record['terminal_seal_sha256'] == args.terminal_seal_sha256
            and record['validation_predictions_sealed_sha256'] == args.validation_seal_sha256
            and record['artifacts'][str(sp)] == args.terminal_seal_sha256
            and record['artifacts'][str(vp)] == args.validation_seal_sha256,
            'Explicit caller-pinned sole payload and terminal/all40 seals')
    require(seal['terminal_epoch'] == seal['additional_epochs'] == 2 and seal['updates'] == 80
            and seal['training_graphs'] == 320 and seal['cumulative_train_passes'] == 3
            and seal['fresh_Adam'] and not seal['optimizer_resume'] and seal['parent_lineage'] == lineage
            and seal['cstar_raw_sha256'] == c.P1_CSTAR_SHA and seal['trainable_keys'] == list(keys)
            and seal['checkpoint_selection'] is False and seal['no_native_gradient_claim']
            and seal['native_score_measured'] is False and seal['source_bindings'] == record['source_bindings']
            and predictions['terminal_seal_sha256'] == args.terminal_seal_sha256
            and predictions['all_40_complete'] and predictions['no_native_gradient_claim']
            and predictions['native_score_measured'] is False,
            'Sole final payload sealed before all40 target-blind predictions')
    for item in record['validation_outputs'].values():
        require(item['dtype'] == 'float32' and item['shape'] == [16, 1024]
                and len(item['raw_sha256']) == 64, 'Sealed served output metadata; no prediction decoding')
    require(set(terminal['arrays']) == set(lineage['payload']['arrays']), 'Complete full terminal field inventory')
    for key, meta in terminal['arrays'].items():
        old = lineage['payload']['arrays'][key]
        require(meta['dtype'] == old['dtype'] == 'float32' and meta['shape'] == old['shape']
                and len(meta['raw_sha256']) == 64, 'Same-shaped FP32 terminal metadata')
    for folder in (HERE/'qualification_v1', HERE/'qualification_watch_v1'):
        for name in ('result.json', 'completion.json'):
            _put(inputs, folder/name, sha(folder/name))
    return helper, record, keys, sources, inputs


_changes = [
    ("PARENT = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'",
     "PARENT = NEW/'current_d50_native_objective_p1_v1/build_v1/estimator.py'", 1),
    ("PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'",
     "PARENT_SHA = '"+c.P1_CANDIDATE_SHA+"'", 1),
    ("ORIGINAL = NEW/'payload_response_rows_v1.npz'",
     "ORIGINAL = NEW/'current_d50_native_objective_p1_v1/training_v1/terminal_payload.npz'", 1),
    ("ORIGINAL_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'",
     "ORIGINAL_SHA = '"+c.P1_PAYLOAD_SHA+"'", 1),
    (c.P1_PROTOCOL_SHA, c.P2_PROTOCOL_SHA, 1),
    ("SCHEMA = 'current_d50_native_objective_p1_v1'", "SCHEMA = 'current_p1_native_objective_p2_v1'", 1),
    ('def admission(args):', 'def _inherited_admission(args):', 1),
    ("record['epochs'] == 1 and record['updates'] == record['planned_updates'] == 40",
     "record['epochs'] == 2 and record['updates'] == record['planned_updates'] == 80", 1),
    ("record['graphs_completed'] == 200", "record['graphs_completed'] == 360", 1),
    ("'current_d50_native_objective_p1_watch_v1'", "'current_p1_native_objective_p2_watch_v1'", 1),
    ('graph_index = 160+list(valid).index(int(rank))', 'graph_index = 320+list(valid).index(int(rank))', 1),
    ("('source', 'worker', 'common', 'watcher', 'training', 'training-watch')",
     "('source', 'worker', 'common', 'watcher', 'protocol', 'qualification', 'qualification-watch',\n"
     "                 'training', 'training-watch', 'terminal-payload', 'terminal-seal', 'validation-seal')", 1),
    ("require(np.asarray(literal[key], dtype=old.dtype).tobytes() == old.tobytes(), 'Original deployed P bytes')",
     "require(old.dtype == np.float32 and np.asarray(literal[key], dtype=old.dtype).shape == old.shape\n"
     "                and np.asarray(literal[key], dtype=old.dtype).tobytes() == old.tobytes(), 'Original deployed P bytes')\n"
     "        require(dict(shape=list(old.shape), dtype=str(old.dtype), raw_sha256=hashlib.sha256(old.tobytes()).hexdigest())\n"
     "                == training['parent_lineage']['payload']['arrays'][key], 'Sealed initial P1 payload bytes')", 1),
    ("all(np.asarray(replay[k], dtype=v.dtype).tobytes() == v.tobytes() for k, v in terminal.items())",
     "all(np.asarray(replay[k], dtype=v.dtype).shape == v.shape and\n"
     "                np.asarray(replay[k], dtype=v.dtype).tobytes() == v.tobytes() for k, v in terminal.items())", 1),
    ("terminal_epoch=1, updates=40, modified_assignment_only='_GNN_LITERAL',",
     "terminal_epoch=2, updates=80, additional_epochs=2, cumulative_train_passes=3,\n"
     "        fresh_Adam=True, optimizer_resume=False, parent_lineage=training['parent_lineage'],\n"
     "        qualification_result_sha256=args.qualification_sha256,\n"
     "        qualification_watch_sha256=args.qualification_watch_sha256,\n"
     "        terminal_seal_sha256=args.terminal_seal_sha256,\n"
     "        validation_predictions_sealed_sha256=args.validation_seal_sha256,\n"
     "        build_adapter=BUILD_ADAPTER_PROOF, modified_assignment_only='_GNN_LITERAL',", 1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Inherited entrypoint suppressed; this adapter owns the sole explicit entrypoint.\n', 1),
]
_derived, BUILD_ADAPTER_PROOF = c.splice_source(c.P1/'build.py', P1_BUILDER_SHA, _changes)
exec(compile(_derived, str(Path(__file__).resolve())+':inherited', 'exec'), globals())


if __name__ == '__main__':
    raise SystemExit(main())
