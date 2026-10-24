"""Inert, inverse-verified P1 receipt owner with terminal-P1 admission.

No numerical import or stage construction occurs on import. Truth bindings
from historical results are deliberately excluded from lineage admission.
"""
import hashlib
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent/'current_d50_native_objective_p1_v1'
P1_COMMON_SHA = '567ccbae6caf516043d72d54a9073cc60e7fafeac31934d29379ef578fea0a7f'
P1_WORKER_SHA = 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c'
P1_WATCH_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
P1_PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
P2_PROTOCOL_SHA = 'b2ceb8ac16086e19a5630658df7fb43afaa17b47d761267a1d84ffadce8d9b16'
P1_PAYLOAD_SHA = '6b7b5ab8e357e26598cf64febd9f8b6993c3a7508efe6565d74ff3f02d816b65'
P1_CANDIDATE_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
P1_CSTAR_SHA = 'd622c0aed6d62e0453648df32cfe46da73b220163fb5fe32877d430be1540227'
P1_ROSTER_SHA = '20cf90afd23547f75fedf31ef0fdb73c29c673e1bae9b861bbde73aef800b132'
P1_RECEIPTS = {
    'training_v1': ('aaaef3d5f5a3c60583af779895b8db4f769276c64cb0cc032e93755169dbab91',
                    '6964011b2262a70e56fafd74b040432065513706c3fde5586341a09b85fd0253'),
    'training_watch_v1': ('d25b993ab493a3027696997c59189e28c209337c945589cebc10623136754796',
                          '03c3cdd88861830e42eb5bc4f698ca94f6d1ec0126bb397b1f66ceb0d8837fe1'),
    'build_v1': ('877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1',
                 'cdcb0a934224f242cad404dec3bae7eba60a712d8bd430ace0c7ab7111faea5c'),
    'score2_v1': ('4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
                  'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587'),
    'score2_watch_v1': ('c9d015f7ed6b1da7db58bcf237582874042dd2d1e3315e72e7c55b5f1b9afa34',
                        'fa220095fa711c7e80a9268aeb2e0ef410fb94444e775813e758d52eaa01d4f5'),
    'panel_v1/score': ('81ebbb54207e4f0060aaa1a91320c17de89b6fdd53e7d5326c17993829e6e526',
                       'ee2df444d7e3984993e6367f707b3ffd3a8408a182f85ca9dc7701341167f309'),
    'panel_v1/score_watch': ('46e3e7e910c5b4b50f1e8c41a99c417e7a751f82fd7cc4b37fe6e2576f3f10b9',
                             '2e26c1b9c1aa38ec98148a835dbf28b31f06288b3abb27f0927f1452ce0ff50f'),
}


def splice_source(path, expected, replacements):
    """Exact counted substitutions, then byte-for-byte inverse before compile."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Frozen adapter owner changed: '+str(path))
    original = raw.decode('utf-8')
    changed = original
    for old, new, count in replacements:
        if not old or not new or changed.count(old) != count:
            raise RuntimeError('Nonunique source adapter splice: '+old[:100])
        changed = changed.replace(old, new)
    restored = changed
    for old, new, count in reversed(replacements):
        if restored.count(new) != count:
            raise RuntimeError('Nonunique inverse splice: '+new[:100])
        restored = restored.replace(new, old)
    if restored.encode('utf-8') != raw:
        raise RuntimeError('Source inverse is not byte exact')
    return changed, dict(parent_source_sha256=expected, inverse_source_byte_exact=True,
                         substitution_count=len(replacements))


def parent_lineage(pin):
    """pin(path, digest, source=False); metadata/hashes only, no target inventory."""
    seen = {}
    def own(path, digest, source=False):
        path = Path(path).resolve()
        previous = seen.get(str(path))
        require(previous is None or previous == digest, 'Conflicting P1 lineage binding')
        # Callback owns source classification and may use a cached digest.
        if previous is None or source:
            pin(path, digest, source)
        seen[str(path)] = digest
    for name, digest in (('common.py', P1_COMMON_SHA), ('worker.py', P1_WORKER_SHA),
        ('watch.py', P1_WATCH_SHA), ('PROTOCOL.md', P1_PROTOCOL_SHA),
        ('build.py', '57f87081cdcc742266bc8487ef5ba11358bdf25e8341b9b1f4dd4458f06da54c'),
        ('build_v1/estimator.py', P1_CANDIDATE_SHA)):
        own(P1/name, digest, True)
    records = {}
    for folder, (digest, end_digest) in P1_RECEIPTS.items():
        own(P1/folder/'result.json', digest); own(P1/folder/'completion.json', end_digest)
        record, end = read(P1/folder/'result.json'), read(P1/folder/'completion.json')
        require(record['passed'] is True and record['completed'] is True and record['hashes_unchanged'] is True
                and end['passed'] is True and end['completed'] is True and end['result_sha256'] == digest,
                'Completed preserved P1 lineage')
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'bindings', 'artifacts'):
            for path, item in record.get(group, {}).items():
                own(path, item['sha256'] if isinstance(item, dict) else item, group == 'source_bindings')
        records[folder] = record
    train, build = records['training_v1'], records['build_v1']
    require(train['schema'] == build['schema'] == 'current_d50_native_objective_p1_v1'
            and train['stage'] == 'train' and build['stage'] == 'build'
            and train['epochs'] == 1 and train['updates'] == train['planned_updates'] == 40
            and train['graphs_completed'] == 200 and not train['validation_targets_accessed']
            and not train['checkpoint_selection'] and train['no_native_gradient_claim']
            and build['candidate_sha256'] == P1_CANDIDATE_SHA
            and build['training_result_sha256'] == P1_RECEIPTS['training_v1'][0], 'Sole retained P1 terminal')
    payload = train['terminal_payload']
    require(payload == build['terminal_payload'] and payload['sha256'] == P1_PAYLOAD_SHA
            and Path(payload['path']).resolve() == P1/'training_v1/terminal_payload.npz'
            and train['cstar_raw_sha256'] == P1_CSTAR_SHA, 'Exact P1 terminal payload/cstar lineage')
    own(payload['path'], P1_PAYLOAD_SHA)
    seals = []
    for name, digest, key in (
        ('terminal_seal.json', 'f3f733c1e65723ad2963b4df36fd2b89bda21905fedd79937cb2dac7b642fa71', 'terminal_seal_sha256'),
        ('validation_predictions_sealed.json', 'bfc645b5eba01d9d4255eed5361b8c41afce89a040806dc1064b3485193d2b86', 'validation_predictions_sealed_sha256')):
        own(P1/'training_v1'/name, digest)
        require(train[key] == digest, 'P1 terminal seal ownership')
        seals.append(read(P1/'training_v1'/name))
    require(all(x['terminal_payload'] == payload and not x['validation_targets_accessed'] for x in seals)
            and seals[1]['validation_outputs'] == train['validation_outputs'], 'P1 all40 prediction seal')
    for child, supervisor in (('training_v1','training_watch_v1'), ('score2_v1','score2_watch_v1'),
                              ('panel_v1/score','panel_v1/score_watch')):
        watch = records[supervisor]
        require(watch['child_result_sha256'] == P1_RECEIPTS[child][0]
                and watch['child_completion_sha256'] == P1_RECEIPTS[child][1]
                and watch['whole_owned_tree_retired'] and watch['child_execution_completed'], 'P1 external seal')
    score, panel = records['score2_v1'], records['panel_v1/score']
    require(score['schema'] == 'current_d50_native_objective_p1_score_v2' and score['stage'] == 'score'
            and score['integrity_passed'] and score['advances_to_independent14']
            and score['objective_improved'] and score['wins_gate']
            and panel['schema'] == 'current_d50_native_objective_p1_independent14_v1'
            and panel['mode'] == 'score' and panel['integrity_passed'] and panel['retain_local_component'],
            'Preserved P1 native40 and BOTH independent-group advancement')
    item = train['model_call_roster']
    require(Path(item['path']).resolve() == P1/'training_v1/model_call_roster.json'
            and item['sha256'] == P1_ROSTER_SHA, 'Exact original PCG stream receipt')
    own(item['path'], P1_ROSTER_SHA)
    original_roster = read(item['path'])
    require(original_roster['seed'] == SEED and original_roster['generator'] == 'PCG64DXSM'
            and original_roster['train_order'] == train['training_order'], 'Original first permutation ownership')
    return dict(payload=payload, training_order=train['training_order'], model_call_roster=item,
                candidate_sha256=P1_CANDIDATE_SHA, training_result_sha256=P1_RECEIPTS['training_v1'][0],
                training_watch_sha256=P1_RECEIPTS['training_watch_v1'][0],
                native40_score_sha256=P1_RECEIPTS['score2_v1'][0],
                independent14_score_sha256=P1_RECEIPTS['panel_v1/score'][0],
                cstar_raw_sha256=P1_CSTAR_SHA, optimizer_state_loaded=False)


def _admit_p1(stage, cpu):
    lineage = parent_lineage(lambda p, d, source=False: stage.pin(
        p, d, 'source_bindings' if source else 'input_bindings'))
    stage.parent_lineage, stage.payload = lineage, lineage['payload']
    stage.cotangents = cpu['cotangents']
    stage.record.update(parent_lineage=lineage, parent_payload_sha256=P1_PAYLOAD_SHA,
        parent_candidate_sha256=P1_CANDIDATE_SHA, fresh_Adam=True, optimizer_resume=False,
        proposal_cost_weights_owner='archived d50 cold; not measured P1 or changed-candidate costs',
        common_adapter=COMMON_ADAPTER_PROOF)


_old_payload = """        self.cotangents, self.payload = cpu['cotangents'], cpu['payloads']['null']
        self.reference35 = old['outputs']['35']['null']
        self.parent35 = old['parent_outputs']['35']
        require(self.payload['sha256'] == PAYLOAD_SHA and self.reference35['parent_byte_exact']
                and self.reference35['prediction']['raw_sha256'] == self.parent35['raw_sha256'],
                'Exact original d50 payload and old same-backend35 control')"""
_changes = [
    (P1_PROTOCOL_SHA, P2_PROTOCOL_SHA, 1),
    ("PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'",
     "PAYLOAD_SHA = '"+P1_PAYLOAD_SHA+"'", 1),
    ("SCHEMA = 'current_d50_native_objective_p1_v1'", "SCHEMA = 'current_p1_native_objective_p2_v1'", 1),
    (_old_payload, '        _admit_p1(self, cpu)', 1),
    ("len(qual['graph_records']) == 4", "len(qual['graph_records']) == 5", 1),
    ("watch['completed_graph_count'] == 4", "watch['completed_graph_count'] == 5", 1),
    ("'current_d50_native_objective_p1_watch_v1'", "'current_p1_native_objective_p2_watch_v1'", 1),
]
_derived, COMMON_ADAPTER_PROOF = splice_source(P1/'common.py', P1_COMMON_SHA, _changes)
exec(compile(_derived, str(Path(__file__).resolve())+':inherited', 'exec'), globals())
