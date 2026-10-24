"""Source-only adapter of the qualified watch for six fixed local contexts.

No evaluation-B access, numerical imports, source mutation, or automatic next
stage. Full source inversion preserves the actual resource/retirement loop.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
DESIGN_SHA = '360add35b70970cba3af6181e7e786ceb28187c89a07813cba8a4143823a6f31'
CAPTURE_SEALS = {
    'capture_v1': (
        '7867fb6b645d042d88fb359d7f831708d2e7feb37e26d845309805aec4796804',
        '0bdb594c411d07ffe9c1fd5c21f394c51d8d558d147aa18752e4067b75021c54',
        '5e5589cbb3fa91d8ef8c3d6dae6587e68c4f174283c0e0e3ec049a44f15cbb78'),
    'capture_watch_v1': (
        '790b0d8ad7e88df9aa258d16de35ec6f5638c3f202f3a0cae51f46812abb9a34',
        '4fd68816a97ae3fb3ee905923fc55eb2fb258eaa531e6359ae927c893551f04e',
        'aa5e395b811694fdbbedee8141cf62238aef6e017fb1c077b4a4cb12d6c703ed')}
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
ROSTER = ('tiny', 'N', 'T', 'Q', 'TQ', 'ALL-local')


def source_and_proof():
    original = OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != OWNER_SHA:
        raise RuntimeError('Exact independently qualified resource-watch source')
    own_start = original.index('    own = {HERE/')
    own_end = original.index('    require(all(re.fullmatch', own_start)
    prep_start = original.index("        probe = load(HERE/'probe.py'")
    prep_end = original.index('        env = dict(os.environ', prep_start)
    gate_start = original.index("        require(record['candidate_sha256'] == result['candidate_sha256']")
    gate_end = original.index("        for group in ('source_bindings'", gate_start)
    edits = [
        ('CAP, RSS, PROGRESS = 300.,8_000_000_000,120.',
         'CAP, RSS, PROGRESS = 600.,8_000_000_000,120.'),
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",
         "KEYS = ('source','worker','protocol','spec','tiny','tiny-protocol')"),
        (original[own_start:own_end],
         "    own = {HERE/'watch_consumer.py':args.source_sha256,HERE/'consumer.py':args.worker_sha256,\n"
         "        HERE/'CONSUMER_PROTOCOL.md':args.protocol_sha256,HERE/'CONSUMER_WATCH_SPEC.md':args.spec_sha256,\n"
         "        HERE/'tiny.py':args.tiny_sha256,HERE/'TINY_PROTOCOL.md':args.tiny_protocol_sha256,\n"
         "        HERE/'DESIGN.md':_JCA_DESIGN_SHA,TRANSPORT:TRANSPORT_SHA,\n"
         "        _JCA_WATCH_OWNER:_JCA_WATCH_OWNER_SHA}\n"),
        ("HERE/'probe_v1',HERE/'probe_watch_v1'", "HERE/'consumer_v1',HERE/'consumer_watch_v1'"),
        ("schema='current_comp_shared_old384_native37_watch_v1'",
         "schema='current_comp_joint_cell_consumer_watch_v1'"),
        ("numerical_imports=False,truth_accessed=False,score_measured=False,uploaded=False,",
         "numerical_imports=False,supply_A_byte_authentication_only=True,evaluation_B_accessed=False,\n"
         "        score_measured=False,uploaded=False,"),
        ("retired_calls=0,ranks=[37,37]", "retired_calls=0,ranks=[35]*6,roster=list(_JCA_ROSTER)"),
        ("'Whole300s/next retired call120s'", "'Whole600s/next fully retired context120s'"),
        ("        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')\n"
         "        require(not any(int(row['pid']) not in allowed and 'flopscope_server' in str(row.get('command',''))\n"
         "            for row in sample['rows']), 'No foreign underscore-named RPC server')"),
        ("index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37'",
         "index < 6 and item['call'] == index and item['rank'] == 35,'Exact fixed tiny/N/T/Q/TQ/ALL roster'"),
        ("record['call'] == index and record['rank'] == 37 and record['context_retired']",
         "record['call'] == index and record['rank'] == 35 and record['context_retired']\n"
         "                        and record['arm'] == _JCA_ROSTER[index]"),
        ('sha256=digest,call=index,rank=37)',
         'sha256=digest,call=index,rank=35,arm=_JCA_ROSTER[index])'),
        (original[prep_start:prep_end],
         "        probe = load(HERE/'consumer.py','joint_cell_consumer_metadata',args.worker_sha256)\n"
         "        from types import SimpleNamespace\n"
         "        ns = SimpleNamespace(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256,\n"
         "            protocol_sha256=args.protocol_sha256,watch_spec_sha256=args.spec_sha256,\n"
         "            tiny_sha256=args.tiny_sha256,tiny_protocol_sha256=args.tiny_protocol_sha256)\n"
         "        admitted = probe.admission(ns,tick)\n"
         "        context = admitted[0] if isinstance(admitted,tuple) else admitted\n"
         "        for group in ('source_bindings','input_bindings'):\n"
         "            require(isinstance(context[group],dict) and context[group], 'Complete inert admission inventory')\n"
         "            result[group].update(context[group])\n"
         "        _jca_capture_gate(read,sha,require,result,tick)\n"
         "        result['candidate_sha256'] = _JCA_CANDIDATE_SHA\n"
         "        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No monitor numerics')\n"
         "        argv = [str(OFFPY),'-B','-u',str(HERE/'consumer.py'),'--run',\n"
         "            '--source-sha256',args.worker_sha256,'--watcher-sha256',args.source_sha256,\n"
         "            '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256,\n"
         "            '--tiny-sha256',args.tiny_sha256,'--tiny-protocol-sha256',args.tiny_protocol_sha256]\n"),
        ("result['retired_calls'] == 2,'Two complete calls and exit0'",
         "result['retired_calls'] == 6,'Exactly six completely retired contexts and exit0'"),
        (original[gate_start:gate_end],
         "        _jca_consumer_gate(read,sha,require,record,result,tick)\n"),
        ("if p != TRANSPORT)", "if p not in (TRANSPORT,_JCA_WATCH_OWNER))"),
        ("require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not\n"
         "        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No supervisor target access')",
         "require(path.is_relative_to(P2) and path.name != 'key.txt' and 'truthpost' not in path.parts\n"
         "        and 'covariance_replica_truth_v1' not in path.parts and not\n"
         "        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),\n"
         "        'Supervisor may authenticate declared A bytes, never access evaluation B or other targets')")]
    changed = original
    spans = []
    for old, new in edits:
        if changed.count(old) != 1:
            raise RuntimeError('Unique consumer-watch splice: '+old[:100])
        pos = changed.index(old)
        changed = changed[:pos]+new+changed[pos+len(old):]
        spans.append((pos, old, new))
    inverse = changed
    for pos, old, new in reversed(spans):
        if inverse[pos:pos+len(new)] != new:
            raise RuntimeError('Exact consumer-watch positional inverse')
        inverse = inverse[:pos]+old+inverse[pos+len(new):]
    if inverse != original:
        raise RuntimeError('Complete qualified-watch inverse')
    return changed, dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        resource_observation_and_retirement_loop_unchanged=True)


def capture_gate(read, sha, require, result, tick):
    rows = {}
    for folder, expected in CAPTURE_SEALS.items():
        paths = [HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        actual = tuple(sha(p,tick) for p in paths)
        require(actual == expected, 'Exact actual target-blind capture three-seal chain')
        a,b,c = [read(p) for p in paths]
        require(all(x.get('passed') is True and x.get('completed') is True for x in (a,b,c))
            and b['result_sha256'] == c['result_sha256'] == actual[0]
            and c['completion_sha256'] == actual[1], 'Capture positively completed before oracle work')
        result['input_bindings'].update({str(p):d for p,d in zip(paths,actual)})
        rows[folder] = a
    capture, watch = rows['capture_v1'], rows['capture_watch_v1']
    require(capture['candidate_sha256'] == watch['candidate_sha256'] == CANDIDATE_SHA
        and capture['baseline_output_byte_exact'] and capture['all_original_operations_exact']
        and capture['all_capture_arrays_released'] and capture['all_parameter_and_estimator_owners_retired']
        and capture['truth_accessed'] is False and watch['whole_owned_tree_retired'] and watch['cleanup_passed']
        and watch['maximum_combined_rss_bytes'] < 8_000_000_000
        and watch['maximum_interval_s'] <= 1 and watch['maximum_swap_growth_bytes'] == 0
        and [watch['child_'+key] for key in ('result_sha256','completion_sha256','finalization_sha256')]
            == list(CAPTURE_SEALS['capture_v1']), 'Actual same-current capture and complete resource boundary')
    result['capture_seals'] = CAPTURE_SEALS


def consumer_gate(read, sha, require, record, result, tick):
    """No scientific criterion is evaluated by this operational supervisor."""
    require(record['schema'] == 'current_comp_joint_cell_consumer_v1'
        and record['candidate_sha256'] == CANDIDATE_SHA
        and record['retired_call_count'] == 6 and record['roster'] == list(ROSTER)
        and record['tiny_qualified'] and record['null_byte_exact']
        and record['own_complete_bills'] and record['all_parameter_and_estimator_owners_retired']
        and record['supply_A_decoded_after_null_retirement']
        and record['evaluation_B_accessed'] is False and record['score_measured'] is False
        and record['parameter_updates'] == 0, 'Fixed complete local consumers, no fit or B leakage')
    bills = [read(HERE/'consumer_v1'/f'call_{i:02d}_bill.json') for i in range(6)]
    own = [int(x['complete_research_flops']) for x in bills]
    require(all(int(x['flops'])+int(x['input_flops']) == f for x,f in zip(bills,own)),
        'Immutable physical meter plus independently billed input preparation')
    require(all(int(x['operation_ledger']['total_flops']) == int(x['flops'])
        and int(x['operation_ledger']['operation_count']) == int(x['operation_count']) for x in bills),
        'Each immutable physical bill reconciles to its complete original operation ledger')
    require(0 < own[0] <= 10_000_000 and all(0 < f <= 250_000_000_000 for f in own[1:5])
        and 0 < own[5] <= 5_000_000_000
        and sum(own)+403_695_296_665 <= 1_425_000_000_000,
        'Every own partial bill plus actual capture fits original complete diagnostic ceiling')
    require(all(float(x['wall_s']) < 120 for x in bills), 'Every actual owned context stays below120s')
    result.update(complete_consumer_flops=sum(own),capture_plus_consumer_flops=sum(own)+403_695_296_665,
        source_and_null_qualified=True,evaluation_B_accessed=False,scientific_outcome_measured=False)


def main():
    text, proof = source_and_proof()
    namespace = dict(__name__='joint_cell_qualified_consumer_watch',__file__=str(HERE/'watch_consumer.py'),
        _JCA_WATCH_OWNER=OWNER,_JCA_WATCH_OWNER_SHA=OWNER_SHA,_JCA_DESIGN_SHA=DESIGN_SHA,
        _JCA_CANDIDATE_SHA=CANDIDATE_SHA,_JCA_ROSTER=ROSTER,
        _jca_capture_gate=capture_gate,_jca_consumer_gate=consumer_gate,WATCH_SOURCE_PROOF=proof)
    exec(compile(text,'<fixed-six-consumer-qualified-watch>','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
