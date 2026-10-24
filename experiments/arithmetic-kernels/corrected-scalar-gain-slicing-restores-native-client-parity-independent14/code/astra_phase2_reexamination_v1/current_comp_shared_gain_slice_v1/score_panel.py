"""Saved c2/OpenBLAS independent14 score, with historical APUSH comparator."""
import ast
import hashlib
from pathlib import Path
import runpy

_GS_HERE = Path(__file__).resolve().parent
_GS_SHARED = _GS_HERE.parent / 'current_comp_shared_old384_v1'
_GS_OWNER = _GS_SHARED / 'score_panel_v2.py'
_GS_OWNER_SHA = '91e08e120fa9d8b2c1b097c68d69a8dbf8c672a4b8a5d20ddfe8587a771e8fe0'
_GS_PINS = {
    _GS_OWNER: _GS_OWNER_SHA,
    _GS_SHARED / 'watch_score_panel_v2.py': '6500fad9fffd81a82bd5fa4861210ad43d35c6d65b45bbf498dc153087a4e223',
    _GS_SHARED / 'SCORE_PANEL_V2_PROTOCOL.md': '1dfbe1c3ad950c2abf860e385a3bd75f09ef6c7da0085f7c924c2f14d3a07f8e',
    _GS_SHARED / 'SCORE_PANEL_V2_WATCH_SPEC.md': '9c81c336ca49bb255eeb69e37ca0cf1f42e0dd58f830488023a40cad119622fb',
    _GS_HERE / 'build.py': 'd417e12be774fae3cf33c2fd5eb17f098a43827efc556920100b4ee2bd72dd15',
    _GS_HERE / 'BUILD_PROTOCOL.md': '7598dd79a817cfb68a792be005fd2c7e9ff60c7b29bbb7dd1e1a9dd4753cef11',
    _GS_HERE / 'native_panel.py': '4db46ec387d72950bf63dc0408d3fac56f37452b7b5e8aa1502a1fe3e5b79707',
    _GS_HERE / 'watch_native_panel.py': 'e9b9774f6e5325b0b94725558ff2c319411575df738a1b94822fd7761a3c211b',
    _GS_HERE / 'NATIVE_PANEL_PROTOCOL.md': '070dcb92b066b4537f399874143c4a69db8065362679a18d2074b39ece8ffb8a',
    _GS_HERE / 'NATIVE_PANEL_WATCH_SPEC.md': '03a9cd2c4802ccd6d40d850528a755813dc022bedef0309329f46cae6b1fa82a',
    _GS_HERE / 'remote_watch_v2.py': '4a55a8a97e7d96d7abdb6427e06a3e73c8091ab30a9c98a50d1aafd897c7b8b0',
    _GS_HERE / 'REMOTE_PAIR_V2_SPEC.md': 'b6e844ead8ae6cf8c701793bd98e05d7c925e6b25953a50fc500b0b0cf0d1a92',
    _GS_HERE / 'client_v2.py': '2b5818ae5013c9a5e460437fff7770a9059f1139732f627c01253e3cba484e40',
}
if hashlib.sha256(_GS_OWNER.read_bytes()).hexdigest() != _GS_OWNER_SHA:
    raise RuntimeError('Frozen positive original-precision independent14 score owner')
_GS_PARENT = runpy.run_path(str(_GS_OWNER), run_name='_inert_gain_score_parent')
_V1_SOURCE_PINS = dict(_GS_PARENT['_V1_SOURCE_PINS'])
_GS_NATIVE = None
BACKEND_POLICY = dict(
    candidate_inference_backend='actual production OpenBLAS 0.3.31.188.0, NumPy2.4.6, one BLAS thread',
    comparator_inference_backend='historical APUSH/Accelerate acquisition, unchanged positive original14',
    comparator_backend_matched=False,
    comparison_scope='Fixed original J and subgroup selection comparison; not a backend-isolated causal effect.',
    candidate_outputs_and_complete_flops_both_measured=True,
    current35_native_RPC_vector_and_complete_bill_replay_required=True,
    old_4d98_objective_inherited=False,
    historical_native40_selection_only=True,
    current_candidate_native40_qualified=False,
    current_candidate_production_five_qualified=False,
    submission_authorized=False)


def _gain_native_api(admit):
    global _GS_NATIVE
    path = HERE / 'native_panel.py'
    admit.pin(path, _GS_PINS[path])
    if _GS_NATIVE is None:
        _GS_NATIVE = SimpleNamespace(**runpy.run_path(str(path), run_name='_inert_gain_saved_panel_verifier'))
    require(not {'numpy','torch','flopscope','threadpoolctl','scipy'} & set(sys.modules),
        'Native metadata verifier never imports the numerical backend')
    return _GS_NATIVE


def _gain_acquisition(args, admit, env, control, control_seals):
    n,ns = actual_chain(admit,HERE/'native_panel_v1',args.native_sha256,1200,8_000_000_000)
    w,ws = actual_chain(admit,HERE/'native_panel_watch_v1',args.native_watch_sha256,1200,8_000_000_000,watched=True)
    env['linked'](w,ns); admit.inventory(n); admit.inventory(w)
    require(n['schema'] == 'current_comp_shared_gain_slice_native_panel_v1'
        and w['schema'] == 'current_comp_shared_gain_slice_native_panel_watch_v1'
        and n['candidate_sha256'] == w['candidate_sha256'] == CANDIDATE
        and n['ranks'] == w['ranks'] == list(SEQUENCE)
        and n['full_model_calls'] == n['retired_call_count'] == w['retired_calls'] == 15
        and n['initialization_flops'] == INIT and n['truth_accessed'] is False and n['score_measured'] is False
        and n['parameter_updates'] == 0 and n['controls'] == control['trials']
        and n['fixture_bindings'] == control['fixture_bindings']
        and n['control_seals'] == dict(native=list(control_seals['APUSH_native'].values()),
            watch=list(control_seals['APUSH_native_watch'].values())),
        'Actual complete c2 fixed15 and exact original APUSH physical control boundary')
    for name,digest in (('native_panel.py',args.native_source_sha256),
            ('watch_native_panel.py',args.native_watch_source_sha256),
            ('NATIVE_PANEL_PROTOCOL.md',args.native_protocol_sha256),
            ('NATIVE_PANEL_WATCH_SPEC.md',args.native_watch_spec_sha256)):
        path = HERE/name
        require(digest == _GS_PINS[path] == n['source_bindings'].get(str(path)) == w['source_bindings'].get(str(path)),
            'Exact already-reviewed current acquisition owner')
    api = _gain_native_api(admit)
    require(n['panel_source_inverse_proof'] == api.PANEL_SOURCE_INVERSE_PROOF
        and all(n[key] == w[key] for key in api.PANEL_LINEAGE_KEYS)
        and all(n.get(k) is True for k in ('gain_slice_pair_qualified','historical_native40_selection_only',
            'all16_domains_qualified','all47_selectors_qualified','own_complete_bills','repeat_byte_exact',
            'operation_ledgers_verified','operation_counts_reported_not_rebased','APUSH_panel_comparator_qualified',
            'panel_repeat35_physics_exact','production_same_input_controls_sealed','gain_pair35_cold_exact',
            'gain_pair35_repeat_physics_exact','same_production_backend_qualified','backend_actual_origins_qualified'))
        and all(n.get(k) is False for k in ('current_candidate_native40_qualified',
            'current_candidate_production_five_qualified','APUSH_comparator_backend_matched',
            'historical_production_five_reused','cross_backend_APUSH_prefix_equality_required'))
        and n['unaffected_prefix_checks'] == [], 'Explicit current backend and limited eligibility, not transferred old science')
    pair,ps = actual_chain(admit,HERE/'pair_watch_v2',args.pair_sha256,240,8_000_000_000,watched=True)
    native35,ns35 = actual_chain(admit,HERE/'native_v1',pair['native_seals']['result_sha256'],300,8_000_000_000)
    build,bs = actual_chain(admit,HERE/'build_v1',args.build_sha256,60,2_000_000_000)
    for record in (pair,native35,build):
        admit.inventory(record)
    lineage = n['pair_prerequisite_seals']
    require(lineage['pair_watch'] == ps == n['production_runtime_seals']
        and lineage['native'] == ns35 == pair['native_seals']
        and lineage['build'] == bs == pair['build_seals']
        and pair['schema'] == 'current_comp_shared_gain_slice_pair_watch_v2'
        and pair['candidate_sha256'] == native35['candidate_sha256'] == build['candidate_sha256'] == CANDIDATE
        and pair['full_model_calls'] == 2 and pair['fresh_native_calls'] == 0
        and pair['joined_native_remote_calls'] == 4 and pair['current_pair_qualified']
        and pair['prior_native_phase_resource_qualified'] and pair['previous_pair_watch_still_failed']
        and len(pair['remote_records']) == len(pair['prior_native_observed_markers']) == 2
        and pair['native_markers'] == [] and pair['pair_vector_gate_passed'] and pair['pair_timing_gate_passed']
        and pair['own_complete_bills'], 'Actual joined pair, positive child/build, no failed pair-v1 substitution')
    negative = HERE/'pair_watch_v1'
    paths = [negative/name for name in ('result.json','completion.json','finalization.json')]
    hashes = [lineage['prior_negative_pair'][key] for key in
        ('result_sha256','completion_sha256','finalization_sha256')]
    for path,digest in zip(paths,hashes):
        admit.pin(path,digest)
        require(pair['input_bindings'].get(str(path)) == digest, 'New positive pair owns actual earlier negative evidence')
    old,e,f = map(read,paths)
    require(all(record.get('passed') is False and record.get('completed') is False for record in (old,e,f))
        and e['result_sha256'] == f['result_sha256'] == hashes[0] and f['completion_sha256'] == hashes[1]
        and old['native_seals'] == ns35 and old['build_seals'] == bs and old['remote_records'] == [],
        'Old outer watch remains negative; no retrospective rescue')
    proof = n['composition_proof']; observer = n['passive_observer_source_proof']
    require(proof == n['build_proof'] == w['composition_proof'] == build['proof']
        and proof['candidate_sha256'] == CANDIDATE and proof['parent_sha256'] == HISTORICAL_CANDIDATE
        and all(proof.get(k) is True for k in ('one_span_gain_slice','whole_source_inverse_verified',
            'all_learned_literals_byte_exact','no_new_parameter_or_head','shared_basis_and_physics_source_unchanged'))
        and proof['candidate_imported'] is False
        and proof['before'] == 'fnp.tanh(P["gain"][l])' and proof['after'] == 'fnp.tanh(P["gain"][l:l+1])'
        and n['paid_loop_proof']['whole_loop_inverse']
        and n['paid_loop_proof']['no_billing_input_reset_materialization_or_repeat_change']
        and n['paid_loop_proof']['owner_sha256'] == '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
        and all(observer[k] is True for k in ('broad_push_observer_inverse_verified','unchanged_all16_all47_capture',
            'exact_shape_specific_D6_and_seven_D5_tariffs','no_added_participant_array_operations')),
        'Actual sole source correction and original complete paid loop/observer, without rebuilding')
    verify_calls(n,HERE/'native_panel_v1',admit,env,shared=True,watch=w)
    checked = api.verify_native(n,tick=admit.tick)
    require(all(n[k] == w[k] == value for k,value in checked.items())
        and w['backend_attestation'] == n['backend_attestation'], 'Exact retired native verifier and paired-watch checks')
    admit.result.update(gain_score_source_proof=GAIN_SCORE_SOURCE_PROOF,
        backend_policy=dict(BACKEND_POLICY),historical_native40_selection_only=True,
        current_candidate_native40_qualified=False,current_candidate_production_five_qualified=False,
        current_gain_pair_qualified=True,current35_same_source_full_vector_and_cost_replay=True,
        APUSH_comparator_backend_matched=False,old_4d98_objective_inherited=False,
        all_current_native_outputs_and_F_production_backend=True)
    return n,dict(native=ns,native_watch=ws,gain_pair_watch=ps,gain_pair_native=ns35,
        gain_build=bs,prior_negative_pair=lineage['prior_negative_pair'])


def _gain_output_boundary(outputs, native, result, tick):
    require(result['gain_score_source_proof'] == GAIN_SCORE_SOURCE_PROOF
        and result['backend_policy'] == BACKEND_POLICY
        and native['gain_pair35_cold_exact'] and native['gain_pair35_repeat_physics_exact']
        and native['cross_backend_APUSH_prefix_equality_required'] is False,
        'No cross-backend APUSH prefix assertion; qualified current35 replay instead')
    prior = read(HERE/'native_v1/result.json')
    for panel_call,pair_call in ((0,0),(14,1)):
        require(hashlib.sha256(outputs['shared'][panel_call].tobytes()).hexdigest() ==
            prior['trials'][pair_call]['output_bytes_sha256']
            and native['trials'][panel_call]['flops'] == prior['trials'][pair_call]['flops'],
            'Loaded current35 full vectors and own complete F replay actual qualified pair')
        tick()


def _gain_watch_setup(args):
    child = {k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256')
        for k in KEYS if k not in ('source','watcher')}
    child.update(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256)
    ns = SimpleNamespace(**child)
    inputs = {HERE/'native_panel_v1/result.json':args.native_sha256,
        HERE/'native_panel_watch_v1/result.json':args.native_watch_sha256,
        HERE/'build_v1/result.json':args.build_sha256,HERE/'pair_watch_v2/result.json':args.pair_sha256,
        SHARED/'native40_v1/result.json':args.native40_sha256,
        SHARED/'native40_watch_v1/result.json':args.native40_watch_sha256,
        SHARED/'score40_v1/result.json':args.score40_sha256,
        SHARED/'score40_watch_v1/result.json':args.score40_watch_sha256}
    return own_sources(ns),inputs,HERE/'score_panel_v1',HERE/'score_panel_watch_v1'


def watch_contract(args,result,folder,check_sha,tick):
    _gain_original_watch_contract(args,result,folder,check_sha,tick)
    require(result['gain_score_source_proof'] == GAIN_SCORE_SOURCE_PROOF
        and result['backend_policy'] == BACKEND_POLICY
        and result['current_gain_pair_qualified'] and result['historical_native40_selection_only']
        and result['current35_same_source_full_vector_and_cost_replay']
        and result['all_current_native_outputs_and_F_production_backend']
        and result['current_candidate_native40_qualified'] is False
        and result['current_candidate_production_five_qualified'] is False
        and result['APUSH_comparator_backend_matched'] is False
        and result['old_4d98_objective_inherited'] is False
        and result['lineage']['gain_build']['result_sha256'] == args.build_sha256
        and result['lineage']['gain_pair_watch']['result_sha256'] == args.pair_sha256,
        'Same limited historical-comparator objective and actual current pair/build authority')
    require(result['lineage']['native']['result_sha256'] == args.native_sha256
        and result['lineage']['native_watch']['result_sha256'] == args.native_watch_sha256
        and all(result['lineage']['development'][key]['result_sha256'] == getattr(args,key+'_sha256')
            for key in ('native40','native40_watch','score40','score40_watch')),
        'Exact explicit current acquisition and historical-only development seals')


def derive_source():
    for path,digest in _GS_PINS.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Exact scorer/acquisition source owner: '+str(path))
    original = _GS_PARENT['_DERIVED_SOURCE']
    def function(name):
        nodes = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise RuntimeError('Unique score helper '+name)
        n=nodes[0]; return ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    historical = function('qualified40')
    if historical.count("r['candidate_sha256'] == CANDIDATE") != 1:
        raise RuntimeError('Exact historical candidate comparison')
    edits = [
        ("HERE = Path(__file__).resolve().parent", "HERE = Path(__file__).resolve().parent\nSHARED = _GS_SHARED\nHISTORICAL_CANDIDATE = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'",1),
        ("CANDIDATE = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'", "CANDIDATE = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'",1),
        ("SCHEMA = 'current_comp_shared_old384_score_panel_v2'", "SCHEMA = 'current_comp_shared_gain_slice_score_panel_v1'",1),
        ("        'native40','native40-watch','score40','score40-watch')", "        'native40','native40-watch','score40','score40-watch','build','pair')",1),
        ('PINS = {\n', 'PINS = {\n    **_GS_PINS,\n',1),
        (historical,historical.replace("r['candidate_sha256'] == CANDIDATE","r['candidate_sha256'] == HISTORICAL_CANDIDATE"),1),
        (function('acquisition'),'def acquisition(args, admit, env, control, control_seals):\n    return _gain_acquisition(args,admit,env,control,control_seals)\n',1),
        (function('watch_setup'),'def watch_setup(args):\n    return _gain_watch_setup(args)\n',1),
        ('def watch_contract(args, result, folder, check_sha, tick):','def _gain_original_watch_contract(args, result, folder, check_sha, tick):',1),
        ("    require(all(outputs['APUSH'][i][:5].tobytes() == outputs['shared'][i][:5].tobytes() for i in range(15)),\n        'Actual unaffected first5 layers exactly reproduce APUSH for all15 calls')",
         '    _gain_output_boundary(outputs,n,result,tick)',1),
        ("'Same actual full40 advancement authority in score and acquisition'", "'Same actual historical-only full40 selection seals in score and acquisition'",1),
        ("'native40_gate_passed'", "'historical_native40_selection_verified'",1),
        ('native40_gate_passed=True','historical_native40_selection_verified=True',1),
    ]
    # Only historical helper/source/result owners move to their original folder.
    historical_names = ('score_probe.py','check4_v2.py','score40.py','watch_score40.py','SCORE40_PROTOCOL.md',
        'native40.py','watch_native40.py','NATIVE40_PROTOCOL.md','NATIVE40_WATCH_SPEC.md',
        'compose.py','fnp.py','PROTOCOL.md','native40_v1','native40_watch_v1','score40_v1','score40_watch_v1')
    changed,spans=original,[]
    def splice(before,after,count):
        nonlocal changed
        if changed.count(before) != count:
            raise RuntimeError('Exact gain scorer source boundary: '+before[:100])
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    # The candidate definition must change before adding HISTORICAL_CANDIDATE.
    edits[0],edits[1]=edits[1],edits[0]
    for before,after,count in edits:
        splice(before,after,count)
    for name in historical_names:
        before="HERE/'"+name+"'";count=changed.count(before)
        if not count:
            raise RuntimeError('Original historical source/path must still exist: '+name)
        splice(before,"SHARED/'"+name+"'",count)
    for before,after,count in (
        ("HERE/'score_panel_v2.py'","HERE/'score_panel.py'",1),
        ("HERE/'watch_score_panel_v2.py'","HERE/'watch_score_panel.py'",1),
        ("HERE/'SCORE_PANEL_V2_PROTOCOL.md'","HERE/'SCORE_PANEL_PROTOCOL.md'",1),
        ("HERE/'SCORE_PANEL_V2_WATCH_SPEC.md'","HERE/'SCORE_PANEL_WATCH_SPEC.md'",1),
        ("HERE/'score_panel_v2'","HERE/'score_panel_v1'",1)):
        splice(before,after,count)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Positional full gain scorer inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Entire original positive scorer source restored exactly')
    return changed,dict(owner_sha256=_GS_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_FP64_noise_correction_aggregation_and_gates_unchanged=True,
        original_namespace_isolation_fix_retained=True,original_output_and_lifecycle_helpers_unchanged=True,
        backend_policy=dict(BACKEND_POLICY),no_ancestral_numerical_reexecution=True)


_GS_DERIVED, GAIN_SCORE_SOURCE_PROOF = derive_source()
exec(compile(_GS_DERIVED,str(_GS_OWNER)+':gain-source-and-backend-boundary','exec'),globals())
