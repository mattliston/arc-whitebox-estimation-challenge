"""Reviewed original ordinary-1B40 score for the current c2/OpenBLAS acquisition."""
import ast
import hashlib
from pathlib import Path
import runpy
from types import SimpleNamespace

_GS40_HERE=Path(__file__).resolve().parent
_GS40_SHARED=_GS40_HERE.parent/'current_comp_shared_old384_v1'
_GS40_OWNER=_GS40_SHARED/'score40.py'
_GS40_OWNER_SHA='b99c9fef43da28a5f84ba6769d4d0f7b378480862752d0784ceb28bd8378fde7'
if hashlib.sha256(_GS40_OWNER.read_bytes()).hexdigest()!=_GS40_OWNER_SHA:
    raise RuntimeError('Exact original ordinary-1B40 saved-score source')
_GS40_PARENT=runpy.run_path(str(_GS40_OWNER),run_name='_inert_gain_ordinary40_score_parent')
for _gs40_key,_gs40_value in _GS40_PARENT.items():
    if not _gs40_key.startswith('__') and not _gs40_key.startswith('_GS40'):
        globals()[_gs40_key]=_gs40_value
_GS40_PINS={
    _GS40_OWNER:_GS40_OWNER_SHA,
    _GS40_HERE/'build.py':'d417e12be774fae3cf33c2fd5eb17f098a43827efc556920100b4ee2bd72dd15',
    _GS40_HERE/'BUILD_PROTOCOL.md':'7598dd79a817cfb68a792be005fd2c7e9ff60c7b29bbb7dd1e1a9dd4753cef11',
    _GS40_HERE/'native40.py':'630e4128eeb29478b2e53f337c26242202deeeb276dcbaf1f7fcbc0d5c193173',
    _GS40_HERE/'watch_native40.py':'454067244608c0a74bf372bd1e3d17e3ceebf78a33e267f5ed8eaa3b5014196c',
    _GS40_HERE/'NATIVE40_PROTOCOL.md':'9906881601418253a6ccb769bf6660cbf4e408b1b2f5ef91d1fbed0fc1293235',
    _GS40_HERE/'NATIVE40_WATCH_SPEC.md':'8c8d50284c5e5d56a3c83ef8b3376d70a4d965d69e83e79625fe928521567fae',
    _GS40_HERE/'native40_v2.py':'93012679ba77f392353a7311b25c3e770451e8a88ef7a6067505832b969629a6',
    _GS40_HERE/'watch_native40_v2.py':'a854b402e8112e848c3157c0c9fef59739302ac4309bdee5596585d93dd4bb97',
    _GS40_HERE/'NATIVE40_V2_PROTOCOL.md':'ef94ffd39c40b0be0ae8144c6e3e741f79999500c0071ed26f95b2ebf9ea8906',
    _GS40_HERE/'NATIVE40_V2_WATCH_SPEC.md':'29e9b3840fdf7920379a70ae712d02f0ceb527e7c2ca771bfb9772abf0bfaaf3',
    _GS40_HERE/'native_panel.py':'4db46ec387d72950bf63dc0408d3fac56f37452b7b5e8aa1502a1fe3e5b79707',
    _GS40_HERE/'watch_native_panel.py':'e9b9774f6e5325b0b94725558ff2c319411575df738a1b94822fd7761a3c211b',
    _GS40_HERE/'NATIVE_PANEL_PROTOCOL.md':'070dcb92b066b4537f399874143c4a69db8065362679a18d2074b39ece8ffb8a',
    _GS40_HERE/'NATIVE_PANEL_WATCH_SPEC.md':'03a9cd2c4802ccd6d40d850528a755813dc022bedef0309329f46cae6b1fa82a',
    _GS40_HERE/'score_panel.py':'8d14aef64e65aa2980da646ee41020d888d3259f202890a03157b7902171bba5',
    _GS40_HERE/'watch_score_panel.py':'3a7fd5a7125aae140d4f4e81d3d7cf24e703620d9917154df43b9014b8cd64cc',
    _GS40_HERE/'SCORE_PANEL_PROTOCOL.md':'6f2b3e8748aed2b9cd3ea666a4b7f85b04132133681dd9eb2a77c389429b74c6',
    _GS40_HERE/'SCORE_PANEL_WATCH_SPEC.md':'b181bbc9931b8724871c4f251bd6379291ad843a876ee9d16fa70b8e178692ac',
}
_GS40_NATIVE=None
CANDIDATE='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
HISTORICAL_CANDIDATE='4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
PANEL_RANKS=(35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505)
PANEL_KEYS=('panel_native','panel_native_watch','panel_score','panel_score_watch')
BACKEND_POLICY=dict(candidate_inference_backend='actual production OpenBLAS 0.3.31.188.0, NumPy2.4.6, one BLAS thread',
    comparator_inference_backend='historical positive APUSH40/Accelerate',comparator_backend_matched=False,
    comparison_scope='Original fixed ordinary-1B40 objective; not a backend-isolated causal effect.',
    candidate_outputs_and_complete_flops_both_measured=True,old_4d98_objective_inherited=False,
    current_panel150_prediction_and_cost_replay=True,panel150_score_replay_required=False,
    ordinary_1B40_noise_corrected=False,submission_authorized=False)


def own_sources(args):
    require(all(getattr(args,key+'_sha256')==_GS40_PINS[HERE/name] for key,name in (
        ('native_source','native40_v2.py'),('native_watch_source','watch_native40_v2.py'),
        ('native_protocol','NATIVE40_V2_PROTOCOL.md'),('native_watch_spec','NATIVE40_V2_WATCH_SPEC.md'))),
        'Explicit hashes must identify the reviewed current full40 owners')
    return {**PINS,HERE/'score40.py':args.source_sha256,HERE/'watch_score40.py':args.watcher_sha256,
        HERE/'SCORE40_PROTOCOL.md':args.protocol_sha256,HERE/'SCORE40_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'native40_v2.py':args.native_source_sha256,HERE/'watch_native40_v2.py':args.native_watch_source_sha256,
        HERE/'NATIVE40_V2_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE40_V2_WATCH_SPEC.md':args.native_watch_spec_sha256}


def _gain40_prerequisite(admit,args):
    records,seals={},{}
    for key,folder,cap,rss,watched in (
        ('panel_native','native_panel_v1',1200,8_000_000_000,False),
        ('panel_native_watch','native_panel_watch_v1',1200,8_000_000_000,True),
        ('panel_score','score_panel_v1',120,2_000_000_000,False),
        ('panel_score_watch','score_panel_watch_v1',120,2_000_000_000,True)):
        r,s=admit.chain(HERE/folder,getattr(args,key+'_sha256'),cap,rss,watch=watched)
        admit.inventory(r);records[key],seals[key]=r,s
    n,w,s,sw=(records[key] for key in PANEL_KEYS)
    linked(w,seals['panel_native']);linked(sw,seals['panel_score'])
    require(n['schema']=='current_comp_shared_gain_slice_native_panel_v1'
        and w['schema']=='current_comp_shared_gain_slice_native_panel_watch_v1'
        and n['candidate_sha256']==w['candidate_sha256']==CANDIDATE
        and n['ranks']==w['ranks']==list(PANEL_RANKS)+[35]
        and n['full_model_calls']==n['retired_call_count']==w['retired_calls']==15
        and all(n.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'own_complete_bills','operation_ledgers_verified','gain_slice_pair_qualified',
            'gain_pair35_cold_exact','gain_pair35_repeat_physics_exact','same_production_backend_qualified')),
        'Direct actual current panel acquisition and same-source pair boundary')
    for name in ('native_panel.py','watch_native_panel.py','NATIVE_PANEL_PROTOCOL.md','NATIVE_PANEL_WATCH_SPEC.md'):
        path=HERE/name
        require(n['source_bindings'].get(str(path))==w['source_bindings'].get(str(path))==_GS40_PINS[path],
            'Exact current panel acquisition sources')
    for name in ('score_panel.py','watch_score_panel.py','SCORE_PANEL_PROTOCOL.md','SCORE_PANEL_WATCH_SPEC.md'):
        path=HERE/name
        require(s['source_bindings'].get(str(path))==sw['source_bindings'].get(str(path))==_GS40_PINS[path],
            'Exact positive current panel objective sources')
    require(s['schema']=='current_comp_shared_gain_slice_score_panel_v1'
        and sw['schema']=='current_comp_shared_gain_slice_score_panel_watch_v1'
        and s['candidate_sha256']==CANDIDATE and s['ranks']==list(PANEL_RANKS)
        and s['full_model_calls']==s['parameter_updates']==0 and s['repeat_excluded']
        and all(s.get(k) is True for k in ('integrity_passed','information_promising','scientific_gate_passed',
            'bank_scientific_gate_passed','both_panels_three_percent_gain','complete_panel_acquisition_reverified',
            'all_predictions_verified_before_truth','exact_APUSH_score_replay','exact_APUSH_three_summary_replay',
            'own_complete_bills','no_model_imported','all_current_native_outputs_and_F_production_backend'))
        and sw['information_promising'] is True
        and all(s['adjusted_candidate_over_parent'][g]<=.97 for g in ('public8','virgin6'))
        and s['summaries']['shared']['combined14']['mean_adjusted_noise_corrected']<4e-9
        and s['lineage']['native']==seals['panel_native'] and s['lineage']['native_watch']==seals['panel_native_watch']
        and s['lineage']['development']==n['native40_prerequisite_seals']
        and s['current_candidate_native40_qualified'] is False
        and s['current_candidate_production_five_qualified'] is False
        and s['APUSH_comparator_backend_matched'] is False and s['old_4d98_objective_inherited'] is False,
        'Actual prospective current panel gate; historical40 selection is not a current full40 score')
    return records,seals


def _gain40_native_api(args,admit):
    global _GS40_NATIVE
    path=HERE/'native40_v2.py';admit.pin(path,args.native_source_sha256)
    if _GS40_NATIVE is None:
        _GS40_NATIVE=SimpleNamespace(**runpy.run_path(str(path),run_name='_inert_gain_saved_full40_verifier'))
    require(not {'numpy','torch','flopscope','threadpoolctl','scipy'}&set(sys.modules),
        'Saved full40 metadata verifier precedes all numerical imports')
    return _GS40_NATIVE


def _gain40_header(args,result,admit,r,w):
    required={HERE/'native40_v2.py':args.native_source_sha256,HERE/'watch_native40_v2.py':args.native_watch_source_sha256,
        HERE/'NATIVE40_V2_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE40_V2_WATCH_SPEC.md':args.native_watch_spec_sha256,
        HERE/'build.py':_GS40_PINS[HERE/'build.py'],HERE/'BUILD_PROTOCOL.md':_GS40_PINS[HERE/'BUILD_PROTOCOL.md']}
    require(all(r['source_bindings'].get(str(p))==w['source_bindings'].get(str(p))==h for p,h in required.items()),
        'Exact reviewed current full40 acquisition and actual builder owners')
    api=_gain40_native_api(args,admit)
    require(r['native40_source_inverse_proof']==api.NATIVE40_SOURCE_INVERSE_PROOF
        and r['native40_version_source_proof']==api.NATIVE40_VERSION_SOURCE_PROOF
        and w['actual_worker_cli_namespace_qualified'] is True and w['admitted_worker_command']==w['command']
        and r['previous_native40_watch_still_failed'] is True
        and r['previous_native40_watch_seals']==dict(zip(
            ('result_sha256','completion_sha256','finalization_sha256'),api._N40V2_NEGATIVE))
        and r['candidate_sha256']==w['candidate_sha256']==CANDIDATE
        and all(r[key]==w[key] for key in api.PANEL_LINEAGE_KEYS)
        and r['current_panel_prerequisite_seals']==result['development_prerequisite']
        and all(r.get(k) is True for k in ('current_panel_objective_qualified','historical_native40_selection_only',
            'current_candidate_native40_acquisition_qualified','current_panel150_physics_exact',
            'own_repeat37_physics_exact','operation_ledgers_verified','operation_counts_reported_not_rebased',
            'same_production_backend_qualified','backend_actual_origins_qualified'))
        and all(r.get(k) is False for k in ('current_candidate_native40_qualified',
            'current_candidate_production_five_qualified','cross_backend_APUSH_prefix_equality_required'))
        and r['unaffected_prefix_checks']==[], 'Current41 owns every output/bill; no old CHECK4 or backend equality claim')
    pair,ps=admit.chain(HERE/'pair_watch_v2',args.pair_sha256,240,8_000_000_000,watch=True)
    build,bs=admit.chain(HERE/'build_v1',args.build_sha256,60,2_000_000_000)
    admit.inventory(pair);admit.inventory(build)
    lineage=r['pair_prerequisite_seals']
    require(lineage['pair_watch']==ps and lineage['build']==bs==pair['build_seals']
        and lineage['native']==pair['native_seals']
        and pair['schema']=='current_comp_shared_gain_slice_pair_watch_v2'
        and pair['candidate_sha256']==build['candidate_sha256']==CANDIDATE
        and pair['current_pair_qualified'] and pair['pair_vector_gate_passed'] and pair['pair_timing_gate_passed']
        and pair['fresh_native_calls']==0 and pair['full_model_calls']==2 and pair['joined_native_remote_calls']==4
        and pair['previous_pair_watch_still_failed'] and pair['prior_native_phase_resource_qualified'],
        'Same actual positive joined pair and sourcebuild through panel and current41')
    prior=read(HERE/'score_panel_v1/result.json')
    require(prior['lineage']['gain_pair_watch']==ps and prior['lineage']['gain_build']==bs
        and prior['lineage']['gain_pair_native']==lineage['native']
        and prior['lineage']['prior_negative_pair']==lineage['prior_negative_pair'],
        'Exact explicit current panel/pair/build cross-seal joins')
    proof=r['composition_proof'];paid=r['paid_loop_proof'];observer=r['passive_observer_source_proof']
    require(proof==r['build_proof']==w['composition_proof']==build['proof']
        and proof['candidate_sha256']==CANDIDATE and proof['parent_sha256']==HISTORICAL_CANDIDATE
        and all(proof.get(k) is True for k in ('one_span_gain_slice','whole_source_inverse_verified',
            'all_learned_literals_byte_exact','no_new_parameter_or_head','shared_basis_and_physics_source_unchanged'))
        and proof['candidate_imported'] is False
        and proof['before']=='fnp.tanh(P["gain"][l])' and proof['after']=='fnp.tanh(P["gain"][l:l+1])'
        and paid['whole_loop_inverse'] and paid['no_billing_input_reset_materialization_or_repeat_change']
        and paid['owner_sha256']=='8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
        and all(observer.get(k) is True for k in ('broad_push_observer_inverse_verified','unchanged_all16_all47_capture',
            'exact_shape_specific_D6_and_seven_D5_tariffs','no_added_participant_array_operations')),
        'Only gain-slice source correction; complete original physical and paid-loop ownership')
    result.update(candidate_sha256=CANDIDATE,backend_policy=dict(BACKEND_POLICY),
        current_panel_objective_qualified=True,current_gain_pair_qualified=True,
        all_current_native_outputs_and_F_production_backend=True,APUSH_comparator_backend_matched=False,
        old_4d98_objective_inherited=False,current_candidate_native40_qualified=False,
        current_candidate_production_five_qualified=False)


def _gain40_finish(args,result,admit,r,w,seals):
    checked=_gain40_native_api(args,admit).verify_native(r,tick=admit.tick)
    require(all(r[k]==w[k]==v for k,v in checked.items()) and w['backend_attestation']==r['backend_attestation'],
        'Exact current saved41 verifier,82 ledgers, backend/150/37 replay and paired watcher')
    result.update(complete_native40_acquisition_reverified=True,current_panel150_physics_exact=True,
        own_repeat37_physics_exact=True,operation_ledgers_verified=True,
        operation_counts_reported_not_rebased=True,current_candidate_native40_acquisition_qualified=True,
        previous_native40_watch_still_failed=True)
    seals.update(panel=result['development_prerequisite'],gain_pair_watch=r['pair_prerequisite_seals']['pair_watch'],
        gain_build=r['pair_prerequisite_seals']['build'],gain_pair_native=r['pair_prerequisite_seals']['native'],
        prior_negative_pair=r['pair_prerequisite_seals']['prior_negative_pair'],
        previous_native40_watch=r['previous_native40_watch_seals'])
    return r,seals


def _gain40_output_boundary(arrays,native,prior,result,tick):
    panel=prior['panel_native'];a=native['trials'][22];b=panel['trials'][2]
    require(a['rank']==b['rank']==150 and a['flops']==b['flops']
        and hashlib.sha256(arrays['shared'][22].tobytes()).hexdigest()==b['output_bytes_sha256']
        and native['current_panel150_physics_exact'] and native['cross_backend_APUSH_prefix_equality_required'] is False,
        'Loaded current-source150 full predictions and ownF replay; not cross-backend prefix equality')
    result.update(current_panel150_prediction_and_cost_replay=True,panel150_score_replay_required=False,
        ordinary_1B40_noise_corrected=False)
    tick()


def _gain40_score_complete(result,summary,old,promising):
    require({k:summary['APUSH'][k] for k in ('raw_mse','flops','adjusted')}==old['summary']['terminal'],
        'Exact original APUSH40 whole summary replay, not only individual rows')
    result.update(current_candidate_native40_qualified=bool(promising),fresh_current_native40_score_measured=True,
        exact_APUSH_all40_summary_replay=True,current_candidate_production_five_qualified=False,
        production_qualified=False,submission_authorized=False,uploaded=False,automatic_next_stage=False)


def watch_setup(args):
    values={k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256')
        for k in KEYS if k not in ('source','watcher')}
    values.update(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256)
    ns=SimpleNamespace(**values)
    inputs={HERE/'native40_v2/result.json':args.native_sha256,
        HERE/'native40_watch_v2/result.json':args.native_watch_sha256,
        HERE/'native_panel_v1/result.json':args.panel_native_sha256,
        HERE/'native_panel_watch_v1/result.json':args.panel_native_watch_sha256,
        HERE/'score_panel_v1/result.json':args.panel_score_sha256,
        HERE/'score_panel_watch_v1/result.json':args.panel_score_watch_sha256,
        HERE/'build_v1/result.json':args.build_sha256,HERE/'pair_watch_v2/result.json':args.pair_sha256}
    return own_sources(ns),inputs,HERE/'score40_v1',HERE/'score40_watch_v1'


def watch_contract(args,result,folder,check_sha,tick):
    flags=('integrity_passed','hashes_unchanged','sources_unchanged','complete_probe_acquisition_reverified',
        'complete_native40_acquisition_reverified','all_82_outputs_verified','all16_domains_and_all47_selectors_reverified',
        'own_complete_bills','all_predictions_verified_before_truth','exact_APUSH_score_replay',
        'exact_APUSH_all40_summary_replay','repeated_outputs_byte_exact','no_model_imported','metadata_fresh_union_rehashed',
        'current_panel_objective_qualified','current_gain_pair_qualified','current_panel150_physics_exact',
        'current_panel150_prediction_and_cost_replay','own_repeat37_physics_exact','operation_ledgers_verified',
        'all_current_native_outputs_and_F_production_backend','fresh_current_native40_score_measured',
        'previous_native40_watch_still_failed')
    require(all(result.get(k) is True for k in flags) and result['candidate_sha256']==CANDIDATE
        and result['ranks']==list(RANKS) and len(result['records'])==40 and result['repeat_excluded']
        and result['full_model_calls']==result['parameter_updates']==0
        and result['estimator_executed'] is False and result['training_or_refit_performed'] is False
        and result['backend_policy']==BACKEND_POLICY and result['score40_source_proof']==GAIN40_SCORE_SOURCE_PROOF
        and all(result[k] is False for k in ('ordinary_1B40_noise_corrected','panel150_score_replay_required',
            'APUSH_comparator_backend_matched','old_4d98_objective_inherited','current_candidate_production_five_qualified',
            'production_qualified','submission_authorized','uploaded','automatic_next_stage'))
        and type(result['information_promising']) is bool
        and result['information_promising']==result['scientific_gate_passed']==result['current_candidate_native40_qualified']
        ==(result['comparison']['adjusted_ratio']<=.97),
        'Current ordinary40 measurement and exact prospective gate; scientific failure remains operationally complete')
    require(result['truth_bindings']=={str(p):h for p,h in TRUTHS.items()} and result['truth_accessed'] is True,
        'Only the fixed original40 ordinary-1B truth allowlist')
    lineage=result['lineage']
    require(lineage['native']['result_sha256']==args.native_sha256
        and lineage['native_watch']['result_sha256']==args.native_watch_sha256
        and lineage['gain_build']['result_sha256']==args.build_sha256
        and lineage['gain_pair_watch']['result_sha256']==args.pair_sha256
        and all(lineage['panel'][key]['result_sha256']==getattr(args,key+'_sha256') for key in PANEL_KEYS)
        and lineage['panel']==result['development_prerequisite'],'Exact explicit current full40/panel/pair/build lineages')
    path=folder/'predictions_and_bills_sealed.json';seal=read(path)
    require(check_sha(path,tick)==result['predictions_and_bills_seal_sha256']==result['artifacts'][str(path)]
        and seal['truth_accessed'] is False and seal['all_82_outputs_verified'] and seal['own_complete_bills']
        and seal['lineage']==lineage and seal['source_bindings']==result['source_bindings']
        and seal['input_bindings']==result['input_bindings'],'Actual complete current41/APUSH41 pre-truth seal')


def derive_source():
    original=_GS40_PARENT['_S40_DERIVED']
    def function(name):
        nodes=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1:
            raise RuntimeError('Unique original full40 helper '+name)
        n=nodes[0];return ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    acquisition_source=function('acquisition')
    a=acquisition_source.index('    required = {');b=acquisition_source.index("    require(len(r['artifacts'])",a)
    changed_acquisition=acquisition_source[:a]+'    _gain40_header(args,result,admit,r,w)\n'+acquisition_source[b:]
    if changed_acquisition.count('return r,dict(probe=rs,probe_watch=ws)')!=1:
        raise RuntimeError('Unique original acquisition return boundary')
    changed_acquisition=changed_acquisition.replace("return r,dict(probe=rs,probe_watch=ws)",
        "return _gain40_finish(args,result,admit,r,w,dict(native=rs,native_watch=ws))")
    own_a=original.index('    own = {',original.index('def score('))
    own_b=original.index('    admit.bind(',own_a)
    edits=[
        ("SCHEMA = 'current_comp_shared_old384_score40_v1'","SCHEMA = 'current_comp_shared_gain_slice_score40_v1'"),
        ("KEYS = ('source','protocol','watcher','probe-source','probe-watch-source',\n"
         "        'probe-protocol','probe-watch-spec','probe','probe-watch','check4-score','check4-watch')",
         "KEYS = ('source','protocol','watcher','watch-spec','native-source','native-watch-source',\n"
         "        'native-protocol','native-watch-spec','native','native-watch',\n"
         "        'panel-native','panel-native-watch','panel-score','panel-score-watch','build','pair')"),
        ('PINS = {\n','PINS = {\n    **_GS40_PINS,\n'),
        (acquisition_source,changed_acquisition),
        (original[own_a:own_b],'    own = own_sources(args)\n'),
        ('prior,prerequisite = _score40_prerequisite(admit,args)','prior,prerequisite = _gain40_prerequisite(admit,args)'),
        ("    result['score_adapter_proof'] = _CHECK4_PROOF\n"
         "    result['score_metadata_revision_proof'] = _V2_PROOF\n"
         "    result['score40_source_proof'] = _S40_PROOF\n",
         "    result['score40_source_proof'] = GAIN40_SCORE_SOURCE_PROOF\n"),
        ("    require(all(arrays['shared'][i][:5].tobytes() == arrays['APUSH'][i][:5].tobytes() for i in range(40)),\n"
         "        'Unaffected five-layer parent bytes, both repeats already byte-identical')",
         '    _gain40_output_boundary(arrays,r,prior,result,tick)'),
        ("    require(records[:4] == prior['records'], 'Exact previously measured first4 raw/cost/adjusted rows')",
         '    # The panel uses different four-million truths; no panel/full40 raw-score equality is asserted.'),
        ("    promising = comparison['adjusted_ratio'] <= .97\n",
         "    promising = comparison['adjusted_ratio'] <= .97\n    _gain40_score_complete(result,summary,old,promising)\n"),
    ]
    changed,spans=original,[]
    def splice(before,after,count=1):
        nonlocal changed
        if changed.count(before)!=count:
            raise RuntimeError('Exact gain full40 score boundary: '+repr((before[:100],changed.count(before),count)))
        cursor=0
        for unused in range(count):
            pos=changed.index(before,cursor);spans.append((pos,before,after))
            changed=changed[:pos]+after+changed[pos+len(before):];cursor=pos+len(after)
    for before,after in edits:
        splice(before,after)
    for before,after,count in (
        ("'current_comp_shared_old384_native40_v1'","'current_comp_shared_gain_slice_native40_v2'",1),
        ("'current_comp_shared_old384_native40_watch_v1'","'current_comp_shared_gain_slice_native40_watch_v2'",1),
        ("HERE/'native40_v1'","HERE/'native40_v2'",3),
        ("HERE/'native40_watch_v1'","HERE/'native40_watch_v2'",1),
        ('args.probe_sha256','args.native_sha256',1),('args.probe_watch_sha256','args.native_watch_sha256',1)):
        splice(before,after,count)
    for name in ('compose.py','fnp.py','PROTOCOL.md'):
        before="HERE/'"+name+"'";count=changed.count(before)
        if count!=1:
            raise RuntimeError('Only the original immutable physical-source pin relocates: '+name)
        splice(before,"_GS40_SHARED/'"+name+"'",count)
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional full ordinary40 source inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire original ordinary40 score source restored exactly')
    return changed,dict(owner_sha256=_GS40_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        original_per_network_FP64_arithmetic_and_mean_of_products_unchanged=True,
        original40_targets_and_stored_FP32_precision_unchanged=True,ordinary_1B40_noise_corrected=False,
        current_source_acquisition_and_panel_selection_required=True,current_panel150_prediction_and_cost_replay=True,
        panel150_score_replay_required=False,actual_native_v2_parser_namespace_required=True,
        old_zero_call_native40_watch_still_failed=True,prospective_mean_adjusted_ratio=.97,backend_policy=dict(BACKEND_POLICY))


_GS40_DERIVED,GAIN40_SCORE_SOURCE_PROOF=derive_source()
exec(compile(_GS40_DERIVED,str(_GS40_OWNER)+':current-openblas-ordinary40-boundary','exec'),globals())
