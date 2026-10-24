"""DRAFT: target-free actual current-source release evidence; no stage or estimator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
CANDIDATE='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
PINS={
    'build.py':'d417e12be774fae3cf33c2fd5eb17f098a43827efc556920100b4ee2bd72dd15',
    'BUILD_PROTOCOL.md':'7598dd79a817cfb68a792be005fd2c7e9ff60c7b29bbb7dd1e1a9dd4753cef11',
    'native_panel.py':'4db46ec387d72950bf63dc0408d3fac56f37452b7b5e8aa1502a1fe3e5b79707',
    'watch_native_panel.py':'e9b9774f6e5325b0b94725558ff2c319411575df738a1b94822fd7761a3c211b',
    'NATIVE_PANEL_PROTOCOL.md':'070dcb92b066b4537f399874143c4a69db8065362679a18d2074b39ece8ffb8a',
    'NATIVE_PANEL_WATCH_SPEC.md':'03a9cd2c4802ccd6d40d850528a755813dc022bedef0309329f46cae6b1fa82a',
    'score_panel.py':'8d14aef64e65aa2980da646ee41020d888d3259f202890a03157b7902171bba5',
    'watch_score_panel.py':'3a7fd5a7125aae140d4f4e81d3d7cf24e703620d9917154df43b9014b8cd64cc',
    'SCORE_PANEL_PROTOCOL.md':'6f2b3e8748aed2b9cd3ea666a4b7f85b04132133681dd9eb2a77c389429b74c6',
    'SCORE_PANEL_WATCH_SPEC.md':'b181bbc9931b8724871c4f251bd6379291ad843a876ee9d16fa70b8e178692ac',
    'production_watch.py':'103666cdc67071a00681f289e586f16b357f2246ee86a83e44f24bad1dc5bbd8',
    'production_client.py':'833ef452c91ed590e9ee57ed7dd6ba22971e8536ca5299930147fae7cae567ff',
    'production_checks.py':'37e29813d4bfd6dc8bb8277ab877773d18062a91c819030d0cfb4c00323a9a86',
    'PRODUCTION_SPEC.md':'d6baaa3b46cdcabb36690ad47d60849d7a474f6373a0bff2e6fefce0b663deff',
}
FUTURE_SOURCES={
    'native40_source':'native40_v2.py','native40_watch_source':'watch_native40_v2.py',
    'native40_protocol':'NATIVE40_V2_PROTOCOL.md','native40_watch_spec':'NATIVE40_V2_WATCH_SPEC.md',
    'score40_source':'score40.py','score40_watch_source':'watch_score40.py',
    'score40_protocol':'SCORE40_PROTOCOL.md','score40_watch_spec':'SCORE40_WATCH_SPEC.md',
}
BUILD_RESULT='f4a0f937612f18de26e7ee4543036240f4e0a15c6faf8d19c648ddd2c2c32a1f'
PAIR_RESULT='a8c9a0161b07e745987e980fbc7d107594099986ae09899666cece2b90dfed94'


def evidence(args,bindings,g):
    require,bind,read,chain=(g[n] for n in ('require','bind','load','chain'))
    tick=g['TICK']
    owners={HERE/name:digest for name,digest in PINS.items()}
    owners.update({HERE/name:getattr(args,key+'_sha256') for key,name in FUTURE_SOURCES.items()})
    for path,digest in owners.items():
        bind(bindings,path,digest)
    def api(name):
        path=HERE/name
        require(path in owners,'explicit_inert_evidence_source')
        spec=importlib.util.spec_from_file_location('gain_release_'+name.replace('.','_'),path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    records,seals={},{}
    for key,(folder,cap,rss) in g['STAGES'].items():
        records[key],seals[key]=chain(key,folder,getattr(args,key+'_sha256'),cap,rss,
            bindings,watched=key.endswith('_watch'))
    for child in ('native40','score40','native_panel','score_panel'):
        require(all(records[child+'_watch']['child_'+k]==v for k,v in seals[child].items()),
            'exact_child_watch_join_'+child)
    n,s,p,q,prod=(records[k] for k in ('native40','score40','native_panel','score_panel','production_watch'))
    prefix='current_comp_shared_gain_slice_'
    for key,suffix in (('native40','native40_v2'),('native40_watch','native40_watch_v2'),
        ('score40','score40_v1'),('score40_watch','score40_watch_v1'),('native_panel','native_panel_v1'),
        ('native_panel_watch','native_panel_watch_v1'),('score_panel','score_panel_v1'),
        ('score_panel_watch','score_panel_watch_v1'),('production_watch','production_watch_v1')):
        require(records[key]['schema']==prefix+suffix,'exact_current_schema_'+key)
    require(all(r.get('candidate_sha256')==CANDIDATE for r in (n,p,q,s,prod)),
        'same_actual_candidate_in_all_numeric_evidence')
    for child,names in (
        ('native40',('native40_v2.py','watch_native40_v2.py','NATIVE40_V2_PROTOCOL.md','NATIVE40_V2_WATCH_SPEC.md')),
        ('score40',('score40.py','watch_score40.py','SCORE40_PROTOCOL.md','SCORE40_WATCH_SPEC.md')),
        ('native_panel',('native_panel.py','watch_native_panel.py','NATIVE_PANEL_PROTOCOL.md','NATIVE_PANEL_WATCH_SPEC.md')),
        ('score_panel',('score_panel.py','watch_score_panel.py','SCORE_PANEL_PROTOCOL.md','SCORE_PANEL_WATCH_SPEC.md'))):
        require(all(records[child]['source_bindings'].get(str(HERE/name))==
            records[child+'_watch']['source_bindings'].get(str(HERE/name))==owners[HERE/name] for name in names),
            'actual_measured_source_owners_'+child)
    require(all(prod['source_bindings'].get(str(HERE/name))==owners[HERE/name] for name in
        ('production_watch.py','production_client.py','production_checks.py','PRODUCTION_SPEC.md')),
        'actual_five_call_production_source_owners')
    panel_api=api('native_panel.py');native_api=api('native40_v2.py')
    for record,watched,module in ((p,records['native_panel_watch'],panel_api),(n,records['native40_watch'],native_api)):
        checked=module.verify_native(record,tick)
        require(all(record[k]==watched[k]==v for k,v in checked.items()),'actual_saved_native_contract')
    panel_seals=dict(panel_native=seals['native_panel'],panel_native_watch=seals['native_panel_watch'],
        panel_score=seals['score_panel'],panel_score_watch=seals['score_panel_watch'])
    require(n['current_panel_prerequisite_seals']==panel_seals
        and q['lineage']['native']==seals['native_panel'] and q['lineage']['native_watch']==seals['native_panel_watch']
        and q['lineage']['development']==p['native40_prerequisite_seals'],
        'actual_panel_then_current_full40_not_old40_transfer')
    require(s['lineage']['native']==seals['native40'] and s['lineage']['native_watch']==seals['native40_watch']
        and s['lineage']['panel']==panel_seals
        and all(s.get(k) is True for k in ('scientific_gate_passed','information_promising','own_complete_bills',
            'complete_native40_acquisition_reverified','current_panel150_physics_exact'))
        and s['comparison']['adjusted_ratio']<=.97 and len(s['records'])==40
        and s['full_model_calls']==s['parameter_updates']==0 and s['summary']['shared']['adjusted']<4e-9,
        'current_ordinary_full40_ownJ_and_prospective_strict_below4_gate')
    require(q['bank_scientific_gate_passed'] and q['both_panels_three_percent_gain']
        and q['combined_noise_corrected_adjusted_below_4e9'] and q['own_complete_bills']
        and all(q['group_three_percent_gain_gate'][group] is True and
            q['adjusted_candidate_over_parent'][group]<=.97 for group in ('public8','virgin6'))
        and q['summaries']['shared']['combined14']['mean_adjusted_noise_corrected']<4e-9,
        'both_current_independent_groups_and_combined_strict_below4')
    api('score_panel.py').watch_contract(SimpleNamespace(**q['requested_seals']),q,
        HERE/'score_panel_v1',lambda path,unused=None:g['sha'](path),tick)
    score40_args=SimpleNamespace(native_sha256=args.native40_sha256,
        native_watch_sha256=args.native40_watch_sha256,build_sha256=BUILD_RESULT,
        pair_sha256=PAIR_RESULT,panel_native_sha256=args.native_panel_sha256,
        panel_native_watch_sha256=args.native_panel_watch_sha256,
        panel_score_sha256=args.score_panel_sha256,panel_score_watch_sha256=args.score_panel_watch_sha256)
    api('score40.py').watch_contract(score40_args,s,HERE/'score40_v1',
        lambda path,unused=None:g['sha'](path),tick)
    require(prod['native_seals']==seals['native_panel'] and prod['native_watch_seals']==seals['native_panel_watch']
        and prod['score_seals']==seals['score_panel'] and prod['score_watch_seals']==seals['score_panel_watch']
        and prod['full_model_calls']==5 and prod['fresh_native_calls']==0
        and all(prod.get(k) is True for k in ('native_fidelity_qualified','production_five_qualified',
            'full_five_network_fidelity_qualified','pair_vector_gate_passed','pair_timing_gate_passed','own_complete_bills')),
        'current_production_five_owns_the_actual_native14_scientific_source')
    checks=api('production_checks.py');tag='current_comp_shared_gain_slice_production_v1'
    paths=[HERE.parent/f'{tag}_{i}_r{rank}.f32' for i,rank in enumerate(checks.RANKS)]
    comparison=checks.compare(p,prod['remote_records'],paths,tick)
    require(comparison==prod['comparison'] and comparison['all_complete_flops_exact']
        and all(row['output_byte_exact'] and row['all16_max_abs']==0 and row['complete_flops_delta']==0
            for row in comparison['checks']) and comparison['native_repeat_byte_exact']
        and comparison['production_repeat_byte_exact'] and comparison['native_initialization_flops']==
            comparison['production_initialization_flops']==1944970298,
        'stronger_byte_exact_all16_and_completeF_fidelity_proves_same_rawJ_without_truth')
    series=read(HERE.parent/(tag+'.json'))
    require(series['source_sha256']==CANDIDATE and series['records']==prod['remote_records']
        and series['status']=='completed' and series['repeated_output_byte_identical'],
        'same_actual_saved_production_series_and_complete_call_bills')
    pair_args=SimpleNamespace(pair_sha256=PAIR_RESULT,build_sha256=BUILD_RESULT,candidate_sha256=CANDIDATE,
        builder_sha256=PINS['build.py'],build_protocol_sha256=PINS['BUILD_PROTOCOL.md'])
    unused,pair,pair_seals=panel_api.pair_prerequisite(pair_args,bindings,tick)
    require(pair_seals==p['pair_prerequisite_seals']==n['pair_prerequisite_seals']
        and q['lineage']['gain_pair_watch']==pair_seals['pair_watch']
        and q['lineage']['gain_build']==pair_seals['build'] and prod['build_seals']==pair_seals['build']
        and s['lineage']['gain_pair_watch']==pair_seals['pair_watch']
        and s['lineage']['gain_build']==pair_seals['build'] and pair['previous_pair_watch_still_failed'],
        'exact_qualified_one_span_build_pair_lineage_with_old_failure_still_false')
    del unused,pair
    seals.update(gain_pair_watch=pair_seals['pair_watch'],build=pair_seals['build'],
        gain_pair_native=pair_seals['native'],prior_negative_pair=pair_seals['prior_negative_pair'])
    records['release_fidelity_proof']=dict(all16_bytes_and_complete_F_exact=True,
        identical_raw_MSE_for_any_same_fixed_truth=True,new_truth_access=False,production_calls=5,
        inferred_MSE_ratio_not_recomputed=True)
    return records,seals
