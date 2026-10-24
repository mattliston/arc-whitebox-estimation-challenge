"""Fresh c2/OpenBLAS full40, conditional on the current independent14 objective."""
import ast
import hashlib
from pathlib import Path
import runpy

_G40_HERE=Path(__file__).resolve().parent
_G40_SHARED=_G40_HERE.parent/'current_comp_shared_old384_v1'
_G40_OWNER=_G40_HERE/'native_panel.py'
_G40_OWNER_SHA='4db46ec387d72950bf63dc0408d3fac56f37452b7b5e8aa1502a1fe3e5b79707'
if hashlib.sha256(_G40_OWNER.read_bytes()).hexdigest()!=_G40_OWNER_SHA:
    raise RuntimeError('Exact current OpenBLAS panel paid-loop owner')
_G40_PANEL=runpy.run_path(str(_G40_OWNER),run_name='_inert_gain_full40_panel_parent')
for _g40_key,_g40_value in _G40_PANEL.items():
    if not _g40_key.startswith('__') and not _g40_key.startswith('_G40'):
        globals()[_g40_key]=_g40_value
_G40_OLD_OWNER=_G40_SHARED/'native40.py'
_G40_OLD_SHA='0ac42cceec2bcba30813536deb33906a30cc2873ae99e5799d74c4c5e2a3472a'
if hashlib.sha256(_G40_OLD_OWNER.read_bytes()).hexdigest()!=_G40_OLD_SHA:
    raise RuntimeError('Exact original fixed40 physical-control owner')
_G40_OLD=runpy.run_path(str(_G40_OLD_OWNER),run_name='_inert_gain_full40_control_parent')
RANKS=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
    146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
SEQUENCE=RANKS+(37,)
GATE_KEYS=('panel_native','panel_native_watch','panel_score','panel_score_watch')
_G40_SOURCE_PINS={
    _G40_OWNER:_G40_OWNER_SHA,_G40_OLD_OWNER:_G40_OLD_SHA,
    HERE/'watch_native_panel.py':'e9b9774f6e5325b0b94725558ff2c319411575df738a1b94822fd7761a3c211b',
    HERE/'NATIVE_PANEL_PROTOCOL.md':'070dcb92b066b4537f399874143c4a69db8065362679a18d2074b39ece8ffb8a',
    HERE/'NATIVE_PANEL_WATCH_SPEC.md':'03a9cd2c4802ccd6d40d850528a755813dc022bedef0309329f46cae6b1fa82a',
}
PANEL_LINEAGE_KEYS=('current_panel_prerequisite_seals','control_seals','panel_rows','fixture_bindings',
    'pair_prerequisite_seals','current_panel_objective_qualified','historical_native40_selection_only',
    'current_candidate_native40_qualified','current_candidate_production_five_qualified',
    'APUSH_comparator_backend_matched','old_fidelity_still_failed')


def full40_source_owners(args):
    require(args.builder_sha256==_GP_PINS[HERE/'build.py']
        and args.build_protocol_sha256==_GP_PINS[HERE/'BUILD_PROTOCOL.md'],'Frozen actual c2 builder owners')
    owners={HERE/'native40.py':args.source_sha256,HERE/'watch_native40.py':args.watch_sha256,
        HERE/'NATIVE40_PROTOCOL.md':args.protocol_sha256,HERE/'NATIVE40_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'score_panel.py':args.panel_score_source_sha256,
        HERE/'watch_score_panel.py':args.panel_score_watch_source_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.panel_score_protocol_sha256,
        HERE/'SCORE_PANEL_WATCH_SPEC.md':args.panel_score_watch_spec_sha256,
        LEGACY:LEGACY_SHA,OBSERVER:OBSERVER_SHA,**EXTRA_SOURCES,**_G40_SOURCE_PINS}
    require(all(re.fullmatch('[0-9a-f]{64}',h or '') for h in owners.values()),'Reviewed current and future source hashes')
    return {str(p):h for p,h in owners.items()}


def panel_admission(args,inputs,tick=None):
    """Compatibility API for the original watch; admits current panel objective, not old CHECK4."""
    records,seals={},{}
    for key,folder,limit,memory,watched in (
        ('panel_native','native_panel_v1',1200,RSS,False),
        ('panel_native_watch','native_panel_watch_v1',1200,RSS,True),
        ('panel_score','score_panel_v1',120,2_000_000_000,False),
        ('panel_score_watch','score_panel_watch_v1',120,2_000_000_000,True)):
        r,s,f=_panel_chain(HERE/folder,getattr(args,key+'_sha256'),inputs,limit,tick)
        require(0<=r['elapsed_s']<=f['elapsed_s']<limit,'Actual current prerequisite whole clock')
        if not watched:
            require(0<r['peak_rss_bytes']<memory,'Actual current prerequisite worker memory')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(r[group],'Complete actual current prerequisite inventories')
            _panel_merge(inputs,r[group])
        records[key],seals[key]=r,s
    n,w,s,sw=(records[k] for k in GATE_KEYS)
    _panel_watch_gate(w,seals['panel_native'],1200,RSS)
    _panel_watch_gate(sw,seals['panel_score'],120,2_000_000_000)
    require(n['schema']=='current_comp_shared_gain_slice_native_panel_v1'
        and w['schema']=='current_comp_shared_gain_slice_native_panel_watch_v1'
        and n['candidate_sha256']==w['candidate_sha256']==COMPOSED_SHA
        and n['full_model_calls']==n['retired_call_count']==w['retired_calls']==15,
        'Actual current-source fixed15 acquisition and independent retired watcher')
    checked=_G40_PANEL['verify_native'](n,tick)
    require(all(n[k]==w[k]==v for k,v in checked.items()),'Exact current panel backend/35 replay/ledger contract')
    for path,digest in _G40_SOURCE_PINS.items():
        if path.parent==HERE:
            require(n['source_bindings'].get(str(path))==w['source_bindings'].get(str(path))==digest,
                'Actual frozen current panel source authority')
    own=full40_source_owners(args)
    for name in ('score_panel.py','watch_score_panel.py','SCORE_PANEL_PROTOCOL.md','SCORE_PANEL_WATCH_SPEC.md'):
        path=str(HERE/name)
        require(s['source_bindings'].get(path)==sw['source_bindings'].get(path)==own[path],
            'Actual explicitly reviewed current scorer source authority')
    require(s['schema']=='current_comp_shared_gain_slice_score_panel_v1'
        and sw['schema']=='current_comp_shared_gain_slice_score_panel_watch_v1'
        and s['candidate_sha256']==COMPOSED_SHA and s['ranks']==list(_G40_PANEL['RANKS'])
        and s['full_model_calls']==s['parameter_updates']==0 and s['repeat_excluded']
        and all(s.get(k) is True for k in ('integrity_passed','information_promising','scientific_gate_passed',
            'bank_scientific_gate_passed','complete_panel_acquisition_reverified','all_predictions_verified_before_truth',
            'exact_APUSH_score_replay','exact_APUSH_three_summary_replay','own_complete_bills','no_model_imported',
            'current_gain_pair_qualified','current35_same_source_full_vector_and_cost_replay',
            'all_current_native_outputs_and_F_production_backend','historical_native40_selection_only'))
        and sw['information_promising'] is True and s['both_panels_three_percent_gain']
        and all(s['adjusted_candidate_over_parent'][g]<=.97 for g in ('public8','virgin6'))
        and s['summaries']['shared']['combined14']['mean_adjusted_noise_corrected']<4e-9
        and s['current_candidate_native40_qualified'] is False
        and s['current_candidate_production_five_qualified'] is False
        and s['APUSH_comparator_backend_matched'] is False and s['old_4d98_objective_inherited'] is False
        and s['lineage']['native']==seals['panel_native']
        and s['lineage']['native_watch']==seals['panel_native_watch']
        and s['lineage']['development']==n['native40_prerequisite_seals'],
        'Actual new own-J panel gate, no old source/backend score inheritance or inferred release')
    unused,pair,pair_seals=_G40_PANEL['pair_prerequisite'](args,inputs,tick)
    require(pair_seals==n['pair_prerequisite_seals']
        and s['lineage']['gain_pair_watch']==pair_seals['pair_watch']
        and s['lineage']['gain_pair_native']==pair_seals['native']
        and s['lineage']['gain_build']==pair_seals['build']
        and s['lineage']['prior_negative_pair']==pair_seals['prior_negative_pair'],
        'Same current candidate, actual build and joined pair through panel and scorer')
    del unused,pair
    rows,fixtures=[],{}
    for rank,(row_sha,weight_sha) in _G40_OLD['_n40_store_rows'](inputs,tick).items():
        row=STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json';weight=row.parent/'weights.npy'
        # Match the original native40 metadata main's canonical FIXTURE setup
        # before invoking its unchanged control verifier (no candidate state).
        _G40_OLD['FIXTURE'][row]=row_sha
        _G40_OLD['FIXTURE'][weight]=weight_sha
        for key,path,digest in (('row_'+str(rank),row,row_sha),('weights_'+str(rank),weight,weight_sha)):
            stat=path.stat()
            fixtures[key]=dict(path=str(path),sha256=digest,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        rows.append(dict(rank=rank,weights_path=str(weight),mlp_seed=read(row)['mlp_seed']))
    controls,control_seals=_G40_OLD['controls'](inputs,tick)
    require([r['rank'] for r in rows]==list(RANKS) and len(controls)==41
        and fixtures['weights_150']==n['fixture_bindings']['weights_150']
        and fixtures['row_150']==n['fixture_bindings']['row_150']
        and rows[22]==n['panel_rows'][2]
        and fixtures['weights_150']['sha256']=='feddd579c2c51c4774c368acdf42d77c5789799900ea46d83095bea2d06b5537',
        'Exact original40 roster and same canonical150 current-source overlap')
    _panel_merge(inputs,{x['path']:x['sha256'] for x in fixtures.values()})
    bind(inputs,tick)
    return dict(current_panel_prerequisite_seals=seals,controls=controls,control_seals=control_seals,
        panel_rows=rows,fixture_bindings=fixtures,pair_prerequisite_seals=pair_seals,
        current_panel_objective_qualified=True,historical_native40_selection_only=True,
        current_candidate_native40_qualified=False,current_candidate_production_five_qualified=False,
        APUSH_comparator_backend_matched=False,old_fidelity_still_failed=True)


def full40_final_checks(result,tick=None):
    require(result['ranks']==list(SEQUENCE) and len(result['trials'])==len(result['input_preparation'])==
        len(result['diagnostic_records'])==41 and len(result['artifacts'])==287
        and _v2_validate_ledgers(result),'All41 current own calls/287 artifacts/82 closed ledgers')
    a,b=result['trials'][0],result['trials'][40]
    da,db=(read(result['diagnostic_records'][i]['path']) for i in (0,40))
    physical=lambda xs:[{k:v for k,v in item.items() if k!='operation_count'} for item in xs]
    require(a['rank']==b['rank']==37 and a['flops']-b['flops']==1944970298
        and a['output_sha256']==b['output_sha256'] and a['output_bytes_sha256']==b['output_bytes_sha256']
        and Path(a['output']).read_bytes()==Path(b['output']).read_bytes()
        and da['domains']==db['domains'] and physical(da['dispatches'])==physical(db['dispatches']),
        'Own37 complete initialization, exact repeat outputs/domains and physical dispatch')
    _G40_PANEL['_GP_PARENT']['_openblas_attestation'](result,tick)
    expected=result['current_panel_prerequisite_seals']['panel_native']
    paths=[HERE/'native_panel_v1'/name for name in ('result.json','completion.json','finalization.json')]
    require([sha(p,tick) for p in paths]==[expected[k] for k in
        ('result_sha256','completion_sha256','finalization_sha256')],'Exact qualified current panel150 reference')
    prior=read(paths[0]);t,p=result['trials'][22],prior['trials'][2]
    dt,dp=read(result['diagnostic_records'][22]['path']),read(prior['diagnostic_records'][2]['path'])
    require(t['rank']==p['rank']==150 and result['fixture_bindings']['weights_150']==prior['fixture_bindings']['weights_150']
        and all(t[k]==p[k] for k in ('flops','multiplier','includes_initialization','output_sha256','output_bytes_sha256'))
        and Path(t['output']).read_bytes()==Path(p['output']).read_bytes()
        and dt['domains']==dp['domains'] and dt['startup']==dp['startup']
        and physical(dt['dispatches'])==physical(dp['dispatches']),
        'Actual same-source OpenBLAS150 full vectors/fullF/all16/all47/physical-dispatch replay')
    require(result['cross_backend_APUSH_prefix_equality_required'] is False
        and result['unaffected_prefix_checks']==[] and result['current_panel_objective_qualified'],
        'Current panel objective owns advancement; no cross-backend or CHECK4 claim')
    return dict(operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,
        repeat_operation_counts_history_aware=True,own_repeat37_physics_exact=True,
        repeat_operation_count_history=dict(cold=a['operation_count'],warm=b['operation_count'],equality_required=False),
        current_panel150_replay=dict(call=22,panel_call=2,rank=150,output_byte_exact=True,
            complete_flops_exact=True,domains_exact=True,startup_exact=True,physical_dispatch_exact=True,
            prior_operation_count=p['operation_count'],current_operation_count=t['operation_count'],
            operation_count_equality_required=False),current_panel150_physics_exact=True,
        same_production_backend_qualified=True,backend_actual_origins_qualified=True,
        current_candidate_native40_acquisition_qualified=True,current_candidate_native40_qualified=False,
        current_candidate_production_five_qualified=False,production_five_qualified=False,
        old_fidelity_still_failed=True,fresh_native40_objective_required_before_bank=True,
        native_fidelity_qualified=False,automatic_next_stage=False,submission_authorized=False,uploaded=False)


def verify_native(record,tick=None):
    require(record['schema']=='current_comp_shared_gain_slice_native40_v1' and record['passed'] is True
        and record['completed'] is True and record['candidate_sha256']==COMPOSED_SHA
        and record['full_model_calls']==record['retired_call_count']==41
        and record['native40_source_inverse_proof']==NATIVE40_SOURCE_INVERSE_PROOF
        and all(record.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills','hashes_unchanged','sources_unchanged'))
        and record['truth_accessed'] is False and record['score_measured'] is False,
        'Actual current-source full41 acquisition, no scientific score yet')
    for group in ('source_bindings','input_bindings','artifacts'):
        bind(record[group],tick)
    for call,(trial,prep,diag) in enumerate(zip(record['trials'],record['input_preparation'],record['diagnostic_records'],strict=True)):
        folder=HERE/'native40_v1';marker=read(folder/f'call_{call:02d}_complete.json')
        require(trial['call']==prep['call']==marker['call']==call
            and trial['rank']==prep['rank']==marker['rank']==SEQUENCE[call]
            and read(folder/f'call_{call:02d}_bill.json')==trial and read(trial['input_bill'])==prep
            and prep['weight_file_sha256']==record['fixture_bindings']['weights_'+str(SEQUENCE[call])]['sha256']
            and prep['flops']==16777216 and trial['multiplier']==max(.1,trial['flops']/BUDGET)
            and marker['passed'] and marker['completed'] and marker['context_retired']
            and marker['bill_sha256']==sha(folder/f'call_{call:02d}_bill.json',tick)
            and marker['diagnostic_sha256']==diag['sha256'] and marker['output_sha256']==trial['output_sha256']
            and record['artifacts'][trial['output']]==trial['output_sha256'],'Every actual full40 paid/retired output')
    checked=full40_final_checks(record,tick)
    require(all(record[k]==v for k,v in checked.items()),'Exact saved full40 backend/replay/ledger checks')
    return checked


def derive_source():
    original=_G40_PANEL['_GP_DERIVED']
    edits=(
        ('BUDGET, CAP, RSS = 2**41, 1200., 8_000_000_000','BUDGET, CAP, RSS = 2**41, 3000., 8_000_000_000'),
        ('return gain_panel_source_owners(args)','return full40_source_owners(args)'),
        ("KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build','candidate','pair')+GATE_KEYS",
         "KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build','candidate','pair','panel_score_source','panel_score_watch_source','panel_score_protocol','panel_score_watch_spec')+GATE_KEYS"),
        ("HERE/'native_panel_v1'","HERE/'native40_v1'"),
        ("'current_comp_shared_gain_slice_native_panel_v1'","'current_comp_shared_gain_slice_native40_v1'"),
        ('panel_source_inverse_proof=PANEL_SOURCE_INVERSE_PROOF','native40_source_inverse_proof=NATIVE40_SOURCE_INVERSE_PROOF'),
        ('Whole native stage1200s/process8GB','Whole native stage3000s/process8GB'),
        ('if call == 14:  # Only the final35 is a repeated network.','if call == 40:  # Only the final37 is a repeated network.'),
        ("require(len(result['trials']) == len(result['diagnostic_records']) == 15, 'All15 complete fixed panel calls')",
         "require(len(result['trials']) == len(result['diagnostic_records']) == 41, 'All41 complete fixed full40 calls')"),
        ('result.update(panel_final_checks(result,tick))','result.update(full40_final_checks(result,tick))'),
        ("result['trials'][0]['flops']-result['trials'][14]['flops']","result['trials'][0]['flops']-result['trials'][40]['flops']"),
        ("result['full_model_calls'] = result['retired_call_count'] = 15","result['full_model_calls'] = result['retired_call_count'] = 41"),
    )
    changed,spans=original,[]
    for before,after in edits:
        if changed.count(before)!=1:
            raise RuntimeError('Exact current full40 metadata boundary: '+before[:100])
        pos=changed.index(before);spans.append((pos,before,after))
        changed=changed[:pos]+after+changed[pos+len(before):]
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)]!=after:
            raise RuntimeError('Positional current full40 inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire current OpenBLAS panel paid-loop source restored byte-exact')
    return changed,dict(owner=str(_G40_OWNER),owner_sha256=_G40_OWNER_SHA,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),source_splices=len(spans),
        candidate_paid_loop_observers_and_closed_ledger_unchanged=True,fixed_roster=list(SEQUENCE),
        whole_cap_s=3000.,actual_current_panel_objective_required=True,current_panel150_replay_required=True,
        old_CHECK4_qualification_not_transferred=True,warm_operation_count_equality_not_required=True)


_G40_DERIVED,NATIVE40_SOURCE_INVERSE_PROOF=derive_source()
exec(compile(_G40_DERIVED,str(_G40_OWNER)+':current-source-full40-metadata','exec'),globals())
