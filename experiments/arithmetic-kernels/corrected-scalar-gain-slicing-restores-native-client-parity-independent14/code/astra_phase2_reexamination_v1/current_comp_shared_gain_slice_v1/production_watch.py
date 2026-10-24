"""Current-source production-five after a positive independent-panel score."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
OWNER=HERE/'remote_watch_v2.py'
OWNER_SHA='4a55a8a97e7d96d7abdb6427e06a3e73c8091ab30a9c98a50d1aafd897c7b8b0'
CANDIDATE_SHA='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
RANKS=(35,100,500,3016500,35)
RESULT_KEYS=('native_result','native_watch_result','score_result','score_watch_result')
TRUTH_METADATA_PINS={
    HERE.parent/'current_comp_aligned_push_d6_v1/panel_v1/score/both_roles_verified_before_truth.json':
        '6e2fc87d7cd4072be53c9f283fe127660cf1e7e60bb541c5b293d022408f0a56',
    HERE.parent/'current_comp_aligned_push_d6_v1/panel_v1/score/truth_access_started.json':
        '1b5a2eec3c0ef5ab0d73df0bb131e9d712826eb8b460e532bfa9fd90ca0bbf98'}
PINS={OWNER:OWNER_SHA,
    HERE/'watch.py':'53e797e07d44a058e0b30dad77ad2dd8456c69be2680bdea37eeda18e2b1950b',
    HERE/'pair_checks.py':'341ab3e95afa891d902d1a68372103f039e8a2fb8aa5f563add3c84b7f07040b',
    HERE/'PAIR_WATCH_SPEC.md':'2a655a495aaf19dcf94361bccec4770ee3e434c6e94bbf1759ea1eb8ed8b1973',
    HERE/'REMOTE_PAIR_V2_SPEC.md':'b6e844ead8ae6cf8c701793bd98e05d7c925e6b25953a50fc500b0b0cf0d1a92',
    HERE.parent/'current_comp_partial_d6_v1/production_watch_v2.py':
        '5e375679330dfdc0623140aede5a5bb3851f71588387a8e65d54e6d889f2361f'}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def truth_metadata_allowed(path):
    """Only two sealed historical JSON event/inventory records, never truth arrays."""
    return (path in TRUTH_METADATA_PINS and not path.is_symlink()
        and path.stat().st_size<2_000_000
        and hashlib.sha256(path.read_bytes()).hexdigest()==TRUTH_METADATA_PINS[path])


def owner():
    require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest()==OWNER_SHA,
            'Frozen qualified direct-server pair supervisor')
    spec=importlib.util.spec_from_file_location('production_pair_watch_owner',OWNER)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def watch_contract(record,child_seals,limit,rss):
    require(all(record.get(k) is True for k in ('cleanup_passed','whole_owned_tree_retired',
        'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked',
        'sources_unchanged','hashes_unchanged')) and record['numerical_imports'] is False
        and record['maximum_swap_growth_bytes']==0 and 0<record['maximum_interval_s']<=1
        and 0<record['maximum_combined_rss_bytes']<rss and record['elapsed_s']<limit
        and all(record['child_'+k]==v for k,v in child_seals.items()),
        'Actual positive independent watch owns complete child/resource chain')


def admit_current(args,chain,sha,read,result,tick,native_api,builder):
    native,ns=chain(HERE/'native_panel_v1',1200.)
    watched,nws=chain(HERE/'native_panel_watch_v1',1200.)
    scored,ss=chain(HERE/'score_panel_v1',120.)
    sw,sws=chain(HERE/'score_panel_watch_v1',120.)
    build,bs=chain(HERE/'build_v1',60.)
    require([x['result_sha256'] for x in (ns,nws,ss,sws)]==[
        getattr(args,key+'_sha256') for key in RESULT_KEYS], 'Four explicit actual current result authorities')
    require(all(x['candidate_sha256']==CANDIDATE_SHA for x in (native,watched,scored,build))
        and native['schema']=='current_comp_shared_gain_slice_native_panel_v1'
        and watched['schema']=='current_comp_shared_gain_slice_native_panel_watch_v1'
        and scored['schema']=='current_comp_shared_gain_slice_score_panel_v1'
        and sw['schema']=='current_comp_shared_gain_slice_score_panel_watch_v1',
        'All actual new native/scored evidence belongs to the exact c2 candidate')
    watch_contract(watched,ns,1200.,8_000_000_000)
    watch_contract(sw,ss,120.,2_000_000_000)
    require(native_api.verify_native(native,tick), 'Fresh complete native15 metadata/domain/selector/ledger/replay checks')
    require(scored['lineage']['native']==ns and scored['lineage']['native_watch']==nws
        and all(scored.get(k) is True for k in ('integrity_passed','bank_scientific_gate_passed',
            'scientific_gate_passed','both_panels_three_percent_gain','combined_noise_corrected_adjusted_below_4e9',
            'all_predictions_verified_before_truth','own_complete_bills','complete_panel_acquisition_reverified'))
        and sw['information_promising'] is True
        and scored['current_candidate_native40_qualified'] is False
        and scored['current_candidate_production_five_qualified'] is False,
        'Actual prospective native14 objective passed, without borrowed current40 or production authority')
    qspec=importlib.util.spec_from_file_location('production_score_contract',HERE/'score_panel.py')
    q=importlib.util.module_from_spec(qspec); qspec.loader.exec_module(q)
    q.watch_contract(types.SimpleNamespace(**scored['requested_seals']),scored,HERE/'score_panel_v1',sha,tick)
    expected_sources={HERE/'native_panel.py':args.native_sha256,
        HERE/'watch_native_panel.py':args.native_watch_sha256,
        HERE/'NATIVE_PANEL_PROTOCOL.md':args.native_protocol_sha256,
        HERE/'NATIVE_PANEL_WATCH_SPEC.md':args.native_spec_sha256,
        HERE/'score_panel.py':args.score_sha256,HERE/'watch_score_panel.py':args.score_watch_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.score_protocol_sha256,
        HERE/'SCORE_PANEL_WATCH_SPEC.md':args.score_spec_sha256}
    require(all(scored['source_bindings'].get(str(p))==d for p,d in expected_sources.items()),
            'Actual scorer and acquisition source owners, not merely files present now')
    builder.verify_build(bs['result_sha256'],CANDIDATE_SHA,args.builder_sha256,args.build_protocol_sha256,tick)
    require(native['pair_prerequisite_seals']['build']==bs and len(native['production_five_native_controls'])==5,
            'Native panel owns the same actual build and five canonical native controls')
    for call,(rank,index) in enumerate(zip(RANKS,(0,1,7,8,14),strict=True)):
        control=native['production_five_native_controls'][call]; trial=native['trials'][index]
        require(control['production_call']==call and control['native_call']==index and control['rank']==rank
            and control['native_output']==trial['output'] and control['native_complete_flops']==trial['flops']
            and control['production_measured'] is False,'Future production mapping was sealed before this run')
    result.update(candidate_sha256=CANDIDATE_SHA,build_seals=bs,native_seals=ns,native_watch_seals=nws,
        score_seals=ss,score_watch_seals=sws,independent14_scientific_gate_passed=True,
        prior_current_candidate_native40_qualified=False,current_candidate_native40_qualified=False,
        original_negative_fidelity_preserved=True,server_max_array_bytes=4294967296,
        native_production_mapping=list(zip(RANKS,(0,1,7,8,14))),fresh_native_calls=0,
        fresh_remote_calls_required=5,raw_error_measured=False,adjusted_score_measured=False)
    return native


def derive_source():
    parent=owner(); original,parent_proof=parent.derive_source()
    tree=ast.parse(original); lines=original.splitlines(keepends=True)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets))
    old_sources=''.join(lines[assignment.lineno-1:assignment.end_lineno])
    new_sources="""SOURCE_FILES = dict(source='production_watch.py',spec='PRODUCTION_SPEC.md',
    builder='build.py',build_protocol='BUILD_PROTOCOL.md',native='native_panel.py',
    native_protocol='NATIVE_PANEL_PROTOCOL.md',native_watch='watch_native_panel.py',
    native_spec='NATIVE_PANEL_WATCH_SPEC.md',client='production_client.py',checks='production_checks.py',
    score='score_panel.py',score_watch='watch_score_panel.py',score_protocol='SCORE_PANEL_PROTOCOL.md',
    score_spec='SCORE_PANEL_WATCH_SPEC.md')"""
    start=original.index('        _repair.admit_previous(')
    end=original.index('        scratch = Path(tempfile.mkdtemp',start)
    old_admit=original[start:end]
    new_admit="""        native = _prod.admit_current(args,chain,sha,read,result,tick,native_api,builder)
        candidate_sha = result['candidate_sha256']
        env = dict(os.environ,PYTHONUNBUFFERED='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
        reader,phase = None,'between-pairs'
        last_progress = time.monotonic()
"""
    edits=[
        (old_sources,new_sources,1),
        ("and 'truth' not in path.name.lower(),",
         "and ('truth' not in path.name.lower() or _prod.truth_metadata_allowed(path)),",1),
        ("CAP, RSS, PROGRESS = 240.","CAP, RSS, PROGRESS = 600.",1),
        ("TAG = 'current_comp_shared_gain_slice_pair_v2'","TAG = 'current_comp_shared_gain_slice_production_v1'",1),
        ("for key in SOURCE_FILES:","for key in (*SOURCE_FILES,*_prod.RESULT_KEYS):",1),
        ("    own[TRANSPORT] = TRANSPORT_SHA","    own[TRANSPORT] = TRANSPORT_SHA\n    own.update(_prod.PINS)\n    own.update(_prod.TRUTH_METADATA_PINS)",1),
        ("HERE/'pair_watch_v2', HERE/'native_v1'","HERE/'production_watch_v1', HERE/'native_panel_v1'",1),
        ("[NEW/f'{TAG}_{i}_r35.f32' for i in range(2)]","[NEW/f'{TAG}_{i}_r{r}.f32' for i,r in enumerate(_prod.RANKS)]",1),
        ("schema='current_comp_shared_gain_slice_pair_watch_v2', repair_source_proof=_repair_proof",
         "schema='current_comp_shared_gain_slice_production_watch_v1', production_source_proof=_production_proof",1),
        ("Whole240s and","Whole600s and",1),
        ("'remote':180.","'remote':360.",3),
        ("HERE/'native.py'","HERE/'native_panel.py'",1),
        ("HERE/'client_v2.py'","HERE/'production_client.py'",2),
        ("HERE/'pair_checks.py',args.checks_sha256","HERE/'production_checks.py',args.checks_sha256",1),
        (old_admit,new_admit,1),
        ("len(result['remote_records']) == 2, 'Two actually completed production contexts'",
         "len(result['remote_records']) == 5, 'Five actually completed production contexts'",1),
        ("pair_timing_gate_passed=True,own_complete_bills=True,full_model_calls=2,\n"
         "            joined_native_remote_calls=4,current_pair_qualified=True,",
         "pair_timing_gate_passed=True,own_complete_bills=True,full_model_calls=5,\n"
         "            native_fidelity_qualified=True,production_five_qualified=True,\n"
         "            full_five_network_fidelity_qualified=True,",1),
        ("actual1s/whole240s tail","actual1s/whole600s tail",1),
    ]
    source=original; history=[]
    for before,after,count in edits:
        require(source.count(before)==count,'Exact production-five supervision span '+before[:90])
        cursor=0; positions=[]
        for _ in range(count):
            at=source.index(before,cursor); positions.append(at); cursor=at+len(before)
        for at in reversed(positions):
            source=source[:at]+after+source[at+len(before):]; history.append((at,before,after))
    inverse=source
    for at,before,after in reversed(history):
        require(inverse[at:at+len(after)]==after,'Exact production supervisor inverse span')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original,'Whole qualified direct server/monitor source inverse')
    return source,dict(owner_sha256=OWNER_SHA,parent=parent_proof,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),fresh_production_calls=5,
        original_client_numerical_body=True,original_numerical_timing_and_resource_gates_preserved=True,
        no_prior_current40_or_production_authority_borrowed=True,whole_cap_s=600,remote_phase_cap_s=360)


def execution_scope():
    scope=owner().execution_scope()
    scope.update(__name__='gain_production_watch_inert',__file__=str(Path(__file__).resolve()),
        _prod=types.SimpleNamespace(**dict(globals())))
    return scope


if __name__=='__main__':
    require('--run' in sys.argv,'Explicit reviewed run only')
    source,proof=derive_source(); scope=execution_scope()
    scope.update(__name__='__main__',_production_proof=proof)
    exec(compile(source,str(OWNER)+':production-five','exec'),scope)
