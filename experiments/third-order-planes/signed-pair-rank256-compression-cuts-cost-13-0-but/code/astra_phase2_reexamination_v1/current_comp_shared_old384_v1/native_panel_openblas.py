"""Fixed independent14 on the qualified production BLAS; no scoring or new model."""
import hashlib
from pathlib import Path
import runpy

_OB_HERE = Path(__file__).resolve().parent
_OB_OWNER = _OB_HERE/'native_panel.py'
_OB_OWNER_SHA = '56a9260456a395a3556f7b391aca975c00b155756f52aa0843ccbe01c1da7e8c'
_OB_BACKEND_SHA = '36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8'
_OB_PINS = {
    _OB_OWNER:_OB_OWNER_SHA,
    _OB_HERE/'watch_native_panel.py':'30c733b47a685387050d7277072ebb996204e85f5e4fd8a5f31486f68ded1b24',
    _OB_HERE/'NATIVE_PANEL_PROTOCOL.md':'c66fc290cc5fa4d4ae827aa4b316a1cc19583bcd317954941d4e22dad52bf97e',
    _OB_HERE/'NATIVE_PANEL_WATCH_SPEC.md':'c29164443218894bf1a3f974332d875d8f81a9a49266c7bc2d86684fc1cffbad',
    _OB_HERE/'backend_probe.py':_OB_BACKEND_SHA,
    _OB_HERE/'watch_backend_probe.py':'149363892f2820d5eff270f4d3c918290cc00b189401296ea0faa487fa7880ac',
    _OB_HERE/'BACKEND_PROBE_PROTOCOL.md':'2d7a773171100f0234e652d68bb606d1d641fcdb2c896477515a6c6ab019a507',
    _OB_HERE/'BACKEND_PROBE_WATCH_SPEC.md':'7937eb19a6a7d74fcd68e882d6d51d506340c420adce89d92bbb0e194debafea'}
if hashlib.sha256(_OB_OWNER.read_bytes()).hexdigest() != _OB_OWNER_SHA:
    raise RuntimeError('Immutable completed Accelerate panel source')
_OB_PARENT = runpy.run_path(str(_OB_OWNER),run_name='inert_openblas_panel_parent')
for _ob_key,_ob_value in _OB_PARENT.items():
    if not _ob_key.startswith('__') and not _ob_key.startswith('_OB'):
        globals()[_ob_key] = _ob_value
_OB_BACKEND_API = None
PANEL_SCHEMA = 'current_comp_shared_old384_native_panel_openblas_v1'
PANEL_WATCH_SCHEMA = 'current_comp_shared_old384_native_panel_openblas_watch_v1'
BACKEND_GATE_KEYS = ('backend_probe','backend_probe_watch')


def backend_api():
    global _OB_BACKEND_API
    if _OB_BACKEND_API is None:
        _OB_BACKEND_API = load(HERE/'backend_probe.py','shared_openblas_panel_backend_owner',_OB_BACKEND_SHA)
    return _OB_BACKEND_API


def backend_prerequisite(args,inputs,tick=None):
    """Actual positive two-call proof, selected only through explicit future SHAs."""
    b = backend_api(); records,seals = {},{}
    for key,folder in (('probe','backend_probe_v1'),('watch','backend_probe_watch_v1')):
        expected = getattr(args,('backend_probe' if key == 'probe' else 'backend_probe_watch')+'_sha256')
        r,s,f = _panel_chain(HERE/folder,expected,inputs,300,tick)
        require(0 <= r['elapsed_s'] <= f['elapsed_s'] < 300, 'Complete actual backend prerequisite clock')
        if key == 'probe':
            require(0 < r['peak_rss_bytes'] < RSS, 'Actual backend worker peak')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[group],dict) and r[group], 'Actual direct backend prerequisite inventory')
            _panel_merge(inputs,r[group])
        records[key],seals[key] = r,s
    r,w = records['probe'],records['watch']
    _panel_watch_gate(w,seals['probe'],300,RSS)
    require(r['schema'] == 'current_comp_shared_old384_backend_probe_v1'
        and w['schema'] == 'current_comp_shared_old384_backend_probe_watch_v1'
        and r['candidate_sha256'] == w['candidate_sha256'] == COMPOSED_SHA
        and r['ranks'] == w['ranks'] == [35,35]
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 2
        and r['composition_proof'] == w['composition_proof']
        and r['backend_probe_source_proof'] == b.BACKEND_SOURCE_PROOF
        and all(r[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills')) and b.backend_contract(r,tick),
        'Actual fully paid target-free same-production-backend35 compatibility')
    require(w['backend_attestation'] == r['backend_attestation']
        and w['production_parity_checks'] == r['production_parity_checks']
        and w['control_seals'] == r['control_seals']
        and all(w[k] is True for k in ('same_production_backend_qualified',
            'target_free_backend_diagnostic_passed','all16_production_vector_parity','old_fidelity_still_failed'))
        and w['native_fidelity_qualified'] is False and w['production_five_qualified'] is False,
        'Independent watcher owns the same limited diagnostic, not a bank')
    for path,digest in _OB_PINS.items():
        if path.name in ('backend_probe.py','watch_backend_probe.py','BACKEND_PROBE_PROTOCOL.md','BACKEND_PROBE_WATCH_SPEC.md'):
            require(r['source_bindings'].get(str(path)) == w['source_bindings'].get(str(path)) == digest,
                    'Exact selected backend worker/watch owners')
    _panel_merge(inputs,{str(p):h for p,h in _OB_PINS.items()})
    bind(inputs,tick)
    return r,seals


def panel_admission(args,inputs,tick=None):
    prior,seals = backend_prerequisite(args,inputs,tick)
    panel = _OB_PARENT['panel_admission'](args,inputs,tick)
    require(prior['input_preparation'][0]['weight_file_sha256'] ==
        panel['fixture_bindings']['weights_35']['sha256'], 'Cold panel35 is the qualified physical fixture')
    panel.update(backend_prerequisite_seals=seals,backend_probe_prerequisite_qualified=True,
        historical_native40_backend='OFFPY NumPy2.4.6 Accelerate; historical development only',
        historical_native40_selection_only=True,APUSH_comparator_backend='recorded Accelerate boundary',
        APUSH_comparator_backend_matched=False,old_fidelity_still_failed=True)
    return panel


def _openblas_attestation(record,tick=None):
    b = backend_api(); a = record['backend_attestation']
    require(a['site'] == str(b._BP_SITE) and a['official_prefix'] == str(b.OFFPY.parent.parent)
        and a['actual_origins_qualified'] and a['production_blas_qualified']
        and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
        and a['whestbench_version'] == '0.16.0' and a['scipy_loaded'] is a['torch_loaded'] is False,
        'Qualified production numerical owners, not an inferred backend')
    required = {'numpy','numpy._core._multiarray_umath','numpy.linalg._linalg',
        'numpy.linalg._umath_linalg','flopscope','flopscope.numpy','threadpoolctl',
        'whestbench.domain','whestbench.scoring','whestbench.seeds'}
    require(required <= set(a['modules']), 'Complete actual core/linalg/official import evidence')
    for name,item in a['modules'].items():
        root = b._BP_OFFSITE if name.split('.')[0] == 'whestbench' else b._BP_SITE
        require(Path(item['path']).resolve().is_relative_to(root.resolve())
            and record['source_bindings'].get(item['path']) == item['sha256'], 'Actual import source pin')
    for path,digest in b._BP_PINS.items():
        require(record['source_bindings'].get(str(path)) == digest, 'Unchanged selected numerical installation')
    blas = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(blas) == 1 and blas[0]['internal_api'] == 'openblas' and blas[0]['num_threads'] == 1
        and blas[0]['version'] == '0.3.31.188.0' and blas[0]['threading_layer'] == 'pthreads'
        and Path(blas[0]['filepath']).resolve() == (b._BP_SITE/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
        'Actual single production OpenBLAS library at one thread')
    return True


def panel_final_checks(result,tick=None):
    checked = _OB_PARENT['panel_final_checks'](result,tick)
    require(result['backend_probe_prerequisite_qualified'] and result['historical_native40_selection_only']
        and result['APUSH_comparator_backend_matched'] is False
        and result['cross_backend_APUSH_prefix_equality_required'] is False
        and result['unaffected_prefix_checks'] == [] and _openblas_attestation(result,tick),
        'Explicit new backend and historical comparison scopes')
    paths = [HERE/'backend_probe_v1'/n for n in ('result.json','completion.json','finalization.json')]
    expected = result['backend_prerequisite_seals']['probe']
    require([sha(p,tick) for p in paths] == [expected[k] for k in
        ('result_sha256','completion_sha256','finalization_sha256')], 'Same qualified actual backend reference')
    prior = read(paths[0]); replay = []
    for call,old_call in ((0,0),(14,1)):
        a,b = result['trials'][call],prior['trials'][old_call]
        da,db = (read(x['diagnostic_records'][i]['path']) for x,i in ((result,call),(prior,old_call)))
        require(a['rank'] == b['rank'] == 35 and all(a[k] == b[k] for k in
            ('flops','multiplier','includes_initialization','output_sha256','output_bytes_sha256'))
            and Path(a['output']).read_bytes() == Path(b['output']).read_bytes()
            and da['domains'] == db['domains'] and da['startup'] == db['startup'],
            'Exact own-cost same-backend35 output/all16/all47/startup replay')
        physical = lambda xs:[{k:v for k,v in x.items() if k != 'operation_count'} for x in xs]
        require(physical(da['dispatches']) == physical(db['dispatches']),
                'Same-backend35 physical dispatch shapes/full F/child tariffs/arena')
        if call == 0:
            require(a['operation_count'] == b['operation_count'] and da['dispatches'] == db['dispatches'],
                    'Same-history cold35 typed operation-count and dispatch replay')
        replay.append(dict(call=call,probe_call=old_call,rank=35,output_byte_exact=True,
            complete_flops_exact=True,domains_exact=True,startup_exact=True,physical_dispatch_exact=True,
            same_history=(call == 0),operation_count_equality_required=(call == 0),
            prior_operation_count=b['operation_count'],current_operation_count=a['operation_count'],
            operation_count_delta=a['operation_count']-b['operation_count']))
        if tick is not None:
            tick()
    checked.update(backend_probe35_replay=replay,backend_probe35_cold_exact=True,
        backend_probe35_repeat_physics_exact=True,same_production_backend_qualified=True,
        backend_actual_origins_qualified=True,historical_native40_selection_only=True,
        APUSH_comparator_backend_matched=False,old_fidelity_still_failed=True,
        fresh_same_backend_objective_required_before_bank=True,native_fidelity_qualified=False,
        automatic_next_stage=False,submission_authorized=False,uploaded=False)
    return checked


EXTRA_SOURCES = {**EXTRA_SOURCES,**_OB_PINS,**backend_api()._BP_PINS}
_OB_ORIGINAL = _OB_PARENT['_NP_DERIVED']
_ob_prefix_start = _OB_ORIGINAL.index('        # The first affected pre-state is layer5.')
_ob_prefix_end = _OB_ORIGINAL.index("        result['unaffected_prefix_checks'] = prefix_checks",_ob_prefix_start)
_ob_prefix_end += len("        result['unaffected_prefix_checks'] = prefix_checks\n")
_OB_PREFIX = _OB_ORIGINAL[_ob_prefix_start:_ob_prefix_end]
_OB_EDITS = [
    ("HERE/'native_panel.py'","HERE/'native_panel_openblas.py'"),
    ("HERE/'watch_native_panel.py'","HERE/'watch_native_panel_openblas.py'"),
    ("HERE/'NATIVE_PANEL_PROTOCOL.md'","HERE/'NATIVE_PANEL_OPENBLAS_PROTOCOL.md'"),
    ("HERE/'NATIVE_PANEL_WATCH_SPEC.md'","HERE/'NATIVE_PANEL_OPENBLAS_WATCH_SPEC.md'"),
    ("HERE/'native_panel_v1'","HERE/'native_panel_openblas_v1'"),
    ("'current_comp_shared_old384_native_panel_v1'","'current_comp_shared_old384_native_panel_openblas_v1'"),
    ("KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')+GATE_KEYS",
     "KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')+GATE_KEYS+BACKEND_GATE_KEYS"),
    ("    environment(); began = time.monotonic()",
     "    began = time.monotonic(); backend_api().backend_select(); environment()"),
    ("        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()",
     "        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()\n"
     "        backend_api().backend_candidate(result,text,tick)"),
    ("threadpool_info=threadpool_info,MLP=MLP,derive_estimator_seed=derive_estimator_seed,",
     "threadpool_info=backend_api().backend_threadpool_info(threadpool_info,result,tick),MLP=MLP,derive_estimator_seed=derive_estimator_seed,"),
    (_OB_PREFIX,"        result['unaffected_prefix_checks'] = []\n"
     "        result['cross_backend_APUSH_prefix_equality_required'] = False\n"
     "        backend_api().backend_origins(result,tick)\n")]
_OB_DERIVED,_ob_spans = _OB_ORIGINAL,[]
for _ob_before,_ob_after in _OB_EDITS:
    if _OB_DERIVED.count(_ob_before) != 1:
        raise RuntimeError('Unique OpenBLAS panel metadata/backend span: '+_ob_before[:100])
    _ob_pos = _OB_DERIVED.index(_ob_before); _ob_spans.append((_ob_pos,_ob_before,_ob_after))
    _OB_DERIVED = _OB_DERIVED[:_ob_pos]+_ob_after+_OB_DERIVED[_ob_pos+len(_ob_before):]
_OB_INVERSE = _OB_DERIVED
for _ob_pos,_ob_before,_ob_after in reversed(_ob_spans):
    if _OB_INVERSE[_ob_pos:_ob_pos+len(_ob_after)] != _ob_after:
        raise RuntimeError('Positional OpenBLAS panel inverse')
    _OB_INVERSE = _OB_INVERSE[:_ob_pos]+_ob_before+_OB_INVERSE[_ob_pos+len(_ob_after):]
if _OB_INVERSE != _OB_ORIGINAL:
    raise RuntimeError('Entire frozen effective panel source restored byte-exact')
PANEL_SOURCE_INVERSE_PROOF = dict(owner=str(_OB_OWNER),owner_sha256=_OB_OWNER_SHA,
    whole_source_inverse_verified=True,original_resolved_sha256=hashlib.sha256(_OB_ORIGINAL.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_OB_DERIVED.encode()).hexdigest(),splices=len(_ob_spans),
    candidate_paid_loop_observers_and_closed_ledger_unchanged=True,fixed_roster=list(SEQUENCE),
    whole_cap_s=1200.,backend_source_sha256=_OB_BACKEND_SHA,explicit_production_backend_only=True,
    historical_Accelerate40_selection_gate_retained=True,old_cross_backend_failure_preserved=True,
    same_backend35_reference_replay_required=True,no_production_rerun=True)
exec(compile(_OB_DERIVED,str(_OB_OWNER)+':same-production-backend-panel','exec'),globals())
