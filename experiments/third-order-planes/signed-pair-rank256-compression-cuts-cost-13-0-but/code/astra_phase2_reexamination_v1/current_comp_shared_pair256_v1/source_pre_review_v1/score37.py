"""Inert source adapter: one fixed pair256/current-c2 cold37 saved score.

The original rank37 numerical body, finite output/lifecycle checks and .95
gate are retained by a positional whole-source inverse. No model is imported.
Only an explicit --run may create a stage or import NumPy after admission.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

_PAIR_HERE = Path(__file__).resolve().parent
_PAIR_NEW = _PAIR_HERE.parent
_PAIR_SHARED = _PAIR_NEW/'current_comp_shared_old384_v1'
_PAIR_GAIN = _PAIR_NEW/'current_comp_shared_gain_slice_v1'
_PAIR_OWNER = _PAIR_SHARED/'score_probe.py'
_PAIR_OWNER_SHA = 'a279e9bf196c50ef3157ade817946f5f6a5f75521040f0ce03820649a8c1e7cb'
_PAIR_NATIVE = None
_PAIR_CANDIDATE = 'd7e9cba805c286bc7e1f81d36d05e4bc8cfb9b992324e61037d3aa6bee5cf6ba'
_PAIR_CURRENT = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
_PAIR_SEAL_KEYS = ('result_sha256','completion_sha256','finalization_sha256')
_PAIR_SEALS = {
    'native': ('95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e',
        'a4896323273ce08d9e8c18a8ee9b2e0a8ada6439bbfbb5d334c530f276eb513c',
        'a113ac65d0360f4021ccc89d10f9bc0165e0be8086fe95afb77ff23e44e56319'),
    'native_watch': ('1bc1135f275609a4ea64bd7414de2cae9881f9396af3b561f6af6e1fe2bc7bfb',
        'ff445d8e23b4b010c03b4137685294ad0bd85b4484f9a2862096c610b229490c',
        'be9293ac03e21ac637bb1dc04af8fb4646639d5cd58c39688e027add935d9541'),
    'score': ('886cefd68a93386c8e04e0508db291bc6a8ecb972c2237217a266be8a73b99b6',
        '54e801b0c33ab19e9962fd0733e31d9196c5309e86fe0c6520d76bc8cbb041ac',
        '5b6d28475ee1531312ef750442d8a280fa555f9aa3be955ceb5dc567883d05e2'),
    'score_watch': ('5c302feae0c38412610144183faa463323ddaba090f0f3f823c31073f58321bd',
        '82820322c963f31149f3f33a67cc5db06e166631a06cea927234bf82d06f6b76',
        'ae5830fcd496498fd6266afec6498df0132ba13d4b240dc8fafeaf7da25914fa'),
}
_PAIR_PINS = {
    _PAIR_OWNER:_PAIR_OWNER_SHA,
    _PAIR_SHARED/'SCORE_PROBE_PROTOCOL.md':'8b2aa405fe83056eb018ddd4f0eaad01e84f5457d4d2144e4d6e9a672fddf570',
    _PAIR_HERE/'metric_fnp.py':'52731186b0ee2ecd932fb9b3a7a44bd23a305dd17d3afbf58f8949019df5c44e',
    _PAIR_HERE/'compose.py':'1d73f6b9c951917a5af0bbc57ec2cd020e01cff471cddbfd37720f73995b1b3e',
    _PAIR_HERE/'DESIGN.md':'5290ec27913961c438768730320f1641bf0d7fcd530b70fbce4e5dc6d30a6653',
    _PAIR_GAIN/'build_v1/estimator.py':_PAIR_CURRENT,
    _PAIR_GAIN/'native40_v2.py':'93012679ba77f392353a7311b25c3e770451e8a88ef7a6067505832b969629a6',
    _PAIR_GAIN/'watch_native40_v2.py':'a854b402e8112e848c3157c0c9fef59739302ac4309bdee5596585d93dd4bb97',
    _PAIR_GAIN/'NATIVE40_V2_PROTOCOL.md':'ef94ffd39c40b0be0ae8144c6e3e741f79999500c0071ed26f95b2ebf9ea8906',
    _PAIR_GAIN/'NATIVE40_V2_WATCH_SPEC.md':'29e9b3840fdf7920379a70ae712d02f0ceb527e7c2ca771bfb9772abf0bfaaf3',
    _PAIR_GAIN/'score40.py':'4693db42cfd0bc3780ada1cd59cfebeb3f7de513d1b4a855d0f06895c23502c9',
    _PAIR_GAIN/'watch_score40.py':'3b0043c99300595d27a118b255c1707cfe25ab084487dd9e17589626a16a3165',
    _PAIR_GAIN/'SCORE40_PROTOCOL.md':'f216866d4810020ad17c2890c1dd34aa6b439cf48821e20fef6dd5a8df6a459c',
    _PAIR_GAIN/'SCORE40_WATCH_SPEC.md':'a66923ad145e84873cfaa578c0049c224822bbc56c33eb7b2013544581360473',
}


def _own_sources(args):
    require(args.candidate_sha256 == _PAIR_CANDIDATE, 'Only the prospectively fixed pair256 source')
    return {**PINS,HERE/'score37.py':args.source_sha256,HERE/'watch_score37.py':args.watcher_sha256,
        HERE/'SCORE37_PROTOCOL.md':args.protocol_sha256,HERE/'SCORE37_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'native37.py':args.native_source_sha256,HERE/'watch_native37.py':args.native_watch_source_sha256,
        HERE/'NATIVE37_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE37_WATCH_SPEC.md':args.native_watch_spec_sha256,
        HERE/'build37.py':args.builder_sha256,HERE/'BUILD37_PROTOCOL.md':args.build_protocol_sha256}


def _native_api(args, admit):
    global _PAIR_NATIVE
    path = HERE/'native37.py'
    admit.pin(path,args.native_source_sha256)
    if _PAIR_NATIVE is None:
        spec = importlib.util.spec_from_file_location('pair256_saved_native37_verifier',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PAIR_NATIVE = module
    require(not {'numpy','scipy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'New native/build/tiny verification must remain strictly metadata-only')
    return _PAIR_NATIVE


def _acquisition_header(args, result, admit):
    r,rs = admit.chain(HERE/'native37_v1',args.native_sha256,300,8_000_000_000)
    w,ws = admit.chain(HERE/'native37_watch_v1',args.native_watch_sha256,300,8_000_000_000,watch=True)
    linked(w,rs); admit.inventory(r); admit.inventory(w)
    require(r['schema'] == 'current_comp_shared_pair256_native37_v1'
        and w['schema'] == 'current_comp_shared_pair256_native37_watch_v1'
        and r['ranks'] == w['ranks'] == [37,37]
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 2
        and r['candidate_sha256'] == w['candidate_sha256'] == args.candidate_sha256 == _PAIR_CANDIDATE
        and all(r.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills','actual_production_backend_qualified',
            'current_c2_controls_qualified','build_watch_qualified'))
        and r['truth_accessed'] is False and r['score_measured'] is False and r['parameter_updates'] == 0,
        'Fixed new two-call source/roster/backend before any arrays or truth')
    required = {HERE/'native37.py':args.native_source_sha256,HERE/'watch_native37.py':args.native_watch_source_sha256,
        HERE/'NATIVE37_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE37_WATCH_SPEC.md':args.native_watch_spec_sha256,
        HERE/'build37.py':args.builder_sha256,HERE/'BUILD37_PROTOCOL.md':args.build_protocol_sha256}
    require(all(r['source_bindings'].get(str(p)) == w['source_bindings'].get(str(p)) == h for p,h in required.items()),
        'Exact explicit new native/watch/builder source owners')
    build_path = HERE/'build37_v1/result.json'
    require(r['input_bindings'].get(str(build_path)) == w['input_bindings'].get(str(build_path)) == args.build_sha256,
        'Explicit build RESULT is the one used by both actual acquisition owners')
    native = _native_api(args,admit)
    authority = r['authority']
    expected = dict(source=args.native_source_sha256,watch=args.native_watch_source_sha256,
        protocol=args.native_protocol_sha256,watch_spec=args.native_watch_spec_sha256,
        builder=args.builder_sha256,build_protocol=args.build_protocol_sha256,
        build=args.build_sha256,candidate=args.candidate_sha256)
    parsed = native.parse_child_command(w['command'])
    require(set(authority) == set(native.KEYS) and authority == w['authority']
        and all(authority.get(k) == v for k,v in expected.items())
        and vars(parsed) == dict(run=True,**{k+'_sha256':v for k,v in authority.items()})
        and w['actual_worker_cli_namespace_qualified'] is True,
        'Actual eleven-field worker command; separate build-watch pins retained through the native API')
    checked = native.verify_native(r,tick=admit.tick)
    require(isinstance(checked,dict) and checked
        and all(r.get(k) == w.get(k) == v for k,v in checked.items()),
        'Actual new saved-native verifier and matching external qualification')
    require(not {'numpy','scipy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'No model or numerical import through complete prerequisite verification')
    result.update(candidate_sha256=_PAIR_CANDIDATE,control_candidate_sha256=_PAIR_CURRENT,
        complete_native37_acquisition_reverified=True,build_and_tiny_prerequisites_reverified=True,
        actual_production_backend_qualified=True,current_c2_controls_qualified=True,
        source_adapter_proof=PAIR_SCORE_SOURCE_PROOF,comparison_is_local_current_c2=True,
        official_submission_success_claimed=False,native_authority=authority,native_verification=checked)
    return r,w,rs,ws


def _selected_control(admit,env,base,call):
    folder = _PAIR_GAIN/'native40_v2'; t = base['trials'][call]
    paths = [Path(t['output'])]+[folder/f'call_{call:02d}_{kind}.json' for kind in
        ('bill','input_bill','input_reset','lifecycle','diagnostics','complete')]
    for path in paths:
        admit.pin(path,base['artifacts'][str(path)])
    bp,ip,rp,lp,dp,mp = paths[1:]
    require(read(bp) == t and t['rank'] == 37 and t['call'] == call
        and t['includes_initialization'] is (call == 0)
        and t['input_bill'] == str(ip) and t['input_bill_sha256'] == sha(ip,admit.tick)
        and t['input_reset_receipt'] == str(rp) and t['input_reset_sha256'] == sha(rp,admit.tick),
        'Selected current cold/repeat owns complete bill and input-reset packet')
    env['lifecycle'](t,bp,lp)
    p = read(ip)
    require(p == base['input_preparation'][call] and p['rank'] == 37 and p['call'] == call
        and p['flops'] == 16777216 and p['operation_count'] == p['layers'] == 16
        and p['weight_file_sha256'] == WEIGHT and p['charged_to_participant'] is False
        and all(p.get(k) is True for k in ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed')),
        'Exact selected canonical37 physical input, not a target-derived fixture')
    env['lifecycle']({**p,'lifecycle_receipt':str(rp)},ip,rp)
    packet = read(dp)
    require(packet['call'] == call and packet['rank'] == 37
        and base['diagnostic_records'][call] == dict(path=str(dp),sha256=sha(dp,admit.tick)), 'Selected current diagnostic owner')
    d = diagnostics(packet,call == 0)
    m = read(mp)
    require(all(m.get(k) is True for k in ('passed','completed','context_retired'))
        and m['call'] == call and m['rank'] == 37 and m['bill_sha256'] == sha(bp,admit.tick)
        and m['lifecycle_sha256'] == sha(lp,admit.tick) and m['diagnostic_sha256'] == sha(dp,admit.tick)
        and m['output_sha256'] == t['output_sha256'], 'Current selected post-retirement marker')
    return d


def _current_boundary(r,lineage,admit,env):
    records,seals = {},{}
    for key,folder,cap,rss,watched in (
        ('native','native40_v2',3000,8_000_000_000,False),
        ('native_watch','native40_watch_v2',3000,8_000_000_000,True),
        ('score','score40_v1',120,1_000_000_000,False),
        ('score_watch','score40_watch_v1',120,1_000_000_000,True)):
        record,chain = admit.chain(_PAIR_GAIN/folder,_PAIR_SEALS[key][0],cap,rss,watch=watched)
        require(tuple(chain[k] for k in _PAIR_SEAL_KEYS) == _PAIR_SEALS[key], 'All three actual fixed current control seals')
        records[key],seals[key] = record,chain
    base,bw,old,sw = (records[k] for k in ('native','native_watch','score','score_watch'))
    linked(bw,seals['native']); linked(sw,seals['score'])
    require(base['schema'] == 'current_comp_shared_gain_slice_native40_v2'
        and bw['schema'] == 'current_comp_shared_gain_slice_native40_watch_v2'
        and base['candidate_sha256'] == bw['candidate_sha256'] == _PAIR_CURRENT
        and base['full_model_calls'] == base['retired_call_count'] == bw['retired_calls'] == 41
        and base['initialization_flops'] == 1944970298
        and all(base.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'own_complete_bills','repeat_byte_exact','own_repeat37_physics_exact','operation_ledgers_verified',
            'same_production_backend_qualified','backend_actual_origins_qualified',
            'current_candidate_native40_acquisition_qualified')),
        'Direct completed actual current-c2/OpenBLAS control, no historical objective inheritance')
    require(old['schema'] == 'current_comp_shared_gain_slice_score40_v1'
        and sw['schema'] == 'current_comp_shared_gain_slice_score40_watch_v1'
        and old['candidate_sha256'] == _PAIR_CURRENT and old['ranks'][0] == 37 and len(old['records']) == 40
        and old['full_model_calls'] == old['parameter_updates'] == 0 and old['repeat_excluded']
        and all(old.get(k) is True for k in ('integrity_passed','all_predictions_verified_before_truth',
            'current_candidate_native40_qualified','own_complete_bills','fresh_current_native40_score_measured'))
        and old['ordinary_1B40_noise_corrected'] is False and old['lineage']['native'] == seals['native']
        and old['lineage']['native_watch'] == seals['native_watch']
        and old['truth_bindings'].get(str(TRUTH)) == TRUTH_SHA
        and old['records'][0]['rank'] == 37 and old['records'][0]['target_original_dtype'] == 'float32'
        and old['records'][0]['target_sha256'] == TRUTH_SHA,
        'Direct actual ordinary1B current37 reference; no other truth or ancestor admission')
    for names,left,right in (
        (('native40_v2.py','watch_native40_v2.py','NATIVE40_V2_PROTOCOL.md','NATIVE40_V2_WATCH_SPEC.md'),base,bw),
        (('score40.py','watch_score40.py','SCORE40_PROTOCOL.md','SCORE40_WATCH_SPEC.md'),old,sw)):
        require(all(left['source_bindings'].get(str(_PAIR_GAIN/name))
            == right['source_bindings'].get(str(_PAIR_GAIN/name)) == _PAIR_PINS[_PAIR_GAIN/name] for name in names),
            'Actual completed comparator source owners remain exact')
    require(r['controls'] == [base['trials'][i] for i in (0,40)]
        and r['control_seals'] == dict(native=[seals['native'][k] for k in _PAIR_SEAL_KEYS],
            watch=[seals['native_watch'][k] for k in _PAIR_SEAL_KEYS]),
        'New acquisition used precisely these two current same-input controls')
    first,last = (_selected_control(admit,env,base,call) for call in (0,40))
    require(first == last and base['trials'][0]['flops']-base['trials'][40]['flops'] == 1944970298,
        'Current own repeated domains and complete startup; no cross-arm bill equality')
    lineage.update(current_native=seals['native'],current_native_watch=seals['native_watch'],
        current_score=seals['score'],current_score_watch=seals['score_watch'])
    return old


def watch_contract(args,result,folder,check_sha,tick):
    flags = ('integrity_passed','hashes_unchanged','sources_unchanged','complete_probe_acquisition_reverified',
        'complete_native37_acquisition_reverified','all_four_outputs_verified','all16_domains_and_all47_selectors_reverified',
        'own_complete_bills','all_predictions_verified_before_truth','exact_current_c2_score_replay',
        'repeated_outputs_byte_exact','no_model_imported','metadata_fresh_union_rehashed',
        'build_and_tiny_prerequisites_reverified','actual_production_backend_qualified','current_c2_controls_qualified')
    require(all(result.get(k) is True for k in flags) and result['ranks'] == [37] and result['repeat_excluded']
        and result['candidate_sha256'] == args.candidate_sha256 == _PAIR_CANDIDATE
        and result['control_candidate_sha256'] == _PAIR_CURRENT and result['source_adapter_proof'] == PAIR_SCORE_SOURCE_PROOF
        and result['full_model_calls'] == result['parameter_updates'] == 0
        and all(result.get(k) is False for k in ('estimator_executed','training_or_refit_performed',
            'production_qualified','submission_authorized','uploaded','automatic_next_stage','official_submission_success_claimed'))
        and type(result['information_promising']) is bool
        and result['information_promising'] == result['scientific_gate_passed'] == (result['comparison']['adjusted_ratio'] <= .95),
        'Sole cold37 complete-cost gate; scientific FAIL is operationally complete')
    require(result['lineage']['native']['result_sha256'] == args.native_sha256
        and result['lineage']['native_watch']['result_sha256'] == args.native_watch_sha256
        and result['lineage']['build']['result_sha256'] == args.build_sha256
        and result['input_bindings'].get(str(HERE/'build37_v1/result.json')) == args.build_sha256,
        'Actual requested native/watch/build authority joins')
    for key in ('native','native_watch','score','score_watch'):
        require(tuple(result['lineage']['current_'+key][k] for k in _PAIR_SEAL_KEYS) == _PAIR_SEALS[key],
            'Pinned direct current comparator seals; independent of JSON object ordering')
    require(result['truth_bindings'] == {str(TRUTH):TRUTH_SHA} and result['truth_accessed'] is True,
        'Only the original fixed rank37 truth metadata')
    path = folder/'predictions_and_bills_sealed.json'; seal = read(path)
    require(check_sha(path,tick) == result['predictions_and_bills_seal_sha256'] == result['artifacts'][str(path)]
        and seal['truth_accessed'] is False and seal['all_four_outputs_verified'] and seal['own_complete_bills']
        and seal['lineage'] == result['lineage'] and seal['source_bindings'] == result['source_bindings']
        and seal['input_bindings'] == result['input_bindings'], 'Actual all-output/bill pretruth seal')


def derive_source():
    raw = _PAIR_OWNER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _PAIR_OWNER_SHA:
        raise RuntimeError('Frozen original one-rank saved-score source')
    original = raw.decode()
    def function(name):
        nodes = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise RuntimeError('Unique original score function '+name)
        n = nodes[0]
        return ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    acquisition = function('acquisition')
    tail = acquisition[acquisition.index("    require(len(r['artifacts'])"):]
    replacement = 'def acquisition(args, result, admit, env):\n    r,w,rs,ws = _acquisition_header(args,result,admit)\n'+tail
    before_control = original.index("    base,bs = admit.chain(",original.index('def score('))
    after_control = original.index("    require(not {'numpy'",before_control)
    before_own = original.index('    own = {',original.index('def score('))
    after_own = original.index('    admit.bind(',before_own)
    before_pins = original.index('PINS = {\n')
    after_pins = original.index('\n}\n',before_pins)+3
    new_pins = "PINS = {**_PAIR_PINS,\n    JOINT:'7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e',\n    RC:'3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',\n    CENSUS:'7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf',\n}\n"
    edits = [
        ("SCHEMA = 'current_comp_shared_old384_score_probe_v1'","SCHEMA = 'current_comp_shared_pair256_score37_v1'",1),
        ("APUSH = NEW/'current_comp_aligned_push_d6_v1'","APUSH = _PAIR_GAIN",1),
        ("KEYS = ('source','protocol','watcher','probe-source','probe-watch-source',\n        'probe-protocol','probe-watch-spec','probe','probe-watch')",
         "KEYS = ('source','protocol','watcher','watch-spec','native-source','native-watch-source',\n        'native-protocol','native-watch-spec','native','native-watch','builder','build-protocol','build','candidate')",1),
        (original[before_pins:after_pins],new_pins,1),
        (original[before_own:after_own],'    own = _own_sources(args)\n',1),
        (acquisition,replacement,1),
        (original[before_control:after_control],'    old = _current_boundary(r,lineage,admit,env)\n',1),
        ("    lineage.update(APUSH_native=bs,APUSH_native_watch=bws,APUSH_score=ss,APUSH_score_watch=sws)",
         '    # Direct current comparator chains already joined by _current_boundary.',1),
        ("return r,dict(probe=rs,probe_watch=ws)",
         "return r,dict(native=rs,native_watch=ws,build=r['build_seals'],build_watch=r['build_watch_seals'])",1),
        ("HERE/'probe_v1'","HERE/'native37_v1'",2),
        ("APUSH/'native40_terminal_v1'","APUSH/'native40_v2'",2),
        ("HERE/'score_probe_v1'","HERE/'score37_v1'",1),
        ("'APUSH'","'current'",6),
        ("'shared'","'pair256'",4),
        ("old['records'][0]['per_role']['terminal']","old['records'][0]['per_role']['shared']",1),
        ('exact_APUSH_score_replay=True','exact_current_c2_score_replay=True',1),
        ('Exact entire prior APUSH cold score replay','Exact entire prior current-c2 cold score replay',1),
        ("{'numpy','torch','flopscope','threadpoolctl'}","{'numpy','scipy','torch','flopscope','threadpoolctl'}",2),
        ("{'torch','flopscope','threadpoolctl'}","{'scipy','torch','flopscope','threadpoolctl'}",1),
    ]
    source,spans = original,[]
    for before,after,count in edits:
        if source.count(before) != count:
            raise RuntimeError('Exact one-rank source splice: '+repr((before[:100],source.count(before),count)))
        cursor = 0
        for unused in range(count):
            at = source.index(before,cursor); spans.append((at,before,after))
            source = source[:at]+after+source[at+len(before):]; cursor = at+len(after)
    inverse = source
    for at,before,after in reversed(spans):
        if inverse[at:at+len(after)] != after:
            raise RuntimeError('Exact positional one-rank score inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    if inverse != original:
        raise RuntimeError('Complete original saved rank37 scorer restored')
    return source,dict(owner_sha256=_PAIR_OWNER_SHA,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_splices=len(spans),
        original_JOINT_FP64_arithmetic_unchanged=True,original_trial_output_and_lifecycle_unchanged=True,
        rank37_only=True,repeat_excluded=True,own_cold_complete_bills=True,
        actual_current_c2_OpenBLAS_control=True,prospective_adjusted_ratio=.95,
        raw_only_gate=False,noise_correction=False,legacy_module_imports=False)


_PAIR_SOURCE,PAIR_SCORE_SOURCE_PROOF = derive_source()
exec(compile(_PAIR_SOURCE,str(_PAIR_OWNER)+':pair256-current-c2-boundary','exec'),globals())
