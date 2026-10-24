"""DRAFT: exact BOX watcher inverse; ONE unchanged cold37 closed-summary profile."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OWNER = NEW/'current_comp_push_frontier_boxes_v1/watch_rpc37.py'
OWNER_SHA256 = 'b00d1ab822d6e9ba9071741d5bbc4adc02c76bb175a34d323a02490abc928645'
BOX = OWNER.parent/'rpc37_v1'
CANDIDATE = BOX/'estimator.py'
CANDIDATE_SHA256 = '378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
BOX_SEALS = (
    '9db5c5f9fd0afbe1b8bd00fc4b8bb5e146501996d8bd008b129be95944d9bcfa',
    '880f49684a5e71031387c596c5a847e6b322e4d486c7cb39fbe1e94684a2c88c',
    '2827bff6c5ab20dce11501172b86b5bc4d7dae18ccbc835064fdf98718d9111d')
TAG = 'current_comp_box_rpc_profile37_v1'
EXPECTED_FLOPS = 407741701711
CLIENT_SITE = P2/'data_gate/.client-venv/lib/python3.11/site-packages'
SERVER_SITE = NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages'
PINS = {
    OWNER: OWNER_SHA256,
    OWNER.parent/'RPC37_PROTOCOL.md': '2eb20392b14e516fbd49a6c25a75e3038add17ec33758fcaa4bfa653e030b6c5',
    CLIENT_SITE/'flopscope/_budget.py': '8468c0cf31344ab28e470e3b40422e5c4eb0a361af24f51e918a6f6e1ebadc47',
    CLIENT_SITE/'flopscope/_connection.py': '6e1e822bdf1fec065d045dfb9da3e67b73e592b993d822aeacf250a218b78e82',
    CLIENT_SITE/'flopscope/_dispatch.py': 'dd83969c8f842cfcde43dea0bc61aea57c96cc5cd5f8aa664011567d7cb108ec',
    SERVER_SITE/'flopscope/_budget.py': 'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    SERVER_SITE/'flopscope_server/_request_handler.py': 'c6157a3512193573c425c93514c44cdb9438f91b7d52ced86e2ed2cff8348fde',
    SERVER_SITE/'flopscope_server/_session.py': '76c22d0b368b03c98a33e97c5e33af42acaf509e7ec100b3dbe692e5775d9ee1',
    SERVER_SITE/'flopscope_server/_server.py': 'e95879772cb7860c4762f8b6e3890db176f2946fd8b6eadbd67e6f2aac3df8ed',
    SERVER_SITE/'flopscope_server/_comms_tracker.py': '4fc1c2d34c5b31b14e850fe02bd98bc8208984926964c4db3d44a629f813e2e7',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_and_proof():
    """Inert complete watcher inverse; no candidate execution or materialization."""
    original = OWNER.read_bytes()
    require(hashlib.sha256(original).hexdigest() == OWNER_SHA256, 'Frozen full-tree BOX watcher')
    source = original.decode('utf-8')
    changed, edits = source, []

    def replace(before, after):
        nonlocal changed
        require(before and after and changed.count(before) == 1 and after not in changed,
            'Unique nonempty full-source watcher splice: '+before[:90])
        changed = changed.replace(before,after)
        edits.append((before,after))

    def replace_span(first, last, after):
        require(changed.count(first) == changed.count(last) == 1, 'Unique bounded watcher source span')
        start = changed.index(first)
        finish = changed.index(last,start)+len(last)
        replace(changed[start:finish],after)

    replace('CLIENT = NEW/\'current_comp_padded_push_d6_v1/client.py\'\n',
        "CLIENT = HERE/'client.py'\n")
    replace("    CLIENT: '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',\n",
        "    NEW/'current_comp_padded_push_d6_v1/client.py': '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',\n")
    replace("STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'\n",
        "PINS.update(_PROFILE_PINS)\nSTORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'\n")
    replace('CAP, RSS, PROGRESS = 300., 8_000_000_000, 120.\n',
        'CAP, RSS, PROGRESS = 180., 8_000_000_000, 120.\n')
    replace("OUT = HERE/'rpc37_v1'\nTAGS = ('unused_frozen_negative_baseline','current_comp_push_frontier_boxes37_v1')\n",
        "OUT = HERE/'profile_v1'\nTAGS = ('archived_positive_BOX_cold37','current_comp_box_rpc_profile37_v1')\n")
    replace("    for key in ('source','composer','protocol'):\n",
        "    for key in ('source','client','protocol','spec'):\n")
    replace('    began = previous = last_progress = time.monotonic()\n',
        '    began = previous = last_progress = _PROFILE_BEGAN\n')
    replace("    own = {**PINS,HERE/'watch_rpc37.py':args.source_sha256,HERE/'compose.py':args.composer_sha256,\n        HERE/'RPC37_PROTOCOL.md':args.protocol_sha256}\n",
        "    own = {**PINS,HERE/'watch.py':args.source_sha256,HERE/'client.py':args.client_sha256,\n        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256}\n")
    replace("    paths = [NEW/f'{TAGS[1]}{suffix}' for suffix in ('.json','_0_r37.f32','_1_r37.f32')]\n",
        "    paths = [NEW/f'{TAGS[1]}{suffix}' for suffix in ('.json','_0_r37.f32')]\n")
    replace("    result = dict(schema='current_comp_push_frontier_boxes_rpc37_v1',passed=False,completed=False,\n",
        "    result = dict(schema='current_comp_box_rpc_profile_v1',passed=False,completed=False,\n")
    replace("        require(now-began < CAP and now-last_progress < PROGRESS, 'Whole300s/actual-retired120s cap')\n",
        "        require(now-began < CAP and now-last_progress < PROGRESS, 'Whole180s/actual-retired120s cap')\n")
    replace("                    deadline(); index = len(result['records'][active]); require(index < 2,'Exactly two calls per role')\n",
        "                    deadline(); index = len(result['records'][active]); require(index < 1,'Exactly one fresh cold37 context')\n")
    replace('        negative_control(result,tick)\n',
        '        profile_control(globals(),result,tick)\n')
    replace("        client_api = load(CLIENT,PINS[CLIENT],'push_runtime_fixed_client')\n",
        "        client_api = load(CLIENT,args.client_sha256,'box_closed_profile_client')\n")
    replace_span("        composer = load(HERE/'compose.py',args.composer_sha256,'push_runtime_composer')\n",
        '        del text,client_source\n',
        "        candidate = Path(result['candidate_path'])\n"
        "        require(sha(candidate,tick) == result['candidate_sha256'],'Immutable archived BOX candidate')\n"
        "        del client_source\n")
    replace("            scratch = Path(tempfile.mkdtemp(prefix='astra_frontier_boxes_rpc37_',dir='/private/tmp'))\n",
        "            scratch = Path(tempfile.mkdtemp(prefix='astra_box_profile_rpc37_',dir='/private/tmp'))\n")
    replace("            server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','300'],\n",
        "            server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','180'],\n")
    replace("                and len(series['records']) == 2 and series['repeated_output_byte_identical'], 'Complete persistent same-source pair')\n",
        "                and len(series['records']) == 1 and series['no_repeat_requested'] is True\n"
        "                and series['full_model_calls'] == 1, 'Complete single cold37; no repeat claim')\n")
    replace_span("        payloads = [[values(Path(result['historical_cold_output']))],\n",
        '            current37_exact_replay=True,own_complete_bills=True,all16_vector_gate_passed=True)\n',
        '        final_profile_checks(globals(),result,tick,client_api)\n')
    replace("    print(json.dumps({k:result.get(k) for k in ('passed','completed','information_advances','retired_calls',\n        'cold_wall_ratio_to_saved_negative','maximum_new_wall_s','elapsed_s','maximum_combined_rss_bytes',\n",
        "    print(json.dumps({k:result.get(k) for k in ('passed','completed','information_advances','retired_calls',\n        'cached_closed_summary_verified','maximum_new_wall_s','elapsed_s','maximum_combined_rss_bytes',\n")
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique complete watcher inverse')
        inverse = inverse.replace(after,before)
    require(inverse.encode('utf-8') == original, 'Exact whole frozen watcher inverse')
    proof = dict(schema='current_comp_box_rpc_profile_watch_source_v1',
        owner_path=str(OWNER),owner_sha256=OWNER_SHA256,source_splices=len(edits),
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True,unchanged_resource_collection_and_cleanup=True,
        unchanged_context_bill_gate=True,unchanged_final_publication_scope=True,
        fresh_calls=1,repeat_claim=False,whole_cap_s=180,retired_progress_cap_s=120)
    return changed, proof


def profile_control(api, result, tick):
    """Direct actual BOX seals/inventories, not recursive ancestral re-admission."""
    check, sha, read = (api[k] for k in ('require','sha','read'))
    paths = tuple(BOX/name for name in ('result.json','completion.json','finalization.json'))
    hashes = tuple(sha(p,tick) for p in paths)
    check(hashes == BOX_SEALS, 'Exact actual BOX result/completion/finalization')
    r,e,f = map(read,paths)
    check(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == hashes[0]
        and f['completion_sha256'] == hashes[1], 'Actual positive operational BOX three-seal chain')
    check(r['schema'] == 'current_comp_push_frontier_boxes_rpc37_v1'
        and r['candidate_sha256'] == CANDIDATE_SHA256 and r['retired_calls'] == 2
        and r['information_advances'] is False
        and all(r[k] is True for k in ('cleanup_passed','whole_owned_tree_retired',
            'observational_groups_retired','actual_completed_samples','hashes_unchanged',
            'sources_unchanged','own_complete_bills','own_repeat_byte_exact',
            'current37_exact_replay','all16_vector_gate_passed'))
        and 0 < r['maximum_interval_s'] <= 1 and r['maximum_swap_growth_bytes'] == 0
        and r['maximum_combined_rss_bytes'] < 8_000_000_000
        and 0 < r['elapsed_s'] <= f['elapsed_s'] < 300
        and 0 <= f['terminal_publication_interval_s'] < 1,
        'Actual operational/resource qualification, prior45s/.80 runtime advancement still FAILED')
    check(len(r['records']) == 2 and len(r['records'][1]) == 2, 'Actual completed BOX pair metadata')
    for index,row in enumerate(r['records'][1]):
        api['check_row'](row,index)
        check(row['output_sha256'] == api['BASE_OUTPUT']
            and row['flops'] == (EXPECTED_FLOPS,404454713141)[index], 'Archived BOX own exact F/output')
    control = NEW/'current_comp_push_frontier_boxes37_v1_0_r37.f32'
    check(r['artifacts'][str(CANDIDATE)] == CANDIDATE_SHA256
        and r['artifacts'][str(control)] == api['BASE_OUTPUT'], 'Original candidate/output inventory ownership')
    for group in ('source_bindings','input_bindings','artifacts'):
        check(isinstance(r[group],dict) and r[group], 'Nonempty exact direct historical inventory')
        target = result['source_bindings' if group == 'source_bindings' else 'input_bindings']
        for path,digest in r[group].items():
            check(path not in target or target[path] == digest, 'No contradictory source/input authority')
            check(sha(path,tick) == digest, 'Every direct BOX inventory binding unchanged')
            target[path] = digest
    for path,digest in zip(paths,hashes):
        result['input_bindings'][str(path)] = digest
    check(sha(CANDIDATE,tick) == CANDIDATE_SHA256, 'Actual immutable BOX candidate')
    result['records'][0] = [r['records'][1][0]]
    result.update(candidate_path=str(CANDIDATE),candidate_sha256=CANDIDATE_SHA256,
        historical_cold_output=str(control),historical_BOX_seals=list(hashes),
        historical_BOX_operationally_passed=True,historical_BOX_runtime_advancement_failed=True,
        historical_cold_record_only=True,fresh_baseline_calls=0,
        watch_source_proof=api['_PROFILE_PROOF'],profile_only=True,
        candidate_source_modified=False,repeat_requested=False,
        per_dispatch_or_shape_timing_measured=False)


def final_profile_checks(api, result, tick, client):
    check, sha = api['require'],api['sha']
    check(result['retired_calls'] == 1 and len(result['records'][1]) == 1,
        'Exactly one actual closed cold37 call')
    row = result['records'][1][0]
    api['check_row'](row,0)
    current_path = NEW/f'{TAG}_0_r37.f32'
    original, _ = api['values'](Path(result['historical_cold_output']))
    current, _ = api['values'](current_path)
    check(current == original and sha(current_path,tick) == api['BASE_OUTPUT']
        and row['output_sha256'] == api['BASE_OUTPUT'] and row['flops'] == EXPECTED_FLOPS,
        'All16 exact original BOX/c2 bytes and complete coldF; no diagnostic subtraction')
    profile = row['operation_profile']
    check(client.profile_summary(profile['summary'],TAG+'-0') == profile,
        'Independent exact saved cached-summary reconciliation')
    check({k:v for k,v in profile['summary'].items() if 'time' in k} == row['timing'],
        'Summary and original call timing are one actual context')
    result.update(operation_profile=profile,cached_closed_summary_verified=True,
        own_complete_bills=True,all16_vector_gate_passed=True,current37_exact_replay=True,
        full_model_calls=1,retired_call_count=1,one_cold_no_repeat=True,
        maximum_new_wall_s=row['timing']['wall_time_s'],information_advances=False,
        diagnostic_flops=0,diagnostic_array_operations=0,extra_rpc_requests=0,
        initialization_included_not_separately_measured=True,
        raw_error_measured=False,adjusted_score_measured=False,runtime_gain_claimed=False)


def main():
    began = time.monotonic()
    text, proof = source_and_proof()
    namespace = dict(__name__='_current_comp_box_profile_watch_body',__file__=str(HERE/'watch.py'),
        _PROFILE_PINS=PINS,_PROFILE_PROOF=proof,_PROFILE_BEGAN=began,
        profile_control=profile_control,final_profile_checks=final_profile_checks)
    exec(compile(text,str(OWNER),'exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
