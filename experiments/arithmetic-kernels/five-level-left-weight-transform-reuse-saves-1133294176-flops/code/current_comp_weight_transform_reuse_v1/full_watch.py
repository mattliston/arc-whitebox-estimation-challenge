"""DRAFT: one qualified CB2 RPC pair, with explicit W5 component gates.

Derives one effective supervisor program; no second supervisor/process layer.
No numerical stage or source preflight has been executed by this adapter's author.
"""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW/'current_comp_compact_boxes_v2/watch_fast_v1.py'
OWNER_SHA = 'de851017892634c66c2e6ba0e85527162765d876536f87d5fb8b3e89588091c3'
CLIENT_SHA = '319e354c6338b083d34d7a3b0bc7cf56723a9876449f84a205989a959ae74828'
SOURCE_PINS = {
    OWNER: OWNER_SHA,
    OWNER.parent/'FAST_COMPOSITION_SPEC.md':
        'a74c8aabadfb08d4899fd7f47dcf171db202e7f2086cfadfcb080210e9d13ba7',
    OWNER.parent/'watch_rpc37.py':
        '637ac9a1d93997adbd27f28a9916cd67a0582334322f969a619387613d27a7b9',
    OWNER.parent/'RPC37_PROTOCOL.md':
        '2e82c292d58a9b98880f62a4e02b1cb5e4da5e7a1473913d6eb953d259e71e0c',
    HERE/'full_client.py': CLIENT_SHA,
    HERE/'port.py': '741e6ed41b7cbdf338cb85690f463fa2d813056a1650fa5081ebbb2a5731567b',
    NEW/'current_comp_compact_frames_v1/client.py':
        'de87726e4b1feae01e2a36d586c6103e68910f0f7f316c586b75ccd7e4a30e0e',
    NEW/'current_comp_padded_push_d6_v1/client.py':
        '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',
}


CONTROL_SOURCE = '''

# BEGIN DIRECT SAVED CB2 COMPONENT CONTROL V1
def _w5_cb2_control(result,tick):
    folder = NEW/'current_comp_compact_boxes_v2/rpc37_v1'
    pins = (
        'd4f365526f2a5042bd9211d28afea5dec958e8214a264eec454933d7fca0ef7c',
        '03020e0abcfc921e0dd3296a89db133ab43b1b2c77a71ca56510a2a74e7e077a',
        '02e6d63b5b84e9f374d7220ba1ce45d4bb97fdb847a787dc6cf6c87c15fb90f5')
    paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
    require(tuple(sha(path,tick) for path in paths) == pins,'Actual direct CB2 three seals')
    r,c,f = map(read,paths)
    require(all(item.get('passed') is True and item.get('completed') is True for item in (r,c,f))
        and c['result_sha256'] == f['result_sha256'] == pins[0]
        and f['completion_sha256'] == pins[1]
        and r['schema'] == 'current_comp_compact_boxes_rpc37_v2'
        and r['candidate_sha256'] == 'b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'
        and r['information_advances'] is False
        and all(r[key] is True for key in ('cleanup_passed','whole_owned_tree_retired',
            'observational_groups_retired','own_complete_bills','own_repeat_byte_exact',
            'all16_vector_gate_passed','current37_exact_replay'))
        and r['maximum_combined_rss_bytes'] < 8_000_000_000
        and r['maximum_swap_growth_bytes'] == 0 and 0 < r['maximum_interval_s'] <= 1
        and 0 < r['elapsed_s'] <= f['elapsed_s'] < 300
        and 0 < f['terminal_publication_interval_s'] <= 1,
        'CB2 operationally positive, runtime advancement still negative')
    rows = r['records'][1]
    require(len(rows) == 2 and tuple(row['flops'] for row in rows) == (411230669135,404454713141)
        and [row['compact_binding_flops'] for row in rows] == [3488967424,0]
        and [row['compact_binding_events'] for row in rows] == [27,0]
        and r['initialization_flops'] == [None,6775955994]
        and [row['timing']['wall_time_s'] for row in rows] == [47.658680166,46.81287175],
        'Exact same-source CB2 cold/repeat costs and historical timings')
    for index,row in enumerate(rows):
        check_row(row,index)
        path = NEW/f'current_comp_compact_boxes37_v2_{index}_r37.f32'
        require(row['output_sha256'] == BASE_OUTPUT and r['artifacts'][str(path)] == BASE_OUTPUT
            and sha(path,tick) == BASE_OUTPUT,'Both actual CB2 all16 output controls')
        result['input_bindings'][str(path)] = BASE_OUTPUT
    for path,digest in zip(paths,pins):
        result['input_bindings'][str(path)] = digest
    result.update(cb2_control_seals=list(pins),cb2_control_records=rows,
        cb2_runtime_advancement_still_failed=True,fresh_cb2_control_calls=0,
        component_timing_comparison_is_historical_not_fresh_paired=True)


def _w5_complete_rows(rows):
    require(len(rows) == 2,'Exactly one new cold37/repeat pair')
    records = []
    for row in rows:
        value = row['weight_reuse']
        require(isinstance(value,dict) and value['completed'] is True
            and value['numeric_cache_retired'] is True
            and value['additional_initialization_flops'] == 0
            and value['creators'] == 15 and value['hits'] == 41
            and value['copied_flops'] == 230188672
            and value['removed_flops'] == 1363482848
            and value['net_saved_flops'] == 1133294176
            and value['maximum_cache_bytes'] == 68841472
            and value['source_certified_arithmetic_copy_calls_saved'] == 1215
            and len(value['layers']) == 16,
            'Actual per-call W5 scalar lifecycle/creator/hit/copy receipt')
        totals = dict(creators=0,hits=0,copy_flops=0,removed_flops=0)
        for layer,item in enumerate(value['layers']):
            k = item['previous_count']
            require(type(k) is int and 0 < k <= 1024 and k%32 == 0,'Actual aligned support count')
            roles = {} if layer == 0 else dict(cov_outer=1,newborn_common=1,star_push=1)
            if layer >= 5:
                roles['shared_basis'] = 1
            creators,hits = (0,0) if layer == 0 else (1,len(roles)-1)
            copied,removed = (0,0) if layer == 0 else (16807*k,hits*36827*k)
            capacity = 0 if layer == 0 else 67228*(1<<(k-1).bit_length())
            require(item == dict(layer=layer,previous_count=k,roles=roles,
                creators=creators,hits=hits,copy_flops=copied,removed_flops=removed,
                capacity_bytes=capacity),'Independent exact per-layer cache price/provenance roster')
            for key in totals:
                require(type(item[key]) is int,'Exact integer creator/hit/paid-cost metadata')
                totals[key] += item[key]
        require(tuple(totals.values()) == (15,41,230188672,1363482848),
            'All actual layer contributions reconcile complete W5 delta')
        records.append(value)
    require(records[0] == records[1],'Fresh per-call W caches, same complete deterministic scalar history')
    return records
# END DIRECT SAVED CB2 COMPONENT CONTROL V1
'''


def _load_owner():
    for path,digest in SOURCE_PINS.items():
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Exact full-call source owner: '+str(path))
    spec = importlib.util.spec_from_file_location('_w5_full_watch_owner',OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _derived():
    fast = _load_owner()
    base = fast._owner()
    original,parent = fast.derive_source()
    source,positions = original,[]

    def replace(before,after,count=1):
        nonlocal source
        if source.count(before) != count:
            raise RuntimeError('Exact whole-supervisor W5 splice: '+before[:100])
        for unused in range(count):
            at = source.index(before); positions.append((at,before,after))
            source = source[:at]+after+source[at+len(before):]

    replace("CLIENT = HERE/'client.py'","CLIENT = HERE/'full_client.py'")
    replace("CLIENT: 'e815c9c82b7730598a63832855b357f2ea6d825fe16b1b8942135a1706f25622'",
            "CLIENT: '"+CLIENT_SHA+"'")
    replace("HERE/'watch_fast_v1.py':args.source_sha256","HERE/'full_watch.py':args.source_sha256")
    replace("HERE/'compose_fast_v1.py'","HERE/'port.py'",2)
    replace("HERE/'FAST_COMPOSITION_SPEC.md':args.protocol_sha256",
            "HERE/'FULL_RPC_PROTOCOL.md':args.protocol_sha256")
    replace("'current_comp_compact_boxes37_v2'","'current_comp_weight_reuse37_v1'")
    replace("'current_comp_compact_boxes_rpc37_v2'","'current_comp_weight_reuse_rpc37_v1'")
    replace("prefix='astra_compact_boxes_rpc37_'","prefix='astra_weight_reuse_rpc37_'")
    replace('        negative_control(result,tick)\n',
            '        negative_control(result,tick)\n        _w5_cb2_control(result,tick)\n')
    replace('        result[\'client_source_proof\'] = client_proof\n',
            '        result[\'client_source_proof\'] = client_proof\n'
            '        result[\'supervisor_source_proof\'] = _W5_WATCH_PROOF\n')
    replace("proof['full_source_inverse_verified'] and proof['parent_sha256'] == "
            "'378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'"
            " and proof['c2_ancestor_sha256'] == BASE_SHA",
            "proof['full_source_inverse_verified'] and proof['parent_sha256'] == "
            "'b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'"
            " and proof['original_product_inverse_verified']"
            " and proof['qualified_copy_generalization_inverse_verified']"
            " and proof['depth_other_than_5_unchanged']")
    replace("'Exact BOX ancestor and original c2-head source inverse'",
            "'Exact CB2 parent plus qualified copied-left D5 source inverse'")
    replace("        init = [None,new[0]['flops']-new[1]['flops']]\n",
            "        weight_reuse = _w5_complete_rows(new)\n"
            "        result['weight_reuse'] = weight_reuse\n"
            "        init = [None,new[0]['flops']-new[1]['flops']]\n")
    replace("(BASE_F[0]+EXTRA_BINDING_FLOPS+extra[0],BASE_F[1]+extra[1])",
            "(BASE_F[0]+EXTRA_BINDING_FLOPS+extra[0]-weight_reuse[0]['net_saved_flops'],"
            "BASE_F[1]+extra[1]-weight_reuse[1]['net_saved_flops'])")
    replace("'Exact own cold/warm bills reconcile every executed reshape'",
            "'Exact complete bills reconcile every reshape, copied leaf and omitted repeated transform'")
    replace("        result['complete_cold_cost_gate_passed'] = cost_gate\n",
            "        result['complete_cold_cost_gate_passed'] = cost_gate\n"
            "        component_ratios = [new[i]['timing']['wall_time_s']/"
            "result['cb2_control_records'][i]['timing']['wall_time_s'] for i in range(2)]\n"
            "        result.update(component_wall_ratios_to_saved_cb2=component_ratios,\n"
            "            component_retained=all(value<=.99 for value in component_ratios),\n"
            "            release_runtime_gates_passed=advances,release_ready=False,\n"
            "            full_domain_selector_qualified=False,native_rpc_candidate_parity_qualified=False)\n")
    replace('        submission_authorized=False,parameter_updates=0,information_advances=False,\n',
            '        submission_authorized=False,parameter_updates=0,information_advances=False,\n'
            '        component_retained=False,release_runtime_gates_passed=False,release_ready=False,\n')
    inverse = source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)] != after:
            raise RuntimeError('Exact complete W5 supervisor positional inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    if inverse != original:
        raise RuntimeError('Entire qualified effective CB2 supervisor restored')
    source += CONTROL_SOURCE
    proof = dict(parent=parent,owner_sha256=OWNER_SHA,
        full_effective_CB2_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_observation_cleanup_publication_and_context_gates_unchanged=True,
        same_single_server_client_lifetime=True,full_model_calls=2,ranks=[37,37],
        fresh_baseline_calls=0,component_comparator_is_saved_CB2=True,
        required_complete_flops=[410097374959,403321418965],
        required_initialization_flops=6775955994,
        required_compact_binding_flops=[3488967424,0],required_compact_binding_events=[27,0],
        component_retention_maximum_both_historical_ratios=.99,
        release_both_wall_cap_s=45,release_original_c2_cold_ratio=.80,
        release_cold_cost_ratio_to_original_c2=1.015,
        information_advances_keeps_original_release_runtime_semantics=True,
        full_native_selector_or_score_qualification_inherited=False,
        source_bindings={str(path):digest for path,digest in SOURCE_PINS.items()},
        numerical_work_executed=False)
    return source,proof,base


def derive_source():
    source,proof,unused = _derived()
    return source,proof


def main():
    source,proof,base = _derived()
    namespace = dict(base.main.__globals__,__file__=str(Path(__file__).resolve()),
                     __name__='_weight_reuse_effective_full_watch')
    inherited = dict(namespace['PINS'])
    exec(compile(source,str(Path(__file__).resolve())+':whole-CB2-inverse','exec'),namespace)
    namespace['PINS'].update(inherited)
    namespace['PINS'].update(SOURCE_PINS)
    namespace['_W5_WATCH_PROOF'] = proof
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
