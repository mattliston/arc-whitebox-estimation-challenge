"""Explicit, separate source-build and fixed37/37 padded-PUSH runtime watches."""
import argparse
import ast
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import struct
import sys
import time
import types

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
OWNER = APUSH/'watch.py'
OWNER_SHA = 'd6eaa5714b42836b530ae3db153dbef12fb2764f01d8403f917848b0d867bd91'
PARENT = APUSH/'build_v1/estimator.py'
PARENT_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
CANDIDATE_SHA = '60ad89e61f95c6447f7e3fbc5636f2a2a48248e2fd62d8d043c5ce94ddf6543c'
CANDIDATE = 'current_comp_padded_push_d6_v1/build_v1/estimator.py'
TAG = 'current_comp_padded_push_d6_runtime_v1'
INIT, OLD_INIT = 1954407482, 1944970298
CONTROL_F = (461231998786,459287028488)
OUTPUT_FILE_SHA = '564b106324446b68284989d8c93755a75a5e3f44f618401cb87f185492b3418d'
OUTPUT_RAW_SHA = 'ba45eae829b5bad9c8827c000564d6d16e917a8a5488b9c9631270e6acaa04b6'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ROW = STORE/'rows/observer_fit/0037'
FIXTURE_PINS = {
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    ROW/'row_manifest.json':'900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
    ROW/'weights.npy':'967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4',
}
FROZEN = {
    'build':('build.py','63ae8a312de114ab03e02e91916e63f997c81475104a0d8351da7dc652aad6f7'),
    'protocol':('PROTOCOL.md','6810ed66afd679210eb17f8c3e9ddc6506615d1d249e8112228063b856567c4c'),
    'client':('client.py','46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3'),
    'runner':('runner.py','5a966460183ecb09c6bdcff240ec64da7bc1b989cedb70d2125e5c6f13c9eb80'),
    'adapters':('ADAPTERS.md','347bf578d0335690d496f339c2350ac2cb0dc6e023872ebcf1fb5adbb575b134'),
    'adapters_v2':('ADAPTERS_V2.md','b55a033ab0f51e98f3aebd84938a7575e2dceaa8873c5d4591b1fbaac69a2483'),
}
KEYS = ('source','spec','build','client','runner','adapters','adapters_v2','protocol')
CONTROL_SOURCES = {
    PARENT:PARENT_SHA,
    APUSH/'native.py':'751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6',
    APUSH/'NATIVE_PROTOCOL.md':'96e1da8241eee46e18fcf9adb75194d22269d2894ad3f8e1e1dee3d2c6ef4c38',
    APUSH/'watch_native.py':'dca03cd8830f02faee4757383e541e759bc0170cb9299cc60d1283326af31064',
    APUSH/'NATIVE_WATCH_SPEC.md':'f6f48bd29779e9b61663250cde281937e6cc0df249d3f278871331686cf9e19b',
    APUSH/'score_v2.py':'5108fdbffaa581abecfc846863a75dca7b2095ca8c0878f31bd838eede34c60e',
    APUSH/'SCORE_PROTOCOL_V2.md':'e9a8f3e4a86fdefe2a501afacd38e50071df3afdf78e3b0e2a0182b50e372ece',
    APUSH/'watch_score.py':'09277db8bf2fd12c881fa6399d81ee103b6012f0832c2061c35e5f4ee77a1c6b',
    APUSH/'SCORE_WATCH_SPEC.md':'031fe1bad20197896801a309bde6cf8411ef5b94a77092f44e9582c4971494ca',
}
CHAINS = {
    'native_terminal_v1':('b9372a024b1257ead34fe8e36fcd85014735f73e0b2a6bbb83bfaca2b8f3c1e8',
        'ed9222dc1cfdc50409746642a8102c426b314e447e920d3b2e1a0e57e62ebba1',
        'b6ccb318ef0763fb2fb09f5266238de7ad633cb711c237e13b07230b135be5c3'),
    'native_watch_v1':('94b6e66e317cb458ebb6000370f4451fac9052f15264aae2d3ad1d49a8da388d',
        '595c8d3a2cf1da3902bba456bbdc3b667ad1880b1bab582c3f2124c74a400804',
        'b13a7bff299a26d87589ef1cddb0337a53494e882b07bb5d171fa55aef87be69'),
    'score_v2':('73b1db635ae9ec4369a7a767b7ed175ba2b0c5e96eb3b921d67b680d7a463a67',
        '0a27c7494c1a602410fff5b40008dd763e0bb2140cc2fb4e5c60fbe65f5a4f87',
        '635aea06e6500e875f41e84baa659910149db7d977c9880a92f584e66e0ca112'),
    'score_watch_v1':('56601e0862eff783e103d6fbf7a7bc71cc3874127320d06b3762bc6e17996d9f',
        '7c8e0ff73e97dacc4e9e3a627e47cf2208ab6452ec58f20bb7dde9489c5776c5',
        'f1deff31731846e5e08574e1cfe53f4763a4bc8eff06ea10f510d617f100a5e8'),
}
SEAL_NAMES = ('result.json','completion.json','finalization.json')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def seal_dict(digests):
    return dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))


def chain(h,folder,expected,inputs,tick,*,positive=True):
    paths = [folder/name for name in SEAL_NAMES]
    actual = [h.sha(p,tick) for p in paths]
    require(actual[0] == expected[0] and (len(expected) == 1 or actual == list(expected)),
        'Actual expected three-seal chain: '+str(folder))
    records = [h.read(p) for p in paths]
    require(all(x.get('passed') is positive and x.get('completed') is positive for x in records)
        and records[1]['result_sha256'] == records[2]['result_sha256'] == actual[0]
        and records[2]['completion_sha256'] == actual[1], 'Actual sign and hash links, never invented seals')
    inputs.update(zip(paths,actual))
    return records,seal_dict(actual)


def merge(target,items):
    for path,digest in items.items():
        path = str(Path(path).resolve())
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and
            (path not in target or target[path] == digest),'Flat inventory digest agreement')
        target[path] = digest


def inventory(h,record,result,tick):
    for name in ('source_bindings','input_bindings','artifacts'):
        values = record[name]
        require(isinstance(values,dict) and values,'Actual nonempty new inventory '+name)
        for path,digest in values.items():
            require(h.sha(path,tick) == digest,'Actual target-free dependency '+str(path))
        merge(result[name],values)


def owners(args,h):
    own = {HERE/'watch.py':args.source_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        OWNER:OWNER_SHA,**{HERE/name:digest for name,digest in FROZEN.values()},
        h.__file__:h.sha(h.__file__),**h.MON_PINS,
        NEW/'run_remote_series_v1.py':'00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1',
        NEW/'check_remote_series_v1.py':'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'}
    require(own[h.__file__] == 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e',
        'Only qualified target-free transport')
    inputs = dict(h.MOCK_PINS)
    if args.phase == 'runtime':
        own.update(CONTROL_SOURCES)
        inputs.update(FIXTURE_PINS)
        for folder,digests in CHAINS.items():
            inputs.update({APUSH/folder/name:digest for name,digest in zip(SEAL_NAMES,digests)})
    return own,inputs


def paths(args):
    out = HERE/('build_watch_v1' if args.phase == 'build' else 'runtime_v1')
    series,usage,server_log = [NEW/(TAG+x) for x in ('.json','_resources.json','_server.log')]
    outputs = [NEW/f'{TAG}_{i}_r37.f32' for i in range(2)]
    require(not out.exists() and not HERE.is_symlink() and
        not any(p.exists() for p in (series,usage,server_log,*outputs)), 'Exclusive fresh watch/runtime paths')
    if args.phase == 'build':
        require(not (HERE/'build_v1').exists() and not (HERE/'runtime_v1').exists(),
            'Source build only once; never overwrite/resume a failed stage')
    else:
        require((HERE/'build_v1').is_dir() and (HERE/'build_watch_v1').is_dir(),
            'Runtime requires a real separately completed externally watched build')
    return out,series,usage,server_log,outputs


def checked_build(args,h,result,inputs,tick,expected):
    (record,end,final),seals = chain(h,HERE/'build_v1',expected,inputs,tick)
    require(record['schema'] == 'current_comp_padded_push_d6_build_v1'
        and record['candidate_sha256'] == CANDIDATE_SHA and record['parameter_updates'] == 0
        and all(record.get(k) is True for k in ('sources_unchanged','hashes_unchanged'))
        and all(record.get(k) is False for k in ('estimator_executed','estimator_imported',
            'truth_arrays_accessed','truth_accessed','score_measured','numerical_imports'))
        and 0 <= record['elapsed_s'] < 60 and 0 < record['peak_rss_bytes'] < 2_000_000_000
        and 0 <= final['elapsed_s'] < 60, 'Actual source-only PAD build, no numerical acquisition')
    proof = record['proof']
    require(all(proof.get(k) is True for k in ('whole_source_inverse_verified','entire_AST_change_scope_verified',
        'unchanged_imports_literals_heads_physics_and_kernel_arithmetic','original_aligned_branch_byte_unchanged',
        'no_read_or_square_change','same_original_backing_arena','padding_region_disjoint_from_all_live_serial_views',
        'original_caller_output_retained','square_callers_excluded_by_owner_identity',
        'original_serial_busy_and_child_bypass_unchanged','padding_busy_guard_and_full_per_call_overwrite',
        'no_new_full_backing_array','no_import_time_array_work'))
        and proof['parent_sha256'] == PARENT_SHA and proof['candidate_sha256'] == CANDIDATE_SHA
        and proof['backing_arena_bytes'] == 1168900096 and proof['serial_required_elements'] == 118006528
        and proof['padding_specs'] == [[118006528,119055104,[1024,1024]],
            [119055104,127443712,[1024,8192]]]
        and proof['padding_region_bytes'] == 37748736
        and proof['extra_cold_binding_flops'] == 9437184
        and proof['expected_complete_initialization_flops'] == INIT
        and proof['extra_cold_reshape_calls'] == 2 and proof['extra_cold_FP32_zero_calls'] == 1
        and proof['per_padded_call_copy_calls'] == 4
        and proof['per_padded_call_added_copy_flops'] == '(k+32)*(1024+n)'
        and proof['new_policy'] == 'FP32 original PUSH owner; m1024, k%64=32, 0<k<=1024, 1024<n<=8192, n%64=0; kp=k+32'
        and proof['parameter_updates'] == 0 and proof['estimator_executed'] is False,
        'Exact source inverse, same arena/lifetimes, both zero strips, both input copies and paid startup')
    require(record['source_bindings'][str(HERE/'build.py')] == args.build_sha256
        and record['source_bindings'][str(HERE/'PROTOCOL.md')] == args.protocol_sha256
        and record['artifacts'] == {str(HERE/'build_v1/estimator.py'):CANDIDATE_SHA},
        'Actual candidate output and reviewed frozen build owners')
    inventory(h,record,result,tick)
    result.update(candidate_sha256=CANDIDATE_SHA,build_seals=seals,
        exact_build_source_and_padding_price_verified=True,build_proof=proof)
    return record,seals


def qualified_build(args,h,result,inputs,tick):
    record,seals = checked_build(args,h,result,inputs,tick,(args.build_result_sha256,))
    (watch,end,final),ws = chain(h,HERE/'build_watch_v1',(args.build_watch_sha256,),inputs,tick)
    require(watch['schema'] == 'current_comp_padded_push_d6_build_watch_v1' and watch['phase'] == 'build'
        and watch['candidate_sha256'] == CANDIDATE_SHA and watch['build_seals'] == seals
        and watch['full_model_calls'] == 0 and watch['build_falsifier_passed'] is True
        and all(watch.get(k) is True for k in ('hashes_unchanged','sources_unchanged','whole_owned_tree_retired',
            'child_execution_completed','cleanup_passed','observational_groups_retired','actual_completed_samples',
            'foreign_numerical_workers_checked','exact_build_source_and_padding_price_verified'))
        and watch['numerical_imports'] is False and watch['truth_arrays_accessed'] is False
        and watch['submission_authorized'] is False and watch['user_explicit_confirmation_bank_authorized'] is False
        and watch['maximum_swap_growth_bytes'] == 0 and 0 < watch['maximum_combined_rss_bytes'] < 2_000_000_000
        and 0 < watch['maximum_interval_s'] <= 1 and 0 <= watch['elapsed_s'] < 60
        and 0 <= final['elapsed_s'] < 60 and 0 <= final['terminal_publication_interval_s'] <= 1,
        'Actual positive separate60s/2GB full-tree source-build watch and three seals')
    require(watch['source_bindings'][str(HERE/'watch.py')] == args.source_sha256
        and watch['source_bindings'][str(HERE/'WATCH_SPEC.md')] == args.spec_sha256,
        'Same reviewed supervisor and specification for both explicit phases')
    inventory(h,watch,result,tick)
    result.update(build_watch_seals=ws,external_build_qualified=True)


def qualified_control(h,result,inputs,tick):
    (native,ne,nf),ns = chain(h,APUSH/'native_terminal_v1',CHAINS['native_terminal_v1'],inputs,tick)
    (failed,fe,ff),fs = chain(h,APUSH/'native_watch_v1',CHAINS['native_watch_v1'],inputs,tick,positive=False)
    (score,se,sf),ss = chain(h,APUSH/'score_v2',CHAINS['score_v2'],inputs,tick)
    (watch,we,wf),ws = chain(h,APUSH/'score_watch_v1',CHAINS['score_watch_v1'],inputs,tick)
    require(failed['schema'] == 'current_comp_aligned_push_d6_native_watch_v1'
        and failed['retired_call_count'] == 5 and failed['full_model_calls'] == 0
        and "KeyError: 'input_bindings'" in failed['error'], 'Original exact negative watch remains failed')
    require(score['schema'] == 'current_comp_aligned_push_d6_score_v2' and score['candidate_sha256'] == PARENT_SHA
        and all(score.get(k) is True for k in ('complete_APUSH_acquisition_reverified','saved_external_watch_revalidated',
            'integrity_passed','all_five_new_outputs_verified','all_five_domains_and_selectors_reverified',
            'own_complete_bills','repaired_metadata_only','original_external_watch_still_failed','sources_unchanged','hashes_unchanged'))
        and score['numerical_acquisition_repeated'] is False and score['original_observation_samples_verified'] == 644
        and score['parameter_updates'] == score['full_model_calls'] == 0 and score['training_occurred'] is False
        and score['submission_authorized'] is False and score['user_explicit_confirmation_bank_authorized'] is False
        and 0 <= score['elapsed_s'] < 120 and 0 < score['peak_rss_bytes'] < 1_000_000_000
        and 0 <= sf['elapsed_s'] < 120 and score['lineage']['terminal']['native'] == ns
        and score['lineage']['terminal']['watch'] == fs
        and score['requested_seals']['native_sha256'] == ns['result_sha256']
        and score['requested_seals']['native_watch_sha256'] == fs['result_sha256'],
        'Positive completed saved-acquisition authority only, no repair or truth rerun')
    require(watch['schema'] == 'current_comp_aligned_push_d6_score_watch_v1'
        and watch['candidate_sha256'] == PARENT_SHA
        and [watch[k] for k in ('child_result_sha256','child_completion_sha256','child_finalization_sha256')]
            == list(CHAINS['score_v2'])
        and all(watch.get(k) is True for k in ('sources_unchanged','hashes_unchanged','whole_owned_tree_retired',
            'child_execution_completed','cleanup_passed','observational_groups_retired','actual_completed_samples',
            'foreign_numerical_workers_checked'))
        and watch['numerical_imports'] is False and watch['maximum_swap_growth_bytes'] == 0
        and watch['whole_cap_s'] == 120 and watch['combined_rss_limit_bytes'] == 1_000_000_000
        and 0 < watch['maximum_combined_rss_bytes'] < 1_000_000_000 and 0 < watch['maximum_interval_s'] <= 1
        and 0 <= watch['elapsed_s'] < 120 and 0 <= wf['elapsed_s'] < 120
        and 0 <= wf['terminal_publication_interval_s'] <= 1, 'Actual positive saved-score supervision')
    for path,digest in CONTROL_SOURCES.items():
        require(score['source_bindings'][str(path)] == digest,'Original control/checker source owner')
    require(native['schema'] == 'current_comp_aligned_push_d6_native_v1' and native['candidate_sha256'] == PARENT_SHA
        and native['full_model_calls'] == native['retired_call_count'] == 5
        and native['ranks'] == [37,39,42,48]
        and [row['rank'] for row in native['trials']] == [37,39,42,48,37]
        and [row['call'] for row in native['trials']] == list(range(5))
        and native['initialization_flops'] == OLD_INIT
        and native['input_preparation_in_participant_bill'] is False
        and all(native.get(k) is True for k in ('sources_unchanged','hashes_unchanged','own_complete_bills',
            'all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
            'additional_bindings_paid_once','same_existing_arena_qualified','dispatch_tariffs_qualified')),
        'Actual native controls and complete bills, not borrowed production timing')
    controls = []
    for ordinal,index in enumerate((0,4)):
        trial = native['trials'][index]
        folder = APUSH/'native_terminal_v1'
        paths = [folder/f'call_{index:02d}_{suffix}' for suffix in
            ('bill.json','input_bill.json','input_reset.json','lifecycle.json','diagnostics.json','rank37.npy')]
        for path in paths:
            digest = native['artifacts'][str(path)]
            require(h.sha(path,tick) == digest and score['input_bindings'][str(path)] == digest,
                'Selected control artifact bound by positive saved-acquisition admission')
            inputs[path] = digest
        bill,inp,reset,life,diag = [h.read(path) for path in paths[:-1]]
        require(trial == bill and bill['call'] == index and bill['rank'] == 37
            and bill['flops'] == CONTROL_F[ordinal] and bill['multiplier'] == max(.1,bill['flops']/2**41)
            and bill['includes_initialization'] is (index == 0)
            and bill['output'] == str(paths[-1]) and bill['output_sha256'] == OUTPUT_FILE_SHA
            and bill['output_bytes_sha256'] == OUTPUT_RAW_SHA
            and inp['weight_file_sha256'] == FIXTURE_PINS[ROW/'weights.npy']
            and inp['call'] == index and inp['rank'] == 37 and inp['flops'] == 16777216
            and inp['operation_count'] == inp['layers'] == 16 and inp['charged_to_participant'] is False
            and all(inp[k] is True for k in ('exact_bytes','official_fnp_type','independent_weight_buffers','context_closed')),
            'Exact canonical37 physical inputs, original model bill and output payload')
        for rec,path,digest,f,ops in ((reset,paths[1],native['artifacts'][str(paths[1])],16777216,16),
                (life,paths[0],native['artifacts'][str(paths[0])],bill['flops'],bill['operation_count'])):
            require(rec['saved_bill'] == str(path) and rec['saved_bill_sha256'] == digest
                and rec['closed_metrics']['flops'] == f and rec['closed_metrics']['operation_count'] == ops
                and rec['call'] == index and rec['rank'] == 37 and rec['reset_api'] == 'flopscope.budget_reset'
                and rec['candidate_state_reset'] is False
                and all(rec[k] is True for k in ('after_context_exit','closed_context_bill_unchanged_by_reset',
                    'context_released','complete')), 'Actual independent-input and prediction context retirement')
        require(diag['call'] == index and diag['rank'] == 37
            and diag['domains']['variance_count'] == 16 and diag['domains']['selector_count'] == 47
            and diag['truth_accessed'] is False and diag['old_bill_transplanted'] is False,
            'All16/all47 actual saved control diagnostics')
        controls.append(dict(flops=bill['flops'],output=str(paths[-1]),output_sha256=OUTPUT_FILE_SHA,
            output_bytes_sha256=OUTPUT_RAW_SHA,call=index,rank=37))
    rows = h.read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    require(rows[37]['role_rank'] == 37 and rows[37]['path'] == 'rows/observer_fit/0037/row_manifest.json'
        and rows[37]['sha256'] == FIXTURE_PINS[ROW/'row_manifest.json']
        and h.read(ROW/'row_manifest.json')['files']['weights.npy']['sha256'] == FIXTURE_PINS[ROW/'weights.npy'],
        'Original remote client list-index37 is the exact native37 physical fixture')
    result.update(control_seals=dict(native=ns,negative_native_watch=fs,positive_saved_score=ss,positive_score_watch=ws),
        actual_positive_control_qualified=True,original_native_watch_still_failed=True,
        control_complete_flops=list(CONTROL_F),input_billing=dict(runtime_load_and_basic_views_flops=0,
            native_separate_input_flops=16777216,native_input_charged_to_participant=False,
            any_input_or_setup_flops_subtracted=False,production_timing_borrowed=False))
    return controls


def prepare(args,h,result,inputs,tick,source_proof,load):
    result['supervisor_source_proof'] = source_proof
    if args.phase == 'build':
        return [str(P2/'data_gate/.venv/bin/python'),'-B','-u',str(HERE/'build.py'),'--run',
            '--source-sha256',args.build_sha256,'--protocol-sha256',args.protocol_sha256],[]
    qualified_build(args,h,result,inputs,tick)
    controls = qualified_control(h,result,inputs,tick)
    proofs = {}
    for name in ('client','runner'):
        module = load(HERE/(name+'.py'),'padded_push_'+name+'_source')
        source,proofs[name] = module.source_and_proof()
        require(proofs[name]['whole_source_inverse_verified'] is True,'Exact original adapter inverse')
        compile(source,str(HERE/(name+'.py'))+':inert_source_review','exec'); tick()
    require(proofs['client']['ranks'] == [37,37] and proofs['client']['persistent_estimator'] is True,
        'Exactly one unchanged persistent estimator and fixed37/37')
    result['adapter_proofs'] = proofs
    return [str(P2/'data_gate/.venv/bin/python'),'-B','-u',str(HERE/'runner.py'),
        '--run','--candidate',CANDIDATE,'--tag',TAG],controls


def process_scope(snapshot,allowed,child,result,args):
    server,client = [],[]
    for row in snapshot['rows']:
        try: argv = shlex.split(row['command'])
        except ValueError: argv = row['command'].split()
        if 'flopscope_server' in argv:
            require(row['pid'] in allowed and args.phase == 'runtime','No foreign or source-stage numerical server')
            server.append(row['pid'])
        if str(HERE/'client.py') in argv:
            require(row['pid'] in allowed and args.phase == 'runtime','Only the owned fixed client')
            client.append(row['pid'])
    require(len(server) <= 1 and len(client) <= 1,'At most one persistent numerical server/client')
    if result.get('persistent_processes'):
        declared = result['persistent_processes']
        require(all(pid == declared['server_pid'] for pid in server)
            and all(pid == declared['client_pid'] for pid in client),'No server/client restart or second instance')


def process_marker(row,args,result,child):
    require(args.phase == 'runtime' and 'persistent_processes' not in result and child is not None
        and all(type(row[k]) is int and row[k] > 1 for k in ('server_pid','client_pid'))
        and len({child.pid,row['server_pid'],row['client_pid'],os.getpid()}) == 4,
        'Exactly one original runner declaration, two distinct persistent owned child identities')
    result['persistent_processes'] = {k:row[k] for k in ('server_pid','client_pid')}


def npy_payload(path):
    path = Path(path)
    require(path.stat().st_size < 128*1024 and not path.is_symlink(),'Bounded original all16 output only')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == OUTPUT_FILE_SHA and data[:6] == b'\x93NUMPY',
        'Authenticated native NPY container, not a raw-payload hash')
    version = tuple(data[6:8])
    require(version in ((1,0),(2,0)),'Explicit little-endian NPY1/2 header')
    offset = 10 if version == (1,0) else 12
    size = struct.unpack('<H' if version == (1,0) else '<I',data[8:offset])[0]
    require(0 < size < 4096 and len(data) == offset+size+16384*4,'Exact NPY header/payload boundary')
    header = ast.literal_eval(data[offset:offset+size].decode('latin1'))
    require(header == dict(descr='<f4',fortran_order=False,shape=(16,1024)),
        'Native reference has exactly original FP32 all16 shape/order')
    payload = data[offset+size:]
    require(hashlib.sha256(payload).hexdigest() == OUTPUT_RAW_SHA,'Exact original raw payload binding')
    return payload


def finish(args,h,result,inputs,tick,series,usage,server_log,outputs,controls):
    if args.phase == 'build':
        expected = (h.sha(HERE/'build_v1/result.json',tick),)
        checked_build(args,h,result,inputs,tick,expected)
        result.update(build_falsifier_passed=True,external_build_qualified=True,runtime_falsifier_passed=False)
        return
    require('persistent_processes' in result,'Actual original single-server/client launch announcement')
    measured,usage_record = h.read(series),h.read(usage)
    rows = result['observed_records']
    require(measured['status'] == 'completed' and measured['source_sha256'] == CANDIDATE_SHA
        and measured['candidate'] == usage_record['candidate'] == CANDIDATE
        and measured['records'] == rows and measured['repeated_output_byte_identical'] is True
        and rows[0]['flops']-rows[1]['flops'] == INIT
        and rows[0]['output_sha256'] == rows[1]['output_sha256'], 'Actual complete bills, paid startup and repeat')
    references = [npy_payload(row['output']) for row in controls]
    require(references[0] == references[1],'Original native cold/repeat payloads exactly equal')
    require(outputs[0].read_bytes() == outputs[1].read_bytes(),'New actual persistent repeat bytes exact')
    fidelity = []
    for row,path,base,payload in zip(rows,outputs,controls,references):
        values = struct.unpack('<16384f',path.read_bytes()); reference = struct.unpack('<16384f',payload)
        require(all(math.isfinite(x) for x in (*reference,*values)),'Finite old and new all16 outputs')
        drift = [abs(a-b) for a,b in zip(values,reference)]
        per_layer = [max(drift[j*1024:(j+1)*1024]) for j in range(16)]
        require(max(drift) <= 1e-5 and row['flops'] < base['flops'],
            'Both fixed fidelity and strict own complete-cost gates, no native-time borrowing')
        fidelity.append(dict(index=row['index'],rank=37,all16_max_abs=max(drift),per_layer_max_abs=per_layer,
            own_complete_flops=row['flops'],control_complete_flops=base['flops'],
            complete_cost_ratio=row['flops']/base['flops'],multiplier=max(.1,row['flops']/2**41)))
        tick()
    inner = usage_record['samples']
    require(inner and 0 < usage_record['peak_combined_rss_bytes'] == max(x[1] for x in inner) < 8_000_000_000,
        'Original leader-only runtime check also passes; independent full-tree watch is authoritative')
    for path in (series,usage,server_log,*outputs): result['artifacts'][str(path)] = h.sha(path,tick)
    result.update(initialization_flops=INIT,repeat_byte_exact=True,fidelity=fidelity,runtime_falsifier_passed=True,
        own_complete_bills=True,one_persistent_numpy_free_client_and_server=True,
        all16_fidelity_qualified=True,both_complete_costs_strictly_lower=True,
        all_padding_work_in_participant_bill=True,source_proven_four_copy_calls_per_padded_tile=True,
        per_shape_dispatch_census_measured=False,production_timing_borrowed=False)


def source_and_proof():
    """Only stdlib source manipulation; unchanged monitor and cleanup body."""
    raw = OWNER.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OWNER_SHA,'Qualified whole APUSH runtime-watch owner')
    original = raw.decode('utf-8'); source = original; edits = []
    def splice(before,after):
        nonlocal source
        require(before and after and source.count(before) == 1,'Unique forward supervisor splice')
        source = source.replace(before,after); edits.append((before,after))
    def between(start,end,after):
        require(source.count(start) == source.count(end) == 1,'Unique bounded source-owner region')
        i,j = source.index(start),source.index(end)
        require(i < j,'Ordered source-owner region'); splice(source[i:j],after)
    splice("and row['rank'] == 35", "and row['rank'] == 37")
    splice("'Two actual35 contexts and all16 output shape'", "'Two actual37 contexts and all16 output shape'")
    between('    parser = argparse.ArgumentParser(description=__doc__)\n',
        '    require(args.numerical_slot_clear',
        '    args = _pad_args\n    keys = _pad_keys\n    began = previous = last_progress = _pad_began\n')
    between('    own = {Path(__file__).resolve():args.source_sha256,',
        '    out.mkdir()\n',
        '    own,inputs = _pad_owners(args,h)\n'
        '    out,series,usage,server_log,outputs = _pad_paths(args)\n')
    splice("schema='current_comp_aligned_push_d6_runtime_v1',passed=False,completed=False,",
        "schema=_pad_schema,phase=args.phase,passed=False,completed=False,\n"
        "        submission_authorized=False,user_explicit_confirmation_bank_authorized=False,\n"
        "        parameter_updates=0,estimator_imported_by_supervisor=False,")
    splice("'Whole300s/next-complete-context120s'", "'Whole60s build/300s runtime and next-complete-context120s'")
    splice("        allowed = scope.pop('allowed')\n",
        "        allowed = scope.pop('allowed')\n        _pad_process_scope(snapshot,allowed,child,result,args)\n")
    splice("'Zero added swap/full-owned-tree<8GB'", "'Zero added swap/phase-specific complete owned-tree RSS cap'")
    splice("                if isinstance(row,dict) and 'index' in row and 'rank' in row:\n",
        "                if isinstance(row,dict) and 'server_pid' in row and 'client_pid' in row:\n"
        "                    _pad_process_marker(row,args,result,child)\n"
        "                if isinstance(row,dict) and 'index' in row and 'rank' in row:\n")
    splice("                    deadline();index=len(result['observed_records']);check_bill(row,index)\n",
        "                    deadline();require(args.phase == 'runtime','No model call in source build')\n"
        "                    index=len(result['observed_records']);check_bill(row,index)\n")
    splice("'Owned aligned-PUSH runtime experiment interrupted'", "'Owned padded-PUSH explicit phase interrupted'")
    between("        r,e,f=[h.read(CONTROL/name)", "        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'",
        "        argv,controls = _pad_prepare(args,h,result,inputs,tick,_pad_source_proof,load)\n"
        "        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),\n"
        "            'Stdlib-only before actual separate child launch')\n")
    splice("        require(child.wait(timeout=1)==0 and result['full_model_calls']==2,'Exactly two successful contexts')\n",
        "        require(child.wait(timeout=1)==0 and result['full_model_calls']==(0 if args.phase=='build' else 2),\n"
        "            'Successful separately reaped source child or exactly two successful real contexts')\n")
    between('        measured,usage_record=h.read(series),h.read(usage)\n',
        "        result['input_bindings'].update",
        '        _pad_finish(args,h,result,inputs,tick,series,usage,server_log,outputs,controls)\n')
    between('        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,\n',
        '        observe()\n    except BaseException:',
        "        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No supervisor numerical imports')\n"
        "        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,\n"
        "            whole_owned_tree_retired=True,child_execution_completed=True,\n"
        "            actual_completed_samples=True,foreign_numerical_workers_checked=True,\n"
        "            advancement_scope='Only the explicitly selected source build or fixed37/37 runtime falsifier; no score or automatic next stage')\n")
    inverse = source
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique reverse supervisor splice')
        inverse = inverse.replace(after,before)
    require(inverse.encode('utf-8') == raw,'Whole APUSH supervisor inverse byte-exact')
    return source,dict(owner_path=str(OWNER),owner_sha256=OWNER_SHA,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),splice_count=len(edits),
        whole_source_inverse_verified=True,original_observe_deadline_and_cleanup_loop_retained=True,
        actual_completed_sample_and_final_publication_guards_retained=True,
        metadata_and_fixed_phase_roster_adaptation_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--phase',choices=('build','runtime'))
    parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in (*KEYS,'build_result','build_watch'):
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    began = time.monotonic()
    require(args.phase is not None and args.numerical_slot_clear,'Explicit phase and exclusive numerical slot')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in KEYS),
        'All reviewed current source/spec hashes explicit')
    require(all(getattr(args,key+'_sha256') == digest for key,(name,digest) in FROZEN.items()),
        'Only the unchanged frozen builder/adapters/protocol and authority correction')
    for key in ('build_result','build_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(re.fullmatch('[0-9a-f]{64}',value or '')) if args.phase == 'runtime' else value is None,
            'Actual future build/watch result pins required only for runtime')
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'Inert stdlib-only supervisor entry')
    source,proof = source_and_proof()
    module = types.ModuleType('_current_comp_padded_push_d6_watch_body')
    module.__file__ = str(HERE/'watch.py')
    env = vars(module)
    exec(compile(source,str(OWNER)+':PAD-source-inverse','exec'),env)
    env.update(_pad_args=args,_pad_keys=KEYS,_pad_began=began,_pad_source_proof=proof,
        _pad_owners=owners,_pad_paths=paths,_pad_prepare=prepare,_pad_finish=finish,
        _pad_process_scope=process_scope,_pad_process_marker=process_marker,
        _pad_schema='current_comp_padded_push_d6_build_watch_v1' if args.phase == 'build' else 'current_comp_padded_push_d6_runtime_v1',
        CANDIDATE=CANDIDATE,TAG=TAG,RANKS=() if args.phase == 'build' else (37,37),INIT=INIT,
        CAP=60. if args.phase == 'build' else 300.,RSS=2_000_000_000 if args.phase == 'build' else 8_000_000_000)
    return env['main']()


if __name__ == '__main__': raise SystemExit(main())
