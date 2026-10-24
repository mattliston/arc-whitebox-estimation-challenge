"""Separate source-build and unchanged five-call runtime supervision; inert."""
import argparse
import ast
import hashlib
import importlib.util
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
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
OWNER = NEW/'current_comp_padded_push_d6_v1/watch.py'
OWNER_SHA = 'd1f14bfe01aba14c7b8bedb67513b8e85759107fff4b2d5438fe6784428f3de3'
LOOP_OWNER = NEW/'current_comp_aligned_push_d6_v1/watch.py'
LOOP_OWNER_SHA = 'd6eaa5714b42836b530ae3db153dbef12fb2764f01d8403f917848b0d867bd91'
BILL_OWNER = NEW/'current_comp_partial_d6_v1/production_watch_v2.py'
BILL_OWNER_SHA = '5e375679330dfdc0623140aede5a5bb3851f71588387a8e65d54e6d889f2361f'
CANDIDATE = 'current_comp_shared_old384_v1/build_v1/estimator.py'
CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
TAG = 'current_comp_shared_old384_runtime_v1'
RANKS, INIT = (35,100,500,3016500,35),1944970298
KEYS = ('source','spec','build','protocol','runner','client')
FUTURE = ('build_result','build_watch','check4_score','check4_watch')
SEALS = ('result.json','completion.json','finalization.json')
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MANIFEST_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
FIXTURES = {
    35:('5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
        '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'),
    100:('c1f755a84e36825df94871b49ce83884030b34ae8e307930f7ef7d7a6f54a9a9',
        'c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5'),
    500:('28097b1aafbb3fdd003ef519357610afe8d7d0aab7300bccd6d203c2349abc03',
        '1560fc1442febdbee59a98398241642a526a0df34ba780d17f4020c2ed76007a'),
}
FRESH_WEIGHT_SHA = '5a91b99030d9cd0083add358a0aa1f35c3f6ebb2a7451042fcd6732a5d283a98'


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def no_numerics():
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'Stdlib-only supervisor')


def source_load(path,digest,name):
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,'Exact inert source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    no_numerics()
    return module


def extract(path,name):
    text = Path(path).read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1,'One exact source-owned helper '+name)
    n = nodes[0]
    return ''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def merge(target,values):
    for path,digest in values.items():
        path = str(path)
        require(path not in target or target[path] == digest,'Consistent direct source/input binding')
        target[path] = digest


def chain(h,folder,digest,inputs,tick,cap,rss,watch=False):
    paths = [folder/name for name in SEALS]
    hashes = [h.sha(p,tick) for p in paths]
    require(hashes[0] == digest,'Explicit actual result hash, no invented future receipt')
    r,e,f = map(h.read,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == hashes[0]
        and f['completion_sha256'] == hashes[1] and r['hashes_unchanged'] and r['sources_unchanged']
        and 0 <= r['elapsed_s'] < cap and 0 <= f['elapsed_s'] < cap,'Actual positive complete boundary')
    if watch:
        require(all(r.get(k) is True for k in ('whole_owned_tree_retired','cleanup_passed',
            'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
            and r['numerical_imports'] is False and r['maximum_swap_growth_bytes'] == 0
            and 0 < r['maximum_interval_s'] <= 1 and 0 < r['maximum_combined_rss_bytes'] < rss,
            'Actual bounded external resource and complete retirement authority')
    else:
        require(0 < r['peak_rss_bytes'] < rss,'Actual bounded worker peak')
    inputs.update(zip(paths,hashes))
    return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes))


def owners(args,h):
    own = {HERE/'runtime_watch.py':args.source_sha256,HERE/'RUNTIME_PROTOCOL.md':args.spec_sha256,
        HERE/'build.py':args.build_sha256,HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,
        HERE/'runner.py':args.runner_sha256,HERE/'client.py':args.client_sha256,
        OWNER:OWNER_SHA,LOOP_OWNER:LOOP_OWNER_SHA,BILL_OWNER:BILL_OWNER_SHA,
        Path(h.__file__):'d3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e',
        NEW/'run_remote_series_v1.py':'00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1',
        NEW/'check_remote_series_v1.py':'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
        **h.MON_PINS}
    return own,dict(h.MOCK_PINS)


def paths(args):
    out = HERE/('build_watch_v1' if args.phase == 'build' else 'runtime_v1')
    series,usage,server_log = [NEW/(TAG+s) for s in ('.json','_resources.json','_server.log')]
    outputs = [NEW/f'{TAG}_{i}_r{rank}.f32' for i,rank in enumerate(RANKS)]
    require(not HERE.is_symlink() and not out.exists()
        and not any(p.exists() for p in (series,usage,server_log,*outputs)),'Exclusive source/runtime outputs')
    if args.phase == 'build':
        require(not (HERE/'build_v1').exists(),'Source build cannot overwrite or resume')
    else:
        require((HERE/'build_v1').is_dir() and (HERE/'build_watch_v1').is_dir(),
            'Runtime requires a completed separately reaped build')
    return out,series,usage,server_log,outputs


def checked_build(args,h,result,inputs,tick,digest):
    builder = source_load(HERE/'build.py',args.build_sha256,'shared_source_build_metadata')
    r,seals,pins = builder.verify_build(digest,args.build_sha256,args.protocol_sha256,tick)
    merge(result['source_bindings'],r['source_bindings'])
    merge(result['input_bindings'],pins)
    result.update(candidate_sha256=CANDIDATE_SHA,build_seals=seals,build_proof=r['proof'])
    return r,seals


def prepare(args,h,result,inputs,tick,proof,unused_loader):
    result['runtime_source_proof'] = proof
    if args.phase == 'build':
        return [str(OFFPY),'-B','-u',str(HERE/'build.py'),'--run',
            '--source-sha256',args.build_sha256,'--protocol-sha256',args.protocol_sha256],None
    _,seals = checked_build(args,h,result,inputs,tick,args.build_result_sha256)
    w,ws = chain(h,HERE/'build_watch_v1',args.build_watch_sha256,inputs,tick,60,2_000_000_000,True)
    require(w['schema'] == 'current_comp_shared_old384_build_watch_v1' and w['phase'] == 'build'
        and w['build_seals'] == seals and w['candidate_sha256'] == CANDIDATE_SHA
        and w['full_model_calls'] == 0 and w['build_falsifier_passed']
        and w['source_bindings'][str(HERE/'runtime_watch.py')] == args.source_sha256
        and w['source_bindings'][str(HERE/'RUNTIME_PROTOCOL.md')] == args.spec_sha256,
        'Same reviewed externally watched source-only build')
    result['build_watch_seals'] = ws
    s,ss = chain(h,HERE/'score_check4_v2',args.check4_score_sha256,inputs,tick,120,1_000_000_000)
    sw,sws = chain(h,HERE/'score_check4_watch_v2',args.check4_watch_sha256,inputs,tick,120,1_000_000_000,True)
    require(all(sw['child_'+k] == v for k,v in ss.items())
        and s['schema'] == 'current_comp_shared_old384_score_check4_v2'
        and sw['schema'] == 'current_comp_shared_old384_score_check4_watch_v2'
        and s['ranks'] == [37,39,42,48] and len(s['records']) == 4
        and s['information_promising'] is True and s['scientific_gate_passed'] is True
        and s['comparison']['adjusted_ratio'] <= .97 and s['comparison']['adjusted_wins'] >= 3
        and s['comparison']['maximum_adjusted_ratio'] <= 1.10
        and all(s[k] is True for k in ('integrity_passed','all_ten_outputs_verified',
            'all_predictions_verified_before_truth','exact_APUSH_score_replay','own_complete_bills',
            'complete_probe_acquisition_reverified','metadata_fresh_union_rehashed'))
        and s['full_model_calls'] == s['parameter_updates'] == 0,
        'Actual unchanged CHECK4 advancement decision; do not rerun score or read its targets')
    # Direct positive score/watch boundaries only. No recursive input/artifact
    # traversal or target hash/decode; the prior passed checker owns that scope.
    for name,key in (('score_check4_v2.py','source'),('watch_score_check4_v2.py','watcher'),
                     ('SCORE_CHECK4_V2_PROTOCOL.md','protocol')):
        path = HERE/name
        digest = s['requested_seals'][key+'_sha256']
        require(s['source_bindings'][str(path)] == digest and h.sha(path,tick) == digest,
            'Exact source that issued the positive scientific boundary')
        result['source_bindings'][str(path)] = digest
    result['check4_seals'] = dict(score=ss,watch=sws)
    manifest = STORE/'manifest.json'
    require(h.sha(manifest,tick) == MANIFEST_SHA,'Original client manifest bytes')
    inputs[manifest] = MANIFEST_SHA
    rows = h.read(manifest)['roles']['observer_fit']['row_manifests']
    fixtures = {}
    for rank,(row_sha,weight_sha) in FIXTURES.items():
        row_path = STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'
        weight_path = row_path.parent/'weights.npy'
        require(rows[rank]['role_rank'] == rank and rows[rank]['sha256'] == row_sha
            and STORE/rows[rank]['path'] == row_path and h.sha(row_path,tick) == row_sha
            and h.read(row_path)['files']['weights.npy']['sha256'] == weight_sha
            and h.sha(weight_path,tick) == weight_sha,'Exact production list-index fixture and actual bytes')
        inputs.update({row_path:row_sha,weight_path:weight_sha})
        fixtures[str(rank)] = dict(path=str(weight_path),sha256=weight_sha)
    path = P2/'public_g19_attribution_v1/fresh_3016500_weights.npy'
    require(h.sha(path,tick) == FRESH_WEIGHT_SHA,'Original fourth production fixture')
    inputs[path] = FRESH_WEIGHT_SHA
    fixtures['3016500'] = dict(path=str(path),sha256=FRESH_WEIGHT_SHA)
    result['fixture_bindings'] = fixtures
    proofs = {}
    for name in ('runner','client'):
        module = source_load(HERE/(name+'.py'),getattr(args,name+'_sha256'),'shared_remote_'+name+'_source')
        source,proofs[name] = module.source_and_proof()
        require(proofs[name]['whole_source_inverse_verified'] and proofs[name]['ranks'] == list(RANKS),
            'Original fixed five-call source and no substituted numerical backend')
        compile(source,str(HERE/(name+'.py'))+':source-only','exec')
        tick()
    require(proofs['client']['original_client_body_byte_exact'],'Entire real client numerical body unchanged')
    result['adapter_proofs'] = proofs
    no_numerics()
    return [str(OFFPY),'-B','-u',str(HERE/'runner.py'),'--run','--candidate',CANDIDATE,'--tag',TAG],None


def finish(args,h,result,inputs,tick,series,usage,server_log,outputs,unused_controls):
    if args.phase == 'build':
        checked_build(args,h,result,inputs,tick,h.sha(HERE/'build_v1/result.json',tick))
        result.update(build_falsifier_passed=True,runtime_falsifier_passed=False)
        return
    require('persistent_processes' in result,'Exactly one original server/client launch announcement')
    measured,usage_record = h.read(series),h.read(usage)
    rows = result['observed_records']
    require(len(rows) == 5 and measured['status'] == 'completed' and measured['source_sha256'] == CANDIDATE_SHA
        and measured['candidate'] == usage_record['candidate'] == CANDIDATE and measured['records'] == rows
        and measured['repeated_output_byte_identical'] is True
        and rows[0]['output_sha256'] == rows[-1]['output_sha256']
        and rows[0]['flops']-rows[-1]['flops'] == INIT
        and outputs[0].read_bytes() == outputs[-1].read_bytes(),'Actual five calls, exact own repeat and full startup')
    for path,row in zip(outputs,rows):
        raw = path.read_bytes()
        require(len(raw) == 65536 and hashlib.sha256(raw).hexdigest() == row['output_sha256'],
            'Exact all16 output payload')
        values = struct.unpack('<16384f',raw)
        require(all(math.isfinite(x) and x >= 0 for x in values),'All16 finite nonnegative real-client means')
        tick()
    inner = usage_record['samples']
    require(inner and 0 < usage_record['peak_combined_rss_bytes'] == max(x[1] for x in inner) < 8_000_000_000,
        'Original leader-only sampler also passes; full-tree watch remains authoritative')
    for path in (series,usage,server_log,*outputs):
        result['artifacts'][str(path)] = h.sha(path,tick)
    result.update(runtime_falsifier_passed=True,all5_complete=True,all5_production_timing_passed=True,
        all16_outputs_finite=True,own_complete_bills=True,repeat_byte_exact=True,initialization_flops=INIT,
        one_persistent_numpy_free_client_and_server=True,production_five_qualified=False,
        native_fidelity_qualified=False,native_fidelity_pending=True,raw_error_measured=False,
        adjusted_score_measured=False,production_timing_borrowed=False)


def source_and_proof():
    """Inverse back through the qualified PAD phases to the original live loop."""
    parent = source_load(OWNER,OWNER_SHA,'shared_phase_watch_source_only')
    original,parent_proof = parent.source_and_proof()
    require(hashlib.sha256(BILL_OWNER.read_bytes()).hexdigest() == BILL_OWNER_SHA,'Original production-five bill owner')
    old_bill = extract(LOOP_OWNER,'check_bill').replace("and row['rank'] == 35","and row['rank'] == 37").replace(
        "'Two actual35 contexts and all16 output shape'","'Two actual37 contexts and all16 output shape'")
    new_bill = extract(BILL_OWNER,'check_bill')
    edits = [
        (old_bill,new_bill),
        ("(0 if args.phase=='build' else 2)","(0 if args.phase=='build' else 5)"),
        ("parameter_updates=0,estimator_imported_by_supervisor=False,",
         "parameter_updates=0,estimator_imported_by_supervisor=False,\n"
         "        native_fidelity_qualified=False,native_fidelity_pending=True,"),
        ("'Only the explicitly selected source build or fixed37/37 runtime falsifier; no score or automatic next stage'",
         "'Only separate source build or original fixed-five timing; native fidelity/score/release pending'"),
    ]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique fixed-five phase adaptation')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique reverse fixed-five phase adaptation')
        inverse = inverse.replace(after,before)
    require(inverse == original,'Whole prior phase supervisor source recovered')
    helpers = '\n'.join(extract(OWNER,name) for name in ('process_scope','process_marker'))
    return changed,dict(owner_sha256=OWNER_SHA,parent=parent_proof,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        five_call_bill_owner_sha256=BILL_OWNER_SHA,five_call_bill_source_sha256=hashlib.sha256(new_bill.encode()).hexdigest(),
        process_helper_sha256=hashlib.sha256(helpers.encode()).hexdigest(),
        resource_cleanup_and_final_publication_loop_unchanged=True),helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--numerical-slot-clear',action='store_true')
    parser.add_argument('--phase',choices=('build','runtime'))
    for key in (*KEYS,*FUTURE):
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(args.phase is not None and args.numerical_slot_clear,'Explicit separate phase and exclusive slot')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in KEYS),'Explicit reviewed sources')
    for key in FUTURE:
        value = getattr(args,key+'_sha256')
        require(bool(re.fullmatch('[0-9a-f]{64}',value or '')) if args.phase == 'runtime' else value is None,
            'Actual completed build and CHECK4 result pins required only for runtime')
    no_numerics()
    source,proof,helpers = source_and_proof()
    helper_scope = dict(globals())
    exec(compile(helpers,str(OWNER)+':exact-persistent-process-helpers','exec'),helper_scope)
    module = types.ModuleType('_current_comp_shared_old384_watch_body')
    module.__file__ = str(HERE/'runtime_watch.py')
    env = vars(module)
    exec(compile(source,str(OWNER)+':fixed-five-phase-inverse','exec'),env)
    env.update(_pad_args=args,_pad_keys=KEYS,_pad_began=began,_pad_source_proof=proof,
        _pad_owners=owners,_pad_paths=paths,_pad_prepare=prepare,_pad_finish=finish,
        _pad_process_scope=helper_scope['process_scope'],_pad_process_marker=helper_scope['process_marker'],
        _pad_schema='current_comp_shared_old384_build_watch_v1' if args.phase == 'build' else 'current_comp_shared_old384_runtime_v1',
        CANDIDATE=CANDIDATE,TAG=TAG,RANKS=() if args.phase == 'build' else RANKS,INIT=INIT,
        CAP=60. if args.phase == 'build' else 600.,RSS=2_000_000_000 if args.phase == 'build' else 8_000_000_000)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
