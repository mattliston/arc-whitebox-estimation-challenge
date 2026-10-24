"""One saved rank37 cold score; no estimator, fit, retry or implicit next stage."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import textwrap
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
SCHEMA = 'current_comp_shared_old384_score_probe_v1'
BUDGET, CAP, RSS = 2**41, 120., 1_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
KEYS = ('source','protocol','watcher','probe-source','probe-watch-source',
        'probe-protocol','probe-watch-spec','probe','probe-watch')
PARENT = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
WEIGHT = '967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'
TRUTH = P2/'public_g22_operator_features_v1/truthpost/truthpost_rank0037.npz'
TRUTH_SHA = '48fe26d81da4b12cfa07f9ef310fe8ccbbd28558cbe61d0290ac53293fd04f0d'
OLD_SCORE = 'b9667c17e474f99e394481169f9081d5e3320ca74255e2e3e411b7eaf9995279'
OLD_WATCH = '577c84e81fe81e0f06a199463ffe0ccd68f215f50166a9cd84b2321ef7d9cf14'
OLD_NATIVE = '23b88f12449442cad00eb352e2aca1d90d277120b7ed348f800a9bcff64684a7'
OLD_NATIVE_WATCH = '8665b4705c07d14f7f315ee1e1229154177fc86e3345f12d7029fe31ed8b526b'
JOINT = NEW/'current_comp_joint_p_node_v1/score.py'
RC = NEW/'current_d50_response10_candidate_v1/score.py'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
PINS = {
    JOINT:'7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e',
    RC:'3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',
    CENSUS:'7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf',
    APUSH/'native.py':'751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6',
    HERE/'compose.py':'fdf5bd87eb6411d008043f5e989a670979cde20965366dca212146c075ee1d4e',
    HERE/'fnp.py':'980853fe0b794efcf046b360ea03bbfaa671bd48383312e8e766c9b546021d81',
    HERE/'PROTOCOL.md':'ef2331d9adfe20c8bdea5c173d9f2d19ec523b85bc0cde4a653c93ead4351607',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path, tick=None, *, truth=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2), 'Regular in-scope file')
    target = 'truthpost' in path.parts or (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower())
    require(not target or (truth and path == TRUTH), 'Only the post-seal fixed target may be opened')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def extract(path, name):
    source = Path(path).read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Unique exact helper '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


class Admission:
    def __init__(self, result, tick):
        self.result, self.tick, self.pins = result, tick, result['input_bindings']

    def pin(self, path, digest):
        path = Path(path); key = str(path)
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and (key not in self.pins or self.pins[key] == digest),
                'Consistent explicit digest')
        if key not in self.pins:
            require(sha(path,self.tick) == digest, 'Immutable input '+key)
            self.pins[key] = digest
        if path.suffix in ('.py','.md'):
            self.result['source_bindings'][key] = digest
        self.tick()

    def bind(self, mapping, unused=None):
        require(isinstance(mapping,dict), 'Flat authenticated inventory')
        for path,digest in mapping.items():
            self.pin(path,digest)

    def inventory(self, record):
        for key in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(record[key],dict) and record[key], 'Complete actual acquisition inventory '+key)
            self.bind(record[key])

    def chain(self, folder, digest, limit, rss, *, watch=False, native40=False):
        paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
        self.pin(paths[0],digest)
        hashes = [digest]+[sha(p,self.tick) for p in paths[1:]]
        for p,h in zip(paths,hashes):
            self.pin(p,h)
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == digest and f['completion_sha256'] == hashes[1]
            and 0 <= f['elapsed_s'] < limit,
            'Actual positive three-seal chain '+str(folder))
        if native40:
            require(folder == APUSH/'native40_terminal_v1' and r['full_model_calls'] == 41
                and len(r['trials']) == 41 and all(0 < t['peak_rss_bytes'] < rss
                    and 0 <= t['wall_s'] < 120 and 0 <= t['context_wall_s'] < 120 for t in r['trials']),
                'Actual historical per-call resource fields; whole scope belongs to finalization and paired watch')
        else:
            require(0 <= r['elapsed_s'] < limit, 'Actual worker/watch elapsed field')
        require(r['hashes_unchanged'] and r['sources_unchanged'], 'Sealed immutable inputs and sources')
        if watch:
            require(all(r.get(k) is True for k in ('whole_owned_tree_retired','cleanup_passed',
                'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
                and r['numerical_imports'] is False and r['maximum_swap_growth_bytes'] == 0
                and 0 < r['maximum_combined_rss_bytes'] < rss and 0 < r['maximum_interval_s'] <= 1,
                'Exact retired external resource scope')
        elif not native40:
            require(0 < r['peak_rss_bytes'] < rss, 'Actual bounded worker')
        return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes))


def linked(watch, seals):
    require(all(watch['child_'+k] == h for k,h in seals.items()), 'Watch binds exact three worker seals')


def helpers(admit):
    life = extract(CENSUS,'lifecycle')
    original = extract(RC,'trial_output')
    require(original.count('value.dtype == np.float64') == 1, 'Only actual stored-output dtype adaptation')
    trial = original.replace('value.dtype == np.float64','value.dtype == np.float32')
    require(trial.replace('value.dtype == np.float32','value.dtype == np.float64') == original, 'Full saved-output inverse')
    env = dict(Path=Path,math=math,hashlib=hashlib,require=require,read=read,
        sha=lambda p:sha(p,admit.tick),bind=admit.bind,BUDGET=BUDGET)
    exec(compile(life+'\n'+trial,'<exact-saved-output-helpers>','exec'),env)
    block = """            mse = np.mean((arrays[role][call].astype(np.float64)-target)**2, axis=1)
            raw = float(mse[-1]); f = trial['flops']; multiplier = max(.1, f/BUDGET)
            require(np.isfinite(mse).all() and (mse >= 0).all() and raw > 0 and multiplier == trial['multiplier'],
                'Finite actual raw paired with its own complete cost')
            row['per_role'][role] = dict(raw_mse=raw, flops=f, multiplier=multiplier,
                adjusted=raw*multiplier, layer_mse=mse.tolist())
"""
    require(JOINT.read_text().count(block) == 1, 'Exact unique unchanged JOINT objective body')
    env['arithmetic'] = compile(textwrap.dedent(block),'<exact-JOINT-per-net-score>','exec')
    return env,dict(saved_output_dtype_inverse_verified=True,exact_JOINT_arithmetic=True,
        arithmetic_sha256=hashlib.sha256(block.encode()).hexdigest(),
        lifecycle_sha256=hashlib.sha256(life.encode()).hexdigest(),
        saved_output_sha256=hashlib.sha256(trial.encode()).hexdigest(),legacy_module_imports=False)


def diagnostics(packet, cold):
    d,s = packet['domains'],packet['startup']
    keys = {f'{l}:{j}' for l in range(16) for j in range(2 if l == 15 else 3)}
    require(d['variance_count'] == 16 and d['selector_count'] == 47
        and set(d['variance_domain']) == {str(l) for l in range(16)} and set(d['selection_cells']) == keys
        and all(v['finite'] is True and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10
            and v['below_floor_count'] == 0
            for v in d['variance_domain'].values()), 'All16 original unfloored domains')
    require(all(c['floor'] == .01 and type(c['count']) is int and 0 <= c['count'] <= 1024
        and c['count'] % 32 == 0 for c in d['selection_cells'].values()), 'All47 original selector flags')
    require(d['added_participant_array_operations'] == s['added_participant_array_operations'] == 0
        and s['public_counters_only'] and s['original_arena_bytes'] == 1168900096 and s['no_second_backing_arena']
        and s['observed_startup_region_flops'] == (1944970298 if cold else 0)
        and s['NODE_A_materialization_flops'] == (512 if cold else 0),
        'Actual nested P/cstar/NODE and single-arena cold startup; nothing subtracted')
    # The pinned, positively supervised acquisition owns numerical selector bytes,
    # exact D6 tariffs and all startup nesting; this consumer does not reconstruct them.
    return d


def acquisition(args, result, admit, env):
    r,rs = admit.chain(HERE/'probe_v1',args.probe_sha256,300,8_000_000_000)
    w,ws = admit.chain(HERE/'probe_watch_v1',args.probe_watch_sha256,300,8_000_000_000,watch=True)
    linked(w,rs); admit.inventory(r); admit.inventory(w)
    require(r['schema'] == 'current_comp_shared_old384_native37_v1'
        and w['schema'] == 'current_comp_shared_old384_native37_watch_v1' and r['ranks'] == w['ranks'] == [37,37]
        and r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 2
        and all(r[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills')) and r['truth_accessed'] is False and r['score_measured'] is False
        and r['parameter_updates'] == 0 and r['candidate_sha256'] == w['candidate_sha256'], 'Fixed sealed two-call acquisition')
    required = {HERE/'probe.py':args.probe_source_sha256,HERE/'watch_probe.py':args.probe_watch_source_sha256,
        HERE/'PROBE_PROTOCOL.md':args.probe_protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.probe_watch_spec_sha256,
        **{p:PINS[p] for p in (HERE/'compose.py',HERE/'fnp.py',HERE/'PROTOCOL.md')}}
    require(all(r['source_bindings'].get(str(p)) == w['source_bindings'].get(str(p)) == h for p,h in required.items()),
        'Actual reviewed acquisition and construction owners')
    proof = r['composition_proof']; paid = r['paid_loop_proof']; observer = r['passive_observer_source_proof']
    require(proof == w['composition_proof'] and proof['parent_sha256'] == PARENT
        and proof['composed_source_sha256'] == r['candidate_sha256'] and proof['rank'] == 384
        and proof['first_join_layer'] == 5 and proof['last_join_layer'] == 14
        and proof['weight_rule'] == 'actual_FP32_lambda_squared'
        and all(proof[k] is True for k in ('full_source_inverse_verified','physical_function_inverse_verified',
            'learned_literals_and_imports_byte_exact','original_BD_stored_pair_repair_once',
            'original_all_row_diagonal_source_unchanged','original_omitted_radial_source_unchanged',
            'parent_arena_and_dispatchers_unchanged','no_import_time_helper_array_work'))
        and proof['extra_incumbent_plane_repair'] is False and proof['fitted'] is False,
        'Frozen one representation, original consumers, no recomposition or refit')
    require(paid['whole_loop_inverse'] and paid['no_billing_input_reset_materialization_or_repeat_change']
        and paid['owner_sha256'] == '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
        and observer['broad_push_observer_inverse_verified'] and observer['unchanged_all16_all47_capture']
        and observer['exact_shape_specific_D6_and_seven_D5_tariffs'] and observer['no_added_participant_array_operations'],
        'Exact original paid loop and passive observation source authority')
    require(len(r['artifacts']) == 14 and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 2,
        'All14 actual artifacts and two complete acquisition records')
    domains = []
    for call,t in enumerate(r['trials']):
        folder = HERE/'probe_v1'; bp = folder/f'call_{call:02d}_bill.json'
        ip = folder/f'call_{call:02d}_input_bill.json'; rp = folder/f'call_{call:02d}_input_reset.json'
        dp = folder/f'call_{call:02d}_diagnostics.json'; mp = folder/f'call_{call:02d}_complete.json'
        require(t['call'] == call and t['rank'] == 37 and t['includes_initialization'] is (call == 0)
            and type(t['flops']) is int and 0 < t['flops'] <= BUDGET and t['multiplier'] == max(.1,t['flops']/BUDGET)
            and 0 < t['peak_rss_bytes'] < 8_000_000_000 and 0 <= t['wall_s'] < 120 and 0 <= t['context_wall_s'] < 120
            and read(bp) == t and t['input_bill'] == str(ip) and t['input_bill_sha256'] == sha(ip)
            and t['input_reset_receipt'] == str(rp) and t['input_reset_sha256'] == sha(rp), 'Own ordered complete bills')
        env['lifecycle'](t,bp,Path(t['lifecycle_receipt']))
        p = read(ip)
        require(p == r['input_preparation'][call] and p['call'] == call and p['rank'] == 37
            and p['flops'] == 16777216 and p['operation_count'] == p['layers'] == 16
            and p['weight_file_sha256'] == WEIGHT and p['charged_to_participant'] is False
            and all(p[k] is True for k in ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed')),
            'Exact physical fixture and separately billed independent input copies')
        env['lifecycle']({**p,'lifecycle_receipt':str(rp)},ip,rp)
        require(r['diagnostic_records'][call] == dict(path=str(dp),sha256=sha(dp)), 'Own all-layer diagnostic owner')
        packet = read(dp); require(packet['call'] == call and packet['rank'] == 37, 'Own diagnostic call')
        domains.append(diagnostics(packet,call == 0))
        m = read(mp)
        require(m['passed'] and m['completed'] and m['context_retired'] and m['call'] == call and m['rank'] == 37
            and m['bill_sha256'] == sha(bp) and m['lifecycle_sha256'] == sha(t['lifecycle_receipt'])
            and m['diagnostic_sha256'] == sha(dp) and m['output_sha256'] == t['output_sha256']
            and w['markers'][call] == dict(call=call,rank=37,path=str(mp),sha256=sha(mp)), 'Actual post-release progress')
    require(domains[0] == domains[1] and r['initialization_flops'] == 1944970298
        and r['trials'][0]['flops']-r['trials'][1]['flops'] == r['initialization_flops'], 'Exact repeated chart and full startup')
    return r,dict(probe=rs,probe_watch=ws)


def score(args, out, result, tick):
    admit = Admission(result,tick)
    own = {**PINS,HERE/'score_probe.py':args.source_sha256,HERE/'watch_score_probe.py':args.watcher_sha256,
        HERE/'SCORE_PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'probe.py':args.probe_source_sha256,
        HERE/'watch_probe.py':args.probe_watch_source_sha256,HERE/'PROBE_PROTOCOL.md':args.probe_protocol_sha256,
        HERE/'PROBE_WATCH_SPEC.md':args.probe_watch_spec_sha256}
    admit.bind({str(p):h for p,h in own.items()})
    env,proof = helpers(admit); result['helper_source_proof'] = proof
    r,lineage = acquisition(args,result,admit,env)
    base,bs = admit.chain(APUSH/'native40_terminal_v1',OLD_NATIVE,1900,8_000_000_000,native40=True)
    bw,bws = admit.chain(APUSH/'native40_watch_v1',OLD_NATIVE_WATCH,1900,8_000_000_000,watch=True); linked(bw,bs)
    old,ss = admit.chain(APUSH/'score40_v1',OLD_SCORE,120,1_000_000_000)
    sw,sws = admit.chain(APUSH/'score40_watch_v1',OLD_WATCH,120,1_000_000_000,watch=True); linked(sw,ss)
    require(r['control_seals'] == dict(native=list(bs.values()),watch=list(bws.values()))
        and r['controls'] == [base['trials'][i] for i in (0,40)] and base['candidate_sha256'] == PARENT
        and base['full_model_calls'] == base['retired_call_count'] == 41 and base['initialization_flops'] == 1944970298
        and all(base[k] is True for k in ('own_complete_bills','all16_domains_qualified','all47_selectors_qualified',
            'repeat_output_domain_byte_exact','initialization_ownership_passed','source_inverse_verified',
            'paid_arithmetic_and_lifecycle_unchanged')), 'Direct positive current APUSH native40 control boundary')
    require(old['schema'] == 'current_comp_aligned_push_d6_score40_v1' and old['candidate_sha256'] == PARENT
        and old['integrity_passed'] and old['all_predictions_verified_before_truth'] and old['repeat_excluded']
        and old['truth_bindings'][str(TRUTH)] == TRUTH_SHA and old['records'][0]['rank'] == 37
        and old['records'][0]['target_original_dtype'] == 'float32' and old['records'][0]['target_sha256'] == TRUTH_SHA,
        'Authenticated prior rank37 scoring frame; no old truth or scoring rerun')
    # Only selected direct controls are consumed. Do not replay the prior forty-network ancestry.
    for trial in r['controls']:
        folder = APUSH/'native40_terminal_v1'; c = trial['call']
        for path in (Path(trial['output']),folder/f'call_{c:02d}_bill.json',Path(trial['lifecycle_receipt']),
                     Path(trial['input_bill']),Path(trial['input_reset_receipt']),folder/f'call_{c:02d}_diagnostics.json'):
            admit.pin(path,base['artifacts'][str(path)])
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'All acquisition admission precedes NumPy')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Exact original NumPy score arithmetic')
    for trials in (r['controls'],r['trials']):
        require(trials[0]['output_sha256'] == trials[1]['output_sha256']
            and trials[0]['output_bytes_sha256'] == trials[1]['output_bytes_sha256']
            and Path(trials[0]['output']).read_bytes() == Path(trials[1]['output']).read_bytes(),
            'Both full own repeat files byte-exact before target access')
    arrays = {'APUSH':[env['trial_output'](np,APUSH/'native40_terminal_v1',r['controls'][0],admit.pins)],
              'shared':[env['trial_output'](np,HERE/'probe_v1',r['trials'][0],admit.pins)]}
    require(arrays['shared'][0][:5].tobytes() == arrays['APUSH'][0][:5].tobytes(),
        'Unaffected five-layer parent bytes, both repeats already byte-identical')
    require(sum(v[0].nbytes for v in arrays.values()) < 200_000, 'Two distinct saved input arrays below0.2MB')
    # Repeated files were checked as bytes; no repeated numerical buffer or score.
    lineage.update(APUSH_native=bs,APUSH_native_watch=bws,APUSH_score=ss,APUSH_score_watch=sws)
    seal = out/'predictions_and_bills_sealed.json'
    write(seal,dict(truth_accessed=False,ranks=[37],repeat_excluded=True,all_four_outputs_verified=True,
        complete_probe_acquisition_reverified=True,all16_domains_and_all47_selectors_reverified=True,
        own_complete_bills=True,lineage=lineage,source_bindings=result['source_bindings'],input_bindings=dict(admit.pins)))
    result['artifacts'][str(seal)] = sha(seal,tick)
    result.update(all_predictions_verified_before_truth=True,predictions_and_bills_seal_sha256=sha(seal,tick),lineage=lineage)
    result['truth_accessed'] = True
    require(sha(TRUTH,tick,truth=True) == TRUTH_SHA, 'Only original sealed rank37 target')
    result['truth_bindings'][str(TRUTH)] = TRUTH_SHA
    with np.load(TRUTH,allow_pickle=False) as archive:
        original = archive['mc_post']; dtype = str(original.dtype); target = original.astype(np.float64)
    require(original.dtype == np.float32 and target.shape == (16,1024) and np.isfinite(target).all(),
        'Actual stored FP32 mc_post, promoted not relabeled FP64')
    row = dict(call=0,rank=37,target_sha256=TRUTH_SHA,target_original_dtype=dtype,per_role={})
    for role in ('APUSH','shared'):
        trial = r['controls'][0] if role == 'APUSH' else r['trials'][0]
        scope = dict(np=np,arrays=arrays,role=role,call=0,target=target,trial=trial,BUDGET=BUDGET,require=require,row=row)
        exec(env['arithmetic'],scope)
    require(row['per_role']['APUSH'] == old['records'][0]['per_role']['terminal'], 'Exact entire prior APUSH cold score replay')
    a,b = row['per_role']['shared'],row['per_role']['APUSH']
    comparison = dict(raw_ratio=a['raw_mse']/b['raw_mse'],complete_flops_ratio=a['flops']/b['flops'],
        adjusted_ratio=a['adjusted']/b['adjusted'],adjusted_improved=a['adjusted'] < b['adjusted'])
    promising = comparison['adjusted_ratio'] <= .95
    for path,digest in {**admit.pins,**result['artifacts']}.items():
        require(sha(path,tick) == digest, 'Fresh final complete non-target union')
    require(sha(TRUTH,tick,truth=True) == TRUTH_SHA, 'Fixed truth unchanged after scoring')
    require(not {'torch','flopscope','threadpoolctl'} & set(sys.modules), 'NumPy-only saved-output score, no model imports')
    result.update(passed=True,completed=True,integrity_passed=True,hashes_unchanged=True,sources_unchanged=True,
        record=row,comparison=comparison,information_promising=promising,scientific_gate_passed=promising,
        complete_probe_acquisition_reverified=True,all_four_outputs_verified=True,
        all16_domains_and_all47_selectors_reverified=True,exact_APUSH_score_replay=True,own_complete_bills=True,
        repeated_outputs_byte_exact=True,no_model_imported=True,metadata_fresh_union_rehashed=True,
        aggregation='One cold rank37 final raw*max(.1,own completeF/2^41); repeat excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in KEYS), 'Explicit reviewed sources/results')
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
        and not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules)
        and importlib.metadata.version('numpy') == '2.4.6', 'Original OFFPY nice10/sixBLAS1/no pre-admission numerical import')
    out = HERE/'score_probe_v1'; out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,passed=False,completed=False,integrity_passed=False,source_bindings={},input_bindings={},
        artifacts={},truth_bindings={},truth_accessed=False,ranks=[37],repeat_excluded=True,
        full_model_calls=0,parameter_updates=0,estimator_executed=False,training_or_refit_performed=False,
        native_gradient_qualified=False,production_qualified=False,submission_authorized=False,uploaded=False,
        automatic_next_stage=False,information_promising=False,requested_seals=vars(args))
    def guard(*unused):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        require(time.monotonic()-began < CAP and rss < RSS, 'Whole120s/1GB offline scorer')
        return rss
    write(out/'started.json',result)
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        score(args,out,result,guard); result['peak_rss_bytes'] = guard()
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,error=traceback.format_exc())
    signal.setitimer(signal.ITIMER_REAL,0)  # False-only publication remains possible after a caught resource failure.
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    result['passed'] = result['completed'] = bool(result['passed'] and result['elapsed_s'] < CAP and result['peak_rss_bytes'] < RSS)
    write(out/'result.json',result); rh = sha(out/'result.json')
    ok = bool(result['passed'] and time.monotonic()-began < CAP)
    write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(out/'completion.json')
    elapsed = time.monotonic()-began; ok = bool(ok and elapsed < CAP)
    write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    sha(out/'finalization.json'); ok = bool(ok and time.monotonic()-began < CAP)
    print(json.dumps({k:result.get(k) for k in ('passed','information_promising','record','comparison','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
