"""Saved SHARED production5/native-panel fidelity; no runtime, fit or release."""
import argparse
import hashlib
import importlib.util
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
NEW,P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
SCHEMA = 'current_comp_shared_old384_fidelity_v1'
CAP,RSS,BUDGET,INIT = 120.,1_000_000_000,2**41,1944970298
CANDIDATE = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
RANKS = (35,100,500,3016500,35)
MAPPING = ((0,0,35),(1,1,100),(2,7,500),(3,8,3016500),(4,14,35))
KEYS = ('source','protocol','watcher','watch-spec','panel-source','panel-protocol',
        'panel-watch-source','panel-watch-spec','panel-score','panel-score-watch','native','native-watch','runtime')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
OWNER = NEW/'birth_direction_fit_v1/release_confirmation_v1/check_release.py'
OWNER_SHA = '3aac716546920c844d60e6d51fd8f5673c3ae055b2e10fae0456709fe69d68ee'
RUNTIME_SEALS = ('c8d325e35229396bf67f63fd65e92f09f75ed8305ef1e169388d975507318af0',
    'b37e54981a24af7433fd65b8ca90b38fb84d092a910f91c7655542b658169f6d',
    '40b6bde559cbee4960cd8a5202bd12c08e241cfa3d5dcea38d460b43a67bcc48')
RUNTIME_SOURCES = {
    HERE/'runtime_watch.py':'1c4383e63b25337ddf1e051008174430aeee189ae1c5d9babe99e5c2b4ce5674',
    HERE/'runner.py':'c6f684c7fcd7bb51d5ef52240b777da3787f466d4522b61d01ee877dd7e34af3',
    HERE/'client.py':'cee32e84ca797e5708aced91d3d0c76f5da2f8c436d65d340790782cdf50deae',
    HERE/'RUNTIME_PROTOCOL.md':'d8df84bbd0cc5817ae61ce150263114b6284c817b0363675d9cb65f67d3d1de3',
}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def load_panel(path,digest):
    require(Path(path) == HERE/'score_panel.py' and re.fullmatch('[0-9a-f]{64}',digest or '')
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Explicit actual panel metadata helper owner')
    spec = importlib.util.spec_from_file_location('shared_fidelity_panel_metadata',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Only inert stdlib panel metadata')
    return module


def source_map(args):
    return {OWNER:OWNER_SHA,**RUNTIME_SOURCES,HERE/'fidelity.py':args.source_sha256,
        HERE/'watch_fidelity.py':args.watcher_sha256,HERE/'FIDELITY_PROTOCOL.md':args.protocol_sha256,
        HERE/'FIDELITY_WATCH_SPEC.md':args.watch_spec_sha256,HERE/'score_panel.py':args.panel_source_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.panel_protocol_sha256,HERE/'watch_score_panel.py':args.panel_watch_source_sha256,
        HERE/'SCORE_PANEL_WATCH_SPEC.md':args.panel_watch_spec_sha256}


def arithmetic_source(h):
    require(h.sha(OWNER) == OWNER_SHA, 'Original release fidelity scalar authority')
    source = h.extract(OWNER,'qualify')
    start = source.index("        result['per_net'] = []\n")
    stop = source.index("        unchanged(pins); unchanged(result['artifacts']); unchanged(result['truth_bindings'])",start)
    original = textwrap.dedent(source[start:stop])
    before,after = "panel['per_role']['candidate']", "panel['per_role']['shared']"
    require(original.count(before) == 1, 'Unique current native-panel score label')
    changed = original.replace(before,after)
    require(changed.replace(after,before) == original, 'Whole original observed-MSE/noise/fidelity body restored')
    return changed,dict(owner_sha256=OWNER_SHA,whole_numeric_body_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),original_raw_ratio_tolerance=.01,
        vector_tolerance=1e-5,unclipped_noise_correction=True,own_remote_cost_used=True)


def execute(args,out,result,tick,h):
    env,helper_proof = h.support(); admit = env['Admission'](result,tick)
    admit.bind({str(p):v for p,v in source_map(args).items()})
    call_helpers,call_proof = env['helpers'](admit); env.update(call_helpers)
    panel,ps = h.actual_chain(admit,HERE/'score_panel_v1',args.panel_score_sha256,120,2_000_000_000)
    pw,pws = h.actual_chain(admit,HERE/'score_panel_watch_v1',args.panel_score_watch_sha256,120,2_000_000_000,watched=True)
    env['linked'](pw,ps)
    h.watch_contract(None,panel,HERE/'score_panel_v1',h.sha,tick)
    require(panel['bank_scientific_gate_passed'] and panel['both_panels_three_percent_gain']
        and panel['combined_noise_corrected_adjusted_below_4e9'] and pw['information_promising'],
        'Actual prospective independent14 scientific gate before fidelity stage')
    expected_panel = {HERE/'score_panel.py':args.panel_source_sha256,HERE/'watch_score_panel.py':args.panel_watch_source_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.panel_protocol_sha256,HERE/'SCORE_PANEL_WATCH_SPEC.md':args.panel_watch_spec_sha256}
    require(all(panel['source_bindings'].get(str(p)) == pw['source_bindings'].get(str(p)) == digest
        for p,digest in expected_panel.items()), 'Exact reviewed completed panel scoring sources')
    native,ns = h.actual_chain(admit,HERE/'native_panel_v1',args.native_sha256,1200,8_000_000_000)
    nw,nws = h.actual_chain(admit,HERE/'native_panel_watch_v1',args.native_watch_sha256,1200,8_000_000_000,watched=True)
    env['linked'](nw,ns)
    require(panel['lineage']['native'] == ns and panel['lineage']['native_watch'] == nws
        and native['candidate_sha256'] == nw['candidate_sha256'] == panel['candidate_sha256'] == CANDIDATE
        and native['ranks'] == nw['ranks'] == list(h.SEQUENCE)
        and native['full_model_calls'] == native['retired_call_count'] == nw['retired_calls'] == 15
        and native['initialization_flops'] == INIT and native['fixture_bindings'] == panel['fixture_bindings'],
        'Same complete15 native packet already scored, no substitute output or cost')
    for record in (panel,pw,native,nw):
        admit.inventory(record)
    h.verify_calls(native,HERE/'native_panel_v1',admit,env,shared=True,watch=nw)
    require(args.runtime_sha256 == RUNTIME_SEALS[0], 'Exactly the already-executed production5; no retry or replacement')
    runtime,rs = h.actual_chain(admit,HERE/'runtime_v1',args.runtime_sha256,600,8_000_000_000,watched=True)
    require(tuple(rs.values()) == RUNTIME_SEALS and runtime['schema'] == 'current_comp_shared_old384_runtime_v1'
        and runtime['candidate_sha256'] == CANDIDATE and runtime['ranks'] == list(RANKS)
        and runtime['full_model_calls'] == 5 and len(runtime['observed_records']) == 5
        and runtime['initialization_flops'] == INIT and runtime['numerical_imports'] is False
        and runtime['truth_arrays_accessed'] is False and runtime['native_fidelity_pending']
        and runtime['native_fidelity_qualified'] is False and runtime['production_five_qualified'] is False
        and all(runtime[k] is True for k in ('all5_production_timing_passed','all16_outputs_finite',
            'repeat_byte_exact','one_persistent_numpy_free_client_and_server','own_complete_bills'))
        and all(runtime['source_bindings'].get(str(p)) == digest for p,digest in RUNTIME_SOURCES.items()),
        'Actual positive NumPy-free production5 timing scope, old pending fidelity is not relabelled')
    admit.inventory(runtime)
    require(native['production_runtime_seals'] == rs, 'Native acquisition admitted exactly the same production packet')
    series_path = NEW/(runtime['tag']+'.json')
    series = h.read(series_path)
    require(series['status'] == 'completed' and series['source_sha256'] == CANDIDATE
        and series['records'] == runtime['observed_records'] and series['repeated_output_byte_identical'],
        'Actual persistent client series and externally checked complete bills')
    all_paths = []
    for prod,call,rank in MAPPING:
        row = series['records'][prod]; t = native['trials'][call]
        fixture = native['fixture_bindings'][f'weights_{rank}']
        require(runtime['fixture_bindings'][str(rank)] == dict(path=fixture['path'],sha256=fixture['sha256'])
            and native['input_preparation'][call]['weight_file_sha256'] == fixture['sha256']
            and row['index'] == prod and row['rank'] == t['rank'] == rank
            and row['output_dtype'] == 'float32' and row['output_shape'] == [16,1024]
            and type(row['flops']) is int and 0 < row['flops'] <= BUDGET and row['util'] == row['flops']/BUDGET,
            'All five native/production links use exact physical weights and own complete costs')
        timing = row['timing']
        require(all(isinstance(v,(int,float)) and math.isfinite(v) and v >= 0 for v in timing.values())
            and 0 < timing['wall_time_s'] < 60 and timing['residual_wall_time_s'] < .4,
            'Each actually measured production call retains strict60s/.4s timing')
        path = NEW/f"{runtime['tag']}_{prod}_r{rank}.f32"
        require(path.stat().st_size == 16*1024*4 and runtime['artifacts'][str(path)] == row['output_sha256'],
            'All five raw payloads already authenticated in the positive production inventory')
        admit.pin(path,row['output_sha256']); all_paths.append(path)
    require(series['records'][0]['flops']-series['records'][-1]['flops'] == INIT
        and native['trials'][0]['flops']-native['trials'][-1]['flops'] == INIT,
        'Actual full initialization separately owned once in each backend; no cost transplantation')
    source,numeric_proof = arithmetic_source(h)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'All native/production arrays and bills authenticated before NumPy')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Original offline NumPy fidelity arithmetic')
    native_all = [env['trial_output'](np,HERE/'native_panel_v1',t,admit.pins) for t in native['trials']]
    require(native_all[0].tobytes() == native_all[-1].tobytes(), 'Native own repeat bytes')
    remote_arrays,native_arrays,checks = {},{},[]
    for prod,call,rank in MAPPING:
        row = series['records'][prod]; t = native['trials'][call]
        arr = np.frombuffer(all_paths[prod].read_bytes(),dtype=np.float32).reshape(16,1024)
        prior = native_all[call]
        require(np.isfinite(arr).all() and (arr >= 0).all(), 'All16 real production outputs finite/nonnegative')
        difference = float(np.max(np.abs(arr.astype(float)-prior.astype(float))))
        require(difference <= 1e-5, 'Original all16 vector native/client tolerance<=1e-5')
        if prod < 4:
            remote_arrays[rank],native_arrays[rank] = arr,prior
        else:
            require(arr.tobytes() == remote_arrays[35].tobytes(), 'Production own repeat35 exact')
        checks.append(dict(call=prod,native_call=call,rank=rank,weights_sha256=runtime['fixture_bindings'][str(rank)]['sha256'],
            production_output_sha256=row['output_sha256'],native_output_sha256=t['output_sha256'],native_max_abs=difference,
            production_flops=row['flops'],native_flops=t['flops'],complete_flops_delta=row['flops']-t['flops'],
            production_multiplier=max(.1,row['flops']/BUDGET),native_multiplier=max(.1,t['flops']/BUDGET),
            wall_s=row['timing']['wall_time_s'],residual_s=row['timing']['residual_wall_time_s'],
            cross_backend_cost_equality_required=False,cross_backend_operation_count_equality_required=False))
    lineage = dict(panel_score=ps,panel_score_watch=pws,native=ns,native_watch=nws,runtime=rs)
    seal = out/'outputs_and_bills_sealed_before_truth.json'
    h.write(seal,dict(truth_accessed=False,all20_outputs_verified=True,all15_native_artifacts_verified=True,
        all5_production_outputs_and_bills_verified=True,own_complete_bills=True,checks=checks,lineage=lineage,
        source_bindings=result['source_bindings'],input_bindings=dict(admit.pins)))
    result['artifacts'][str(seal)] = h.sha(seal,tick)
    result.update(outputs_and_bills_sealed_before_truth_sha256=h.sha(seal,tick),all_predictions_verified_before_truth=True,
        checks=checks,lineage=lineage,helper_source_proof=helper_proof,saved_call_helper_proof=call_proof,arithmetic_source_proof=numeric_proof)
    def pin_truth(mapping,path,expected=None):
        rank = next((r for r in RANKS[:4] if h.truth_path(r) == Path(path)),None)
        require(rank is not None and panel['truth_bindings'][str(rank)]['sha256'] == expected,
            'Only the four fixed original independent4M truths after complete all20 seal')
        require(h.sha(path,tick,truth=True) == expected, 'Original four truth bytes')
        mapping[str(path)] = expected
        tick()
    scope = dict(np=np,RANKS=RANKS,panel=panel,result=result,series=series,remote_arrays=remote_arrays,
        native_arrays=native_arrays,pin=pin_truth,BUDGET=BUDGET,math=math)
    exec(compile(source,'<exact-original-release-raw-fidelity-body>','exec'),scope)
    for path,digest in {**admit.pins,**result['artifacts']}.items():
        require(h.sha(path,tick) == digest, 'Fresh final complete non-target union')
    for path,digest in result['truth_bindings'].items():
        require(h.sha(path,tick,truth=True) == digest, 'Original four truth bytes unchanged')
    require(not {'torch','flopscope','threadpoolctl'} & set(sys.modules), 'NumPy-only saved fidelity; no participant/model import')
    result.update(passed=True,completed=True,integrity_passed=True,hashes_unchanged=True,sources_unchanged=True,
        all_five_output_vector_tolerances_passed=True,all_four_unique_raw_mse_ratios_passed=True,
        exact_native_observed_score_replay=True,exact_fixture_compatibility=True,native_and_runtime_repeat_exact=True,
        initialization_flops=INIT,own_complete_bills=True,runtime_timing_qualified=True,production_fidelity_qualified=True,
        production_five_qualified=True,all20_outputs_verified=True,complete_native_acquisition_reverified=True,
        no_model_imported=True,metadata_fresh_union_rehashed=True,information_promising=False,
        local_panel_nc_adjusted=panel['summaries']['shared']['combined14']['mean_adjusted_noise_corrected'],
        four_net_remote_diagnostic_mean_adjusted=sum(r['adjusted_noise_corrected'] for r in result['per_net'])/4,
        cross_backend_costs_reported_not_equated=True,original_runtime_receipt_remains_timing_only=True,
        scope='Local persistent NumPy-free client/OpenBLAS fidelity; not official grading or release authority')


def watch_setup(args):
    child = {k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256') for k in KEYS if k not in ('source','watcher')}
    from types import SimpleNamespace
    child.update(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256)
    own = source_map(SimpleNamespace(**child))
    inputs = {HERE/'score_panel_v1/result.json':args.panel_score_sha256,
        HERE/'score_panel_watch_v1/result.json':args.panel_score_watch_sha256,
        HERE/'native_panel_v1/result.json':args.native_sha256,
        HERE/'native_panel_watch_v1/result.json':args.native_watch_sha256,
        HERE/'runtime_v1/result.json':args.runtime_sha256}
    return own,inputs,HERE/'fidelity_v1',HERE/'fidelity_watch_v1'


def watch_contract(args,r,folder,check_sha,tick):
    flags = ('integrity_passed','hashes_unchanged','sources_unchanged','all_five_output_vector_tolerances_passed',
        'all_four_unique_raw_mse_ratios_passed','exact_native_observed_score_replay','exact_fixture_compatibility',
        'native_and_runtime_repeat_exact','own_complete_bills','runtime_timing_qualified','production_fidelity_qualified',
        'production_five_qualified','all20_outputs_verified','all_predictions_verified_before_truth',
        'complete_native_acquisition_reverified','no_model_imported','metadata_fresh_union_rehashed')
    require(all(r.get(k) is True for k in flags) and r['schema'] == SCHEMA and r['candidate_sha256'] == CANDIDATE
        and r['ranks'] == list(RANKS) and r['full_model_calls'] == r['parameter_updates'] == 0
        and r['estimator_executed'] is False and r['submission_authorized'] is False and r['automatic_next_stage'] is False
        and r['information_promising'] is False and r['initialization_flops'] == INIT,
        'Exactly saved runtime/native fidelity, no new candidate or numerical action')
    require(len(r['checks']) == 5 and len(r['per_net']) == 4
        and [(c['call'],c['native_call'],c['rank']) for c in r['checks']] == list(MAPPING)
        and all(c['native_max_abs'] <= 1e-5 and 0 < c['wall_s'] < 60 and c['residual_s'] < .4 for c in r['checks'])
        and [p['rank'] for p in r['per_net']] == list(RANKS[:4])
        and all(abs(p['remote_native_mse_ratio']-1) <= .01 for p in r['per_net']), 'Both original per-net fidelity gates')
    h = load_panel(HERE/'score_panel.py',args.panel_source_sha256)
    panel_path = HERE/'score_panel_v1/result.json'
    require(check_sha(panel_path,tick) == args.panel_score_sha256, 'Exact actual native-panel score')
    panel = h.read(panel_path)
    expected = {panel['truth_bindings'][str(rank)]['path']:panel['truth_bindings'][str(rank)]['sha256'] for rank in RANKS[:4]}
    require(r['truth_bindings'] == expected and r['truth_accessed'] is True, 'Only the original four fixed truths')
    seal = folder/'outputs_and_bills_sealed_before_truth.json'; s = h.read(seal)
    require(check_sha(seal,tick) == r['outputs_and_bills_sealed_before_truth_sha256'] == r['artifacts'][str(seal)]
        and s['truth_accessed'] is False and s['all20_outputs_verified'] and s['all5_production_outputs_and_bills_verified']
        and s['checks'] == r['checks'] and s['lineage'] == r['lineage']
        and s['source_bindings'] == r['source_bindings'] and s['input_bindings'] == r['input_bindings'],
        'Actual all-output/bill pretruth seal remains immutable')


def truth_sha(path,tick=None):
    # Separate strict four-file byte hasher; supervisor never imports NumPy or decodes targets.
    paths = {P2/'public_g19_attribution_v1'/(f'mps_truth_rank{r:04d}_indep4M.npz'
        if r < 1000 else f'fresh_{r}_truth4M.npz') for r in RANKS[:4]}
    path = Path(path)
    require(path in paths and path.is_file() and not path.is_symlink(), 'Only four explicit original fidelity truth files')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick is not None:
                tick()
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in KEYS), 'Actual reviewed fidelity inputs')
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.dont_write_bytecode and sys.flags.optimize == 0
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
        and not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Exclusive original nice10/OFFPY/sixBLAS1')
    h = load_panel(HERE/'score_panel.py',args.panel_source_sha256)
    out = HERE/'fidelity_v1'; out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,passed=False,completed=False,integrity_passed=False,source_bindings={},input_bindings={},
        artifacts={},truth_bindings={},truth_accessed=False,ranks=list(RANKS),candidate_sha256=CANDIDATE,
        full_model_calls=0,parameter_updates=0,estimator_executed=False,training_or_refit_performed=False,
        production_fidelity_qualified=False,submission_authorized=False,uploaded=False,automatic_next_stage=False,
        information_promising=False,requested_seals=vars(args))
    def guard(*unused):
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        require(time.monotonic()-began < CAP and value < RSS, 'Whole120s/1GB saved fidelity')
        return value
    h.write(out/'started.json',result)
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        execute(args,out,result,guard,h); guard()
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,error=traceback.format_exc())
    signal.setitimer(signal.ITIMER_REAL,0)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    ok = bool(result['passed'] and result['elapsed_s'] < CAP and result['peak_rss_bytes'] < RSS)
    result['passed'] = result['completed'] = ok
    h.write(out/'result.json',result); rh = h.sha(out/'result.json'); ok = bool(ok and time.monotonic()-began < CAP)
    h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = h.sha(out/'completion.json')
    elapsed = time.monotonic()-began; ok = bool(ok and elapsed < CAP)
    h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    h.sha(out/'finalization.json'); ok = bool(ok and time.monotonic()-began < CAP)
    print(json.dumps({k:result.get(k) for k in ('passed','production_fidelity_qualified','checks','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
