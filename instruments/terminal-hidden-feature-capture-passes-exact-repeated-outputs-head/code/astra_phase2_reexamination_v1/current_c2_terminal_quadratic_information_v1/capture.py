"""Inert current-c2 terminal capture: one cold35 and its exact35 repeat.

No targets, fit, quadratic evaluation, candidate change, or automatic expansion.
The original participant loop and current observer owners remain authoritative.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SHARED = NEW/'current_comp_shared_old384_v1'
GAIN = NEW/'current_comp_shared_gain_slice_v1'
TL = NEW/'current_d50_terminal_linear_v2'
OFFPY = P2/'data_gate/.venv/bin/python'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ROW = STORE/'rows/observer_fit/0035'
CANDIDATE = GAIN/'build_v1/estimator.py'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
OUTPUT = HERE/'capture_qualification_v1'
SCHEMA = 'current_c2_terminal_quadratic_capture_qualification_v1'
CAP, RSS, BUDGET, INIT = 300., 8_000_000_000, 2**41, 1944970298
REPLAY_CAP = 10_000_000
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
CHECK = (37,39,42,48)
PINS = {
    CANDIDATE:CANDIDATE_SHA,
    SHARED/'probe.py':'6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    SHARED/'backend_probe.py':'36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8',
    SHARED/'check4_v2.py':'8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
    TL/'native.py':'744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    NEW/'birth_direction_fit_v1/native_v2.py':'8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'current_comp_aligned_push_d6_v1/native.py':'751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6',
}
CONTROL = {
    'native_panel_v1':('9563919f22a938ec084751444cf1c4e643e18fd4aba1cf8b87c3526f2d3f6608',
        'fe75d8c7943c14a58a66025f96ecd4e97b8f89d44d12759a0d1ff2233f8bbf4c',
        '3578b8eb6a9c9a1c183d9c87c3dd32c3a9fa322bfa70209ecc3a6c0f4aeeb36e'),
    'native_panel_watch_v1':('6fe4acd894a5dd13f0bdb558ef48c522622917413b400a0b1ca43850236474fa',
        '6d58d72298d7cda2049b995b3faf4c0bd090394a03b3b61d3cda35ca6a81c040',
        'e9bf477c760ca54805e6610407d0b5d2133035679ef989cb5e35ca40a71ea088')}
FIXTURE = {
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    ROW/'row_manifest.json':'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    ROW/'weights.npy':'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npz','.npy') and 'truth' in path.name.lower()), 'No target path access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x):
            return dict(nonfinite_float=repr(x))
        if isinstance(x,dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value),stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def bind(mapping, tick=None):
    for path,digest in mapping.items():
        require(sha(path,tick) == digest, 'Immutable source/input/artifact: '+str(path))


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source before inert import')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_segment(source, name, kind=ast.FunctionDef):
    nodes = [n for n in ast.parse(source).body if isinstance(n,kind) and n.name == name]
    require(len(nodes) == 1, 'Unique source owner: '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def providers():
    h = load(SHARED/'probe.py','h15_original_paid_owner',PINS[SHARED/'probe.py'])
    backend = load(SHARED/'backend_probe.py','h15_original_backend',PINS[SHARED/'backend_probe.py'])
    owner = h.load(h.OBSERVER,'h15_current_comp_observer',h.OBSERVER_SHA)
    legacy = h.load(h.LEGACY,'h15_original_lifecycle',h.LEGACY_SHA)
    return h,backend,owner,legacy


def admission(args, tick=None):
    """Stdlib-only direct positive current-c2 boundary; no ancestor re-admission."""
    began, outer = time.monotonic(),tick
    def guard():
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000,
                'Metadata admission60s/1GB')
        if outer is not None:
            outer()
    own = {str(HERE/'capture.py'):args.source_sha256,str(HERE/'watch_capture.py'):args.watcher_sha256,
        str(HERE/'PROTOCOL.md'):args.protocol_sha256,str(HERE/'WATCH_SPEC.md'):args.watch_spec_sha256,
        **{str(p):d for p,d in PINS.items()}}
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in own.values()), 'Explicit reviewed four new sources')
    bind(own,guard)
    h,backend,owner,legacy = providers()
    own.update({str(p):d for p,d in {**backend._BP_PINS,**owner.OWNERS,**legacy.PINS}.items()})
    bind(own,guard)
    inputs = {str(p):d for p,d in FIXTURE.items()}
    bind(inputs,guard)
    records = {}
    for folder,expected in CONTROL.items():
        paths = [GAIN/folder/n for n in ('result.json','completion.json','finalization.json')]
        ds = tuple(sha(p,guard) for p in paths)
        require(ds == expected, 'Actual completed current-c2 three-seal boundary')
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1],
            'No failed watch is relabeled or re-admitted')
        inputs.update({str(p):d for p,d in zip(paths,ds)}); records[folder] = r
    r,w = records['native_panel_v1'],records['native_panel_watch_v1']
    require(r['schema'] == 'current_comp_shared_gain_slice_native_panel_v1'
        and w['schema'] == 'current_comp_shared_gain_slice_native_panel_watch_v1'
        and r['candidate_sha256'] == w['candidate_sha256'] == CANDIDATE_SHA
        and [w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]
            == list(CONTROL['native_panel_v1'])
        and all(w[k] for k in ('whole_owned_tree_retired','cleanup_passed','observational_groups_retired'))
        and w['maximum_combined_rss_bytes'] < RSS and w['maximum_interval_s'] <= 1
        and w['maximum_swap_growth_bytes'] == 0, 'Positive same-backend completed reference')
    selected,diags = [],[]
    for ordinal,index in enumerate((0,14)):
        trial = r['trials'][index]
        require(trial['call'] == index and trial['rank'] == 35
            and trial['flops'] == (403400296797,401455326499)[ordinal]
            and trial['includes_initialization'] is (ordinal == 0), 'Fixed current35 cold/repeat comparator')
        prefix = GAIN/'native_panel_v1'/f'call_{index:02d}'
        paths = [trial['output'],str(prefix)+'_bill.json',trial['input_bill'],trial['input_reset_receipt'],
                 trial['lifecycle_receipt'],str(prefix)+'_diagnostics.json',str(prefix)+'_complete.json']
        for p in paths:
            require(sha(p,guard) == r['artifacts'][p], 'Selected current35 artifact')
            inputs[p] = r['artifacts'][p]
        require(read(trial['input_bill'])['weight_file_sha256'] == FIXTURE[ROW/'weights.npy'],
                'Canonical physical35 weights')
        selected.append(trial); diags.append(read(paths[-2]))
    require(selected[0]['output_bytes_sha256'] == selected[1]['output_bytes_sha256'], 'Actual own current repeat')
    row = read(ROW/'row_manifest.json')
    require(row['files']['weights.npy']['dtype'] == '<f4'
        and row['files']['weights.npy']['shape'] == [16,1024,1024], 'Exact physical fixture metadata')
    prepared = prepare(h,owner,legacy)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Admission stays stdlib-only')
    return dict(source_bindings=own,input_bindings=inputs,controls=selected,control_diagnostics=diags,
        control_seals={k:list(v) for k,v in CONTROL.items()},candidate_sha256=CANDIDATE_SHA,
        source_proof=prepared['proof'],mlp_seed=row['mlp_seed'],truth_accessed=False)


def prepare(h, owner, legacy):
    """Exact old paid loop, exact current observer, exact TL passive byte sink."""
    Parent,_,observer_proof = h.observers(owner)
    text = (TL/'native.py').read_text()
    sink_source = source_segment(text,'Observer',ast.ClassDef)
    diagnostic_source = source_segment(text,'diagnostics')
    ns = dict(c=types.SimpleNamespace(require=require),hashlib=hashlib)
    exec(compile(sink_source,'<exact-TL2-passive-terminal-sink>','exec'),ns)
    Sink = ns['Observer']
    class Observer(Parent):
        def begin(self):
            super().begin()
            self.terminal = Sink(self.module)
        def __call__(self,frame,event,answer):
            super().__call__(frame,event,answer)
            if event in ('call','return') and frame.f_code in (self.terminal.gnn_code,self.terminal.predict_code):
                self.terminal(frame,event,answer)
        def finish(self):
            packet,startup,dispatches = super().finish()
            self.terminal.variances.extend(packet['variances'])
            self.terminal.selectors.extend(packet['selectors'])
            captured = self.terminal.export_bytes()
            return captured,startup,dispatches
    loop,loop_proof = h.paid_loop_source()
    ledger_source = source_segment((SHARED/'check4_v2.py').read_text(),'_v2_ledger_metrics')
    ln = dict(time=time,hashlib=hashlib,json=json,require=require)
    exec(compile(ledger_source,'<exact-closed-operation-ledger>','exec'),ln)
    candidate = CANDIDATE.read_text()
    node = ast.parse(source_segment(candidate,'gnn_step'))
    expressions = [n.value for n in node.body[0].body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'delta' for t in n.targets)]
    require(len(expressions) == 1, 'One exact current head expression')
    expression = ast.fix_missing_locations(ast.Expression(expressions[0]))
    require('l:l + 1' in ast.unparse(expression), 'Keep current typed one-element gain slice')
    proof = dict(candidate_sha256=CANDIDATE_SHA,estimator_source_unchanged=True,
        original_paid_loop=loop_proof,current_observer=observer_proof,
        terminal_sink_owner_sha256=PINS[TL/'native.py'],terminal_sink_sha256=hashlib.sha256(sink_source.encode()).hexdigest(),
        diagnostic_body_sha256=hashlib.sha256(diagnostic_source.encode()).hexdigest(),
        head_expression_ast_sha256=hashlib.sha256(ast.dump(expression).encode()).hexdigest(),
        ledger_body_sha256=hashlib.sha256(ledger_source.encode()).hexdigest(),
        no_added_participant_array_operations=True,passive_return_bytes_only=True,
        old_d50_parameters_not_reused=True,original_gain_slice_and_grouping_retained=True)
    return dict(Observer=Observer,loop=loop,metrics=ln['_v2_ledger_metrics'](legacy.metrics),
        diagnostic_source=diagnostic_source,expression=compile(expression,str(CANDIDATE)+':head-replay','eval'),proof=proof)


def captured_arrays(np, packet, trial):
    """Representation only: exact detached bytes, no floating arithmetic."""
    expected = dict(hidden=((1024,64),'float32'),sigma=((1024,),'float64'),
        host=((1024,),'float64'),delta=((1024,),'float32'),hook_delta=((1024,),'float64'))
    arrays = {}
    for key,(shape,dtype) in expected.items():
        item = packet['fields'][key]
        require(tuple(item['shape']) == shape and item['dtype'] == dtype, 'Actual captured header: '+key)
        arrays[key] = np.frombuffer(item['data'],dtype=dtype).reshape(shape)
    require(len(packet['raw']) == 16*1024*8, 'Sixteen exact raw vectors')
    arrays['raw'] = np.frombuffer(packet['raw'],dtype=np.float64).reshape(16,1024)
    arrays['served'] = np.load(trial['output'],allow_pickle=False)
    require(arrays['served'].shape == (16,1024) and arrays['served'].dtype == np.float32,
            'Original materialized served output')
    for key,shape in dict(head=(16,64),head_b=(16,),gain=(16,),rms=(16,)).items():
        item = packet['parameters'][key]
        require(tuple(item['shape']) == shape and item['dtype'] == 'float32', 'Current whole head parameter '+key)
        arrays['P_'+key] = np.frombuffer(item['data'],dtype=np.float32).reshape(shape)
    return arrays


def replay(np, flops, fnp, module, packet, trial, prepared, legacy, result, tick):
    """Paid diagnostics AFTER original model bill/reset; no participant changes."""
    arrays = captured_arrays(np,packet,trial)
    owned = []
    def take(value, dtype=None):
        value = fnp.array(value,dtype=dtype)
        owned.append(weakref.ref(value))
        return value
    def decode(_unused, item):
        view = np.frombuffer(item['data'],dtype=item['dtype']).reshape(item['shape'])
        return take(view)
    dn = dict(c=types.SimpleNamespace(require=require),hashlib=hashlib,decode=decode)
    exec(compile(prepared['diagnostic_source'],'<exact-domain-body-paid-decode>','exec'),dn)
    def evaluate():
        data = {k:take(v) for k,v in arrays.items()}
        for value in data.values():
            require(bool(fnp.all(fnp.isfinite(value))), 'Finite actual capture: no unmetered finite scan')
        parameters = {k:data['P_'+k] for k in ('head','head_b','gain','rms')}
        for key,value in parameters.items():
            literal = take(module._GNN_LITERAL[key],fnp.float32)
            require(value.tobytes() == literal.tobytes() == module._GNN_PARAMS[key].tobytes(),
                    'Actual current source/literal/cache/head bytes, not an old payload')
        # Preserve TL2's aligned-owned-copy repair, exact current AST grouping,
        # FP32 intermediate arithmetic, and the current gain[l:l+1] array.
        state = data['hidden'].copy(); owned.append(weakref.ref(state))
        delta = eval(prepared['expression'],{'fnp':fnp,'__builtins__':{}},
                     dict(l=15,state=state,sigma=data['sigma'],P=parameters))
        widened = delta.astype(fnp.float64)
        raw_final = data['host']+widened
        served_final = fnp.maximum(raw_final,0.0).astype(fnp.float32)
        served_all = fnp.maximum(data['raw'],0.0).astype(fnp.float32)
        for value in (delta,widened,raw_final,served_final,served_all):
            owned.append(weakref.ref(value))
        checks = dict(current_head_parameters_byte_exact=True,
            delta_byte_exact=delta.tobytes() == arrays['delta'].tobytes(),
            hook_widen_byte_exact=widened.tobytes() == arrays['hook_delta'].tobytes(),
            final_raw_byte_exact=raw_final.tobytes() == arrays['raw'][15].tobytes(),
            final_served_byte_exact=served_final.tobytes() == arrays['served'][15].tobytes(),
            all16_raw_clamp_byte_exact=served_all.tobytes() == arrays['served'].tobytes())
        require(all(checks.values()), 'Exact original current head/raw/served replay; no tolerance rescue')
        domain = dn['diagnostics'](fnp,packet)
        return checks,domain
    call = trial['call']; started = time.monotonic(); failure = None
    checks = domain = None
    with flops.BudgetContext(flop_budget=REPLAY_CAP,wall_time_limit_s=30,quiet=True) as ctx:
        try:
            checks,domain = evaluate()
        except BaseException:
            failure = traceback.format_exc()
    bill = dict(call=call,rank=35,**prepared['metrics'](ctx,started),
        charged_to_participant=False,diagnostic_only=True,flop_cap=REPLAY_CAP)
    bp = OUTPUT/f'call_{call:02d}_replay_bill.json'; write(bp,bill)
    result['artifacts'][str(bp)] = sha(bp)
    ref = weakref.ref(ctx)
    lifecycle = legacy.reset_saved_history(flops,ctx,bp)
    del ctx
    gc.collect()
    lifecycle.update(call=call,rank=35,complete=True,context_released=ref() is None,
        replay_array_owners_retired=all(x() is None for x in owned))
    lp = OUTPUT/f'call_{call:02d}_replay_lifecycle.json'; write(lp,lifecycle)
    result['artifacts'][str(lp)] = sha(lp)
    result['replay_bills'].append(dict(path=str(bp),sha256=sha(bp),flops=bill['flops'],
        lifecycle=str(lp),lifecycle_sha256=sha(lp)))
    require(failure is None, 'Exact paid head/domain replay failed: '+str(failure))
    require(0 < bill['flops'] <= REPLAY_CAP and bill['wall_s'] < 30
        and lifecycle['context_released'] and lifecycle['replay_array_owners_retired']
        and lifecycle['closed_context_bill_unchanged_by_reset'], 'Paid replay finite/cost/lifetime gates')
    # Serialization of already detached arrays is representation-only. No model
    # state, old tensor views, or covariance plane survives in the artifact.
    path = OUTPUT/f'call_{call:02d}_capture.npz'
    with path.open('xb') as stream:
        np.savez(stream,**arrays)
    result['artifacts'][str(path)] = sha(path)
    metadata = {k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=int(v.nbytes),
        raw_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
    manifest = dict(call=call,rank=35,candidate_sha256=CANDIDATE_SHA,path=str(path),sha256=sha(path),
        arrays=metadata,observed_layouts={k:{j:v for j,v in item.items() if j != 'data'}
            for k,item in packet['fields'].items()},head_replay=checks,
        expression_ast_sha256=prepared['proof']['head_expression_ast_sha256'],
        diagnostic_bill=str(bp),diagnostic_bill_sha256=sha(bp),truth_accessed=False)
    mp = OUTPUT/f'call_{call:02d}_capture.json'; write(mp,manifest)
    result['artifacts'][str(mp)] = sha(mp)
    result['captures'].append(dict(path=str(mp),sha256=sha(mp),rank=35,call=call))
    del arrays
    tick()
    return domain,checks


def verify_result(record, tick=None):
    """The supervisor calls this after all contexts and the worker are retired."""
    require(record['schema'] == SCHEMA and record['candidate_sha256'] == CANDIDATE_SHA
        and record['ranks'] == [35,35] and record['full_model_calls'] == record['retired_call_count'] == 2
        and record['initialization_flops'] == INIT and record['control_seals'] == {k:list(v) for k,v in CONTROL.items()}
        and all(record[k] is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'all16_domains_qualified','all47_selectors_qualified','repeat_byte_exact','capture_repeat_byte_exact',
            'exact_head_replay','current_control_output_byte_exact','current_control_complete_flops_exact',
            'own_complete_bills','all_parameter_and_estimator_owners_retired','actual_production_backend_qualified'))
        and all(record[k] is False for k in ('truth_accessed','score_measured','fit_performed',
            'training_expansion_authorized','automatic_next_stage','uploaded','submission_authorized'))
        and record['parameter_updates'] == 0, 'Only actual current-c2 passive qualification')
    require(len(record['captures']) == len(record['trials']) == len(record['replay_bills']) == 2
        and len(record['artifacts']) == 22, 'Two original seven-artifact calls plus four capture/replay artifacts per call')
    current = read(GAIN/'native_panel_v1/result.json')
    require(record['controls'] == [current['trials'][i] for i in (0,14)], 'Actual fixed direct control rows')
    a = record['backend_attestation']
    site = NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages'
    require(a['site'] == str(site) and a['actual_origins_qualified'] and a['production_blas_qualified']
        and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
        and a['whestbench_version'] == '0.16.0' and not a['scipy_loaded'] and not a['torch_loaded'],
        'Actual complete production-backend metadata, not a label-only claim')
    pools = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(pools) == 1 and pools[0]['internal_api'] == 'openblas' and pools[0]['num_threads'] == 1
        and pools[0]['version'] == '0.3.31.188.0' and pools[0]['threading_layer'] == 'pthreads'
        and Path(pools[0]['filepath']).resolve() == (site/'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
        'Exactly the pinned production BLAS pool')
    for item in a['modules'].values():
        require(record['source_bindings'].get(item['path']) == item['sha256'], 'Every actual imported source is in the final union')
    manifests = []
    for call,(trial,capture,replay_bill) in enumerate(zip(record['trials'],record['captures'],record['replay_bills'])):
        require(trial['call'] == capture['call'] == call and trial['rank'] == capture['rank'] == 35
            and trial['flops'] == (403400296797,401455326499)[call]
            and trial['includes_initialization'] is (call == 0) and trial['wall_s'] < 120
            and trial['operation_ledger']['total_flops'] == trial['flops'], 'Actual complete current-c2 bills')
        control = record['controls'][call]
        diag = read(record['diagnostic_records'][call]['path'])
        old_diag = read(GAIN/'native_panel_v1'/f"call_{control['call']:02d}_diagnostics.json")
        require(trial['output_bytes_sha256'] == control['output_bytes_sha256']
            and diag['domains'] == old_diag['domains'] and diag['dispatches'] == old_diag['dispatches']
            and all(trial['operation_ledger'][k] == control['operation_ledger'][k] for k in
                ('charged_records_sha256','non_unstack_records_sha256')),
            'Rechecked actual output/domain/dispatch and typed operator evidence')
        if call == 0:
            require(diag['startup'] == old_diag['startup'] and trial['operation_count'] == 63927
                and trial['operation_ledger']['all_records_sha256'] == control['operation_ledger']['all_records_sha256'],
                'Actual cold whole-ledger/startup evidence')
        life = read(trial['lifecycle_receipt']); ilife = read(trial['input_reset_receipt'])
        require(all(l['complete'] and l['context_released'] and l['closed_context_bill_unchanged_by_reset']
                    for l in (life,ilife)) and read(trial['input_bill'])['flops'] == 16777216,
                'Actual participant and independent input bill retirement')
        require(record['artifacts'][capture['path']] == capture['sha256'] == sha(capture['path'],tick),
                'Actual detached capture metadata')
        m = read(capture['path']); manifests.append(m)
        require(m['call'] == call and m['rank'] == 35 and m['candidate_sha256'] == CANDIDATE_SHA
            and all(m['head_replay'].values()) and set(m['arrays']) ==
                {'hidden','sigma','host','delta','hook_delta','raw','served','P_head','P_head_b','P_gain','P_rms'}
            and record['artifacts'][m['path']] == m['sha256'] == sha(m['path'],tick), 'Exact sealed replay and eleven raw arrays')
        require(m['arrays']['hidden']['shape'] == [1024,64] and m['arrays']['hidden']['dtype'] == 'float32'
            and m['arrays']['sigma']['shape'] == [1024] and m['arrays']['sigma']['dtype'] == 'float64',
            'Actual H15/sigma frame')
        for path,digest in ((replay_bill['path'],replay_bill['sha256']),
                            (replay_bill['lifecycle'],replay_bill['lifecycle_sha256'])):
            require(record['artifacts'][path] == digest == sha(path,tick), 'Complete separate replay cost and retirement')
        rb,life = read(replay_bill['path']),read(replay_bill['lifecycle'])
        require(rb['flops'] == replay_bill['flops'] and 0 < rb['flops'] <= REPLAY_CAP
            and rb['wall_s'] < 30 and rb['charged_to_participant'] is False
            and life['context_released'] and life['replay_array_owners_retired']
            and life['closed_context_bill_unchanged_by_reset'], 'No diagnostic cost is hidden or credited to the card')
    require(manifests[0]['arrays'] == manifests[1]['arrays'], 'Own repeated H15/raw/host/head bytes, not only output parity')
    require(record['research_flops'] == sum(x['flops'] for x in record['trials'])
        +sum(x['flops'] for x in record['input_preparation'])+sum(x['flops'] for x in record['replay_bills']),
        'Complete research accounting distinct from unchanged participant scoring bills')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in ('source','watcher','protocol','watch-spec'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(not OUTPUT.exists() and not HERE.is_symlink(), 'Exclusive first35 qualification; no resume')
    began = time.monotonic(); OUTPUT.mkdir(); args.out = OUTPUT
    result = dict(schema=SCHEMA,passed=False,completed=False,source_bindings={},input_bindings={},artifacts={},
        candidate_sha256=CANDIDATE_SHA,ranks=[35,35],trials=[],input_preparation=[],diagnostic_records=[],
        replay_bills=[],captures=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,
        truth_accessed=False,score_measured=False,fit_performed=False,training_expansion_authorized=False,
        automatic_next_stage=False,uploaded=False,submission_authorized=False)
    env = observer = module = None
    def tick():
        require(time.monotonic()-began < CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS,
                'Whole300s/process8GB; external supervisor owns completed samples and full tree')
    try:
        context = admission(args,tick)
        result.update({k:context[k] for k in ('source_bindings','input_bindings','controls','control_seals','source_proof')})
        h,backend,owner,legacy = providers(); prepared = prepare(h,owner,legacy)
        require(prepared['proof'] == context['source_proof'], 'Same admitted actual source owners')
        backend.backend_select(); h.environment()
        write(OUTPUT/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = types.ModuleType('current_c2_passive_h15')
        module.__file__ = str(CANDIDATE)
        exec(compile(CANDIDATE.read_text(),str(CANDIDATE),'exec'),vars(module))
        observer = prepared['Observer'](module)
        def finish_call(np,observer,call,rank,result,folder,trial):
            packet,startup,dispatches = observer.finish()
            require(startup['original_arena_bytes'] == 1168900096 and startup['no_second_backing_arena'],
                    'Original current-c2 arena, all current startup and D6 observers retained')
            domain,checks = replay(np,flops,fnp,module,packet,trial,prepared,legacy,result,tick)
            control = context['controls'][call]; reference = context['control_diagnostics'][call]
            require(trial['flops'] == control['flops']
                and trial['output_bytes_sha256'] == control['output_bytes_sha256']
                and domain == reference['domains'] and dispatches == reference['dispatches'],
                'Unchanged all16 bytes, complete own bill, all16/47 and actual physical dispatch')
            ledger,old = trial['operation_ledger'],control['operation_ledger']
            require(ledger['charged_records_sha256'] == old['charged_records_sha256']
                and ledger['non_unstack_records_sha256'] == old['non_unstack_records_sha256'],
                'Unchanged complete charged and non-unstack typed operator sequence')
            if call == 0:
                require(trial['operation_count'] == 63927 and ledger['all_records_sha256'] == old['all_records_sha256']
                    and startup == reference['startup'], 'Cold35 entire typed sequence and startup byte/metadata parity')
            record = dict(call=call,rank=rank,domains=domain,startup=startup,dispatches=dispatches,
                current_control_call=control['call'],current_output_byte_exact=True,current_complete_flops_exact=True,
                head_replay=checks,charged_and_non_unstack_sequence_exact=True,
                operation_count_delta=trial['operation_count']-control['operation_count'],
                operation_counts_reported_not_rebased=True)
            path = folder/f'call_{call:02d}_diagnostics.json'; write(path,record)
            result['artifacts'][str(path)] = sha(path)
            result['diagnostic_records'].append(dict(path=str(path),sha256=sha(path)))
            result['full_model_calls'] = call+1
            del packet
            tick()
        def retire_call(call,rank,result,folder,trial):
            life = read(trial['lifecycle_receipt'])
            require(life['context_released'] and life['closed_context_bill_unchanged_by_reset']
                and life['complete'] and observer.ctx is None and call == result['retired_call_count'],
                'Actual model/replay context retirement before progress')
            record = dict(passed=True,completed=True,call=call,rank=rank,context_retired=True,
                bill_sha256=sha(folder/f'call_{call:02d}_bill.json'),lifecycle_sha256=sha(trial['lifecycle_receipt']),
                diagnostic_sha256=result['diagnostic_records'][call]['sha256'],
                capture_manifest_sha256=result['captures'][call]['sha256'],
                replay_bill_sha256=result['replay_bills'][call]['sha256'],
                replay_lifecycle_sha256=result['replay_bills'][call]['lifecycle_sha256'],
                output_sha256=trial['output_sha256'],elapsed_s=time.monotonic()-began,
                persistent_pair_parameter_owners_retired=False,
                parameter_retirement_is_final_stage_gate=True)
            path = folder/f'call_{call:02d}_complete.json'; write(path,record)
            result['artifacts'][str(path)] = sha(path); result['retired_call_count'] += 1
            print(json.dumps(dict(event='retired',call=call,rank=rank,complete_sha256=sha(path))),flush=True)
            tick()
        rows = {35:dict(weights=str(ROW/'weights.npy'),mlp_seed=context['mlp_seed'])}
        env = dict(vars(legacy),np=np,flops=flops,fnp=fnp,threadpool_limits=threadpool_limits,
            threadpool_info=backend.backend_threadpool_info(threadpool_info,result,tick),MLP=MLP,
            derive_estimator_seed=derive_estimator_seed,materialise_predictions=materialise_predictions,
            args=args,inputs=result['input_bindings'],rows=rows,ranks=(35,),result=result,module=module,
            estimator=None,first_bytes=None,observer=observer,observed_predict=owner.base().observed_predict,
            finish_call=finish_call,retire_call=retire_call,metrics=prepared['metrics'])
        exec(compile(prepared['loop'],'<exact-current-c2-paid-loop-passive-H15>','exec'),env)
        require(result['full_model_calls'] == result['retired_call_count'] == 2, 'Only cold35/repeat35')
        manifests = [read(x['path']) for x in result['captures']]
        require(manifests[0]['arrays'] == manifests[1]['arrays'], 'All eleven captured arrays repeat exactly')
        result['initialization_flops'] = result['trials'][0]['flops']-result['trials'][1]['flops']
        require(result['initialization_flops'] == INIT, 'Original complete cold initialization is paid, never subtracted')
        backend.backend_origins(result,tick)
        array_refs = [weakref.ref(v) for v in module._GNN_PARAMS.values()]
        array_refs += [weakref.ref(module._BD_COEFFICIENTS),weakref.ref(module._BDNH_PARAMETERS),
                       weakref.ref(module._SCRATCH[0])]
        module_ref = weakref.ref(module); estimator_ref = weakref.ref(env['estimator'])
        env.clear(); env = None; observer = None; module = None
        gc.collect()
        require(module_ref() is None and estimator_ref() is None and all(x() is None for x in array_refs)
            and sys.getprofile() is None, 'All original parameter/arena/estimator owners finally retired')
        result['research_flops'] = sum(t['flops'] for t in result['trials'])+sum(t['flops'] for t in result['input_preparation'])\
            +sum(t['flops'] for t in result['replay_bills'])
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            all16_domains_qualified=True,all47_selectors_qualified=True,repeat_byte_exact=True,
            capture_repeat_byte_exact=True,exact_head_replay=True,current_control_output_byte_exact=True,
            current_control_complete_flops_exact=True,own_complete_bills=True,
            actual_production_backend_qualified=True,all_parameter_and_estimator_owners_retired=True,
            input_preparation_in_participant_bill=False,primary35_reusable_only_after_separate_expansion_selection=True)
        for group in ('source_bindings','input_bindings','artifacts'):
            bind(result[group],tick)
        verify_result(result,tick)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        if env is not None:
            env.clear()
        observer = module = env = None
        gc.collect()
    elapsed = time.monotonic()-began
    result.update(elapsed_s=elapsed,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if elapsed >= CAP or result['peak_rss_bytes'] >= RSS:
        result.update(passed=False,completed=False)
    write(OUTPUT/'result.json',result); rh = sha(OUTPUT/'result.json')
    ok = bool(result['passed'] and time.monotonic()-began < CAP)
    write(OUTPUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
    ch = sha(OUTPUT/'completion.json'); elapsed = time.monotonic()-began
    ok = bool(ok and elapsed < CAP)
    write(OUTPUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
        completion_sha256=ch,elapsed_s=elapsed))
    sha(OUTPUT/'finalization.json'); ok = bool(ok and time.monotonic()-began < CAP)
    print(json.dumps(dict(passed=ok,completed=ok,full_model_calls=result['full_model_calls'],
        elapsed_s=elapsed,error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
