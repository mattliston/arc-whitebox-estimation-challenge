"""Resource-only checkpoint8 component confirmation; preserves failed discovery receipts."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys

D = Path(__file__).resolve().parent
H = D.parent
# Actual resource plan hash is supplied by the source-bound launch.
ROUTE_SHA = 'acf958a2b20cbab5c3a1c1085fc7e1463c387134801665072da40ae101e35994'
VALID8 = [37, 39, 42, 48, 59, 63, 66, 67]
COUNTS = dict(bound=687, lvfuse=521, psdb_fused=9, pair_block_indices=1, read7_fused=4, ksplit=24)
THREADS = dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'), '1')


def pin(path):
    p=Path(path).resolve(); b=p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def checked(record):
    assert pin(record['path']) == record, record['path']
    return Path(record['path']).read_bytes()


def load(record):
    return json.loads(checked(record))


def setup(manifest, counts):
    assert manifest['runtime'] == str(NATIVE) and manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == THREADS
    s=manifest['setup']; assert math.isfinite(s['setup_s']) and 0 <= s['setup_s'] <= 4.1
    for key,value in counts.items():
        assert [r[1] for r in s['pb_log'] if r[0] == key] == [value]


def rows(q, record, ranks, ubc=False):
    out=[json.loads(s) for s in q.checked(record).decode().splitlines()]
    assert [r['rank'] for r in out] == ranks
    for r in out:
        assert math.isfinite(r['cpu_s']) and r['cpu_s'] > 0
        assert type(r['flops']) is int and 0 < r['flops'] <= 2**41
        if ubc:
            assert r['ubc']['retired'] and r['ubc']['retained_cache_bytes'] == r['ubc']['live_cache_bytes'] == 0
            assert 0 <= r['ubc']['max_resident_bytes'] <= 512000000
    return out


def assess(plan_path, sha):
    global NATIVE
    assert plan_path == D/'resource_plan.json'
    pp=pin(plan_path); assert sha == pp['sha256']
    p=load(pp); assert p['assessor'] == pin(__file__)
    study=load(p['component_plan']); assert study['research_permission'] and study['candidate']==p['candidate']
    policy=load(p['root_policy']); assert policy['root_selected'] and policy['candidate']==p['candidate'] and policy['source_unchanged']
    assert study['original_failed_assessment']==p['original_failed_assessment'] and study['followup_failed_assessment']==p['first_accuracy_assessment']
    original=load(p['original_failed_assessment']); assert original['passed'] is False and not original.get('admitted',False)
    route_pin=p['bank_route']; assert route_pin['sha256'] == ROUTE_SHA
    route=load(route_pin); src=route['sources']
    assert p['schema'] == 'detached-star8-component-own-resource-preflight-v1'
    assert p['ranks'] == [35,100] and p['first_accuracy_failed_preserved'] is True
    assert p['accuracy_selection_on_resource_ranks'] is False
    assert (p['native_guard_seconds'],p['boundary_guard_seconds'],p['rss_bytes'],p['added_swap_bytes']) == (150,240,7500000000,0)
    helper=src['boundary_assessor_template']; checked(helper)
    spec=importlib.util.spec_from_file_location('full80_frozen_boundary_helpers',helper['path'])
    rr=importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    m,q=rr.helpers(); NATIVE=q.NATIVE
    for r in (pp,route_pin,helper,src['boundary_config_template'],p['candidate'],p['export_manifest'],p['guard'],p['native_driver'],p['boundary_driver'],p['boundary_source']): q.checked(r)
    assert p['light'] == str(q.LIGHT) and p['native_runtime'] == str(NATIVE)
    assert p['native_driver'] == src['native_probe'] and p['boundary_driver'] == src['boundary_driver']
    assert p['guard'] == src['guard']
    assert p['boundary_source']['sha256'] == q.HARNESS_SHA
    assert p['native_tag'] == D.name+'_resource2_v1' and p['boundary_tag'] == D.name+'_boundary_v1'
    ex=json.loads(q.checked(p['export_manifest']))
    assert ex['passed'] and ex['candidate'] == p['candidate'] and ex['optimizer_updates'] == 8
    assert ex['training_complete8_verified'] and ex['training_guards_checked']==2
    assert ex['source_inverse_exact'] and ex['unrelated_statements_ast_exact']
    c=json.loads(q.checked(src['boundary_config_template']))
    for r in (c['s34_bank'],c['s34_evaluation'],c['parent'],c['parent_boundary'],c['s34_native_config'],c['control_native_engine']): q.checked(r)

    # Saved generic-driver followup8 remains FAILED scientifically; use its timings only.
    first=json.loads(q.checked(p['first_accuracy_assessment']))
    assert first['candidate']==p['candidate'] and first['receipt_checks_passed']
    assert first['passed'] is False and first['retained_evidence_passed'] is False
    assert first['original_admitted'] is False and first['original_failure_preserved']
    assert first['original_assessment']==p['original_failed_assessment']
    assert first['all16_finite'] and first['layer0_byte_exact'] and first['known2_prediction_bytes_mse_bill_exact']
    fp=json.loads(q.checked(first['protocol']))
    assert first['protocol']==p['followup_protocol'] and fp['candidate']==p['candidate']
    assert fp['ranks']==VALID8 and fp['setup_counts']==COUNTS
    assert first['driver']==fp['driver']==p['followup_driver']
    q.checked(first['driver']); q.checked(first['assessor'])
    vg=q.guard(first['native_guard'],[str(NATIVE),'-B',fp['driver']['path'],p['candidate']['path'],','.join(map(str,VALID8)),fp['tag']],300)
    q.guard(p['followup_assessment_guard'],[str(NATIVE),'-B',first['assessor']['path'],first['protocol']['path'],first['protocol']['sha256']],120)
    vm=json.loads(q.checked(first['artifacts']['manifest.json']))
    assert vm['candidate']==p['candidate']['path'] and vm['sha256']==p['candidate']['sha256']
    assert vm['ranks']==VALID8; setup(vm,COUNTS)
    assert Path(first['artifacts']['manifest.json']['path']).parent==H/fp['tag']
    assert Path(first['artifacts']['rows.jsonl']['path']).parent==H/fp['tag']
    valid_rows=rows(q,first['artifacts']['rows.jsonl'],VALID8,True)

    # Matched S32 first8 of its original one-process b1, frozen before endpoint.
    ref=route['runtime_route']['s32_valid8_control']
    controls=json.loads(q.checked(ref['controls'])); arm=controls['arm']; block=arm['blocks']['b1']
    assert arm['source'] == ref['source'] == c['runtime_reference']['source']
    q.checked(arm['source']); q.checked(ref['driver'])
    assert block['ranks'] == ref['actual_process_order'] and block['ranks'][:8] == VALID8 == ref['ranks']
    assert block['driver'] == ref['driver'] and block['no_bytecode'] is False
    for key in ('manifest','rows','guard'): assert block['artifacts'][key] == ref[key]
    sm=json.loads(q.checked(ref['manifest']))
    assert sm['candidate'] == ref['source']['path'] and sm['sha256'] == ref['source']['sha256'] and sm['ranks'] == block['ranks']; setup(sm,arm['setup']['counts'])
    s32vg=q.guard(ref['guard'],[str(NATIVE),ref['driver']['path'],ref['source']['path'],','.join(map(str,block['ranks'])),block['tag']],360)
    s32v=rows(q,ref['rows'],block['ranks'])[:8]
    assert [r['cpu_s'] for r in s32v] == ref['cpu_s_first8']
    valid_ratio=math.fsum(r['cpu_s'] for r in valid_rows)/math.fsum(r['cpu_s'] for r in s32v)

    folder=H/p['native_tag']
    own_records=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(D/'resource2_guard.json'))
    own_arm=dict(source=p['candidate'],setup=dict(counts=COUNTS),ubc_retired_required=True,blocks={'2':dict(ranks=[35,100],tag=p['native_tag'],driver=p['native_driver'],no_bytecode=True,guard=str(D/'resource2_guard.json'),limits=dict(seconds=150,rss_bytes=7500000000))})
    native=rr.native_arm(q,own_arm,2,own_records)
    # Own command punctuation is exact; historical controls alone permit their established normalization.
    ng=q.guard(own_records['guard'],[str(NATIVE),'-B',p['native_driver']['path'],p['candidate']['path'],'35,100',p['native_tag']],150)
    runtime_cfg=json.loads(q.checked(c['runtime_reference']['native_config']))
    assert c['runtime_reference']['native_config']['sha256'] == rr.RUNTIME_CONFIG_SHA
    assert runtime_cfg['engine'] == c['control_native_engine']
    ra=runtime_cfg['arms']['base']; assert ra['source'] == ref['source']
    s32two=rr.native_arm(q,ra,2,{k:ra['blocks']['2']['artifacts'][k] for k in ('manifest','rows','guard')})
    native_ratio=math.fsum(r['cpu_s'] for r in native)/math.fsum(r['cpu_s'] for r in s32two)

    paired_pin=q.fixed(rr.V1/'paired_boundary_assessment.json',rr.CONTROL_SHA)
    paired=json.loads(q.checked(paired_pin)); assert not paired['passed'] and paired['receipt_checks_passed']
    q.guard(q.fixed(rr.V1/'paired_boundary_assess_guard.json',rr.CONTROL_GUARD_SHA),[str(q.LIGHT),'-B',str(rr.V1/'paired_boundary_assess.py')],120)
    control=paired['jobs'][0]; assert control['role']=='s32' and control['source']==ref['source']
    base,bg=q.boundary(control['census'],control['launcher'],control['guard'],control['source'],'hill_climb_precision_g4_symdiag_v1_boundary_fresh_s32_v1')
    base_setup=m.setup_record(q,control['log']['path'],624,600)
    census=pin(H/(p['boundary_tag']+'_census.json')); launcher=pin(H/(p['boundary_tag']+'_launcher.json'))
    boundary_guard=pin(D/'boundary_guard.json')
    current,g=q.boundary(census,launcher,boundary_guard,p['candidate'],p['boundary_tag'])
    current_setup=m.setup_record(q,D/'boundary.log',687,521)
    for key,count in COUNTS.items(): assert [r[1] for r in current_setup['setup_log'] if r[0]==key] == [count]
    comparisons=[]
    for i,(x,y) in enumerate(zip(current,base)):
        assert x['flops'] == native[i]['flops'] and y['flops'] == s32two[i]['flops']
        assert x['round_trips'] == x['requests'] + x['free_trips']
        comparisons.append(dict(rank=x['rank'],full_native_bills_exact=True,request_delta=x['requests']-y['requests'],server_cpu_ratio=x['server_cpu_s']/y['server_cpu_s']))

    official=json.loads(q.checked(q.fixed(H/'eval_332738.json','6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6')))
    ev=official['evaluation']; assert official['submission_id']==332738 and ev['submission_id']=='332738'
    assert ev['progress']['mlps_completed']==100 and ev['progress']['mlps_failed']==0
    telemetry=[r['telemetry'] for r in ev['results']['per_mlp']]; assert len(telemetry)==100
    assert all(math.isfinite(t[k]) and t[k]>=0 for t in telemetry for k in ('participant_predict_wall_s','flopscope_server_kernel_wall_s','residual_wall_time_s'))
    boundary_ratio=statistics.mean(x['server_cpu_ratio'] for x in comparisons)
    disagreement=abs(100*(boundary_ratio-native_ratio))
    growth=max(0.,native_ratio-1,valid_ratio-1,boundary_ratio-1)
    requests=max(0.,statistics.mean(x['request_delta'] for x in comparisons))
    grid=[dict(native_fraction=f,request_cost_s=p,projected_worst_wall_s=max(t['participant_predict_wall_s']+f*growth*t['flopscope_server_kernel_wall_s']+requests*p for t in telemetry)) for f in (.60,.78,.95) for p in (.00030,.00041,.00050)]
    wall=max(x['projected_worst_wall_s'] for x in grid)

    bank=json.loads(q.checked(c['s34_bank'])); assert bank['submission_id']==333050 and bank['candidate']==c['parent']
    assert bank['authenticated_poll']['grading_status_cd']=='graded' and bank['evaluation_receipt']==c['s34_evaluation']
    ev34=json.loads(q.checked(c['s34_evaluation']))['evaluation']
    assert ev34['state']=='completed' and ev34['error'] is None and ev34['submission_id']=='333050'
    assert ev34['progress']['mlps_completed']==100 and ev34['progress']['mlps_failed']==0
    t34=[r['telemetry'] for r in ev34['results']['per_mlp']]; assert len(t34)==100
    maximum=max(t['residual_wall_time_s'] for t in t34); assert maximum==bank['telemetry']['max_residual_wall_s']
    pr=json.loads(q.checked(c['parent_boundary'])); assert pr['receipt_checks_passed'] and pr['candidate']==c['parent'] and pr['setup']['setup_wall_s']<=4.1
    parent_rows,pg=q.boundary(pr['current_census'],pr['launcher'],pr['guard'],c['parent'],Path(c['parent']['path']).parent.name+'_boundary_feasibility_v1')
    s34cfg=json.loads(q.checked(c['s34_native_config'])); assert s34cfg['engine']==c['control_native_engine']
    sa=s34cfg['arms']['base']; assert sa['source']==c['parent']
    s34n=rr.native_arm(q,sa,2,{k:sa['blocks']['2']['artifacts'][k] for k in ('manifest','rows','guard')})
    assert [(r['rank'],r['flops']) for r in s34n]==[(r['rank'],r['flops']) for r in parent_rows]
    deltas=[dict(rank=x['rank'],candidate_residual_s=x['residual_s'],parent_residual_s=y['residual_s'],residual_delta_s=x['residual_s']-y['residual_s']) for x,y in zip(current,parent_rows)]
    residual=maximum+4*max(0.,*(x['residual_delta_s'] for x in deltas))
    gates=dict(setup=max(current_setup['setup_wall_s'],base_setup['setup_wall_s'])<=4.1,residual_projection=residual<.4,runtime_projection=wall<=105,cpu_agreement=disagreement<=5,native_rpc_bills_exact=True,guards_retired_zero_swap=True)
    for record in list(q.EVIDENCE.values()): q.checked(record)
    return dict(schema='detached-star8-component-resource-assessment-v1',resource_passed=all(gates.values()),receipt_checks_passed=True,plan=pp,assessor=pin(__file__),component_plan=p['component_plan'],bank_route=route_pin,candidate=p['candidate'],first_accuracy_assessment=p['first_accuracy_assessment'],original_failed_assessment=p['original_failed_assessment'],initial_VALID8_remains_failed=True,accuracy_selection_on_resource_ranks=False,native_resource2=own_records,current_census=census,launcher=launcher,guard=boundary_guard,setup=current_setup,comparisons=comparisons,gates=gates,runtime=dict(native2_cpu_ratio=native_ratio,matched_valid8_cpu_ratio=valid_ratio,matched_valid8_ranks=VALID8,candidate_valid8_rows=first['artifacts']['rows.jsonl'],s32_valid8_rows=ref['rows'],boundary_mean_cpu_ratio=boundary_ratio,native2_boundary_disagreement_pp=disagreement,used_nonnegative_cpu_change=growth,nonnegative_mean_request_delta=requests,projection_grid=grid,projected_worst_wall_s=wall,gate_s=105,request_cost_s_bounds=[.00030,.00050],per_request_timing_observed=False),residual=dict(official_s34_max_s=maximum,multiplier=4,projected_s=residual,gate_s=.4,strict_upper_bound=True,comparisons=deltas),maximum_own_rss_bytes=max(ng['max_rss_bytes'],g['max_rss_bytes'],vg['max_rss_bytes']),evidence=list(q.EVIDENCE.values()),science_admitted=False,broad_admitted=False,bank_qualified=False,serving_qualified=False,submitted=False,arrays_read=False,model_executed=False)


def main():
    assert len(sys.argv)==3
    target=D/'resource_assessment.json'; assert not target.exists()
    try:
        result=assess(Path(sys.argv[1]).resolve(),sys.argv[2])
    except Exception as exc:
        result=dict(schema='detached-star8-component-resource-assessment-v1',resource_passed=False,receipt_checks_passed=False,assessor=pin(__file__),error=dict(type=type(exc).__name__,message=str(exc)),initial_VALID8_remains_failed=True,science_admitted=False,broad_admitted=False,bank_qualified=False,submitted=False)
        with target.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
        raise
    with target.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:result[k] for k in ('resource_passed','receipt_checks_passed','gates')}))


if __name__=='__main__': main()
