"""Read-only release gate: completed official setup and paired v3 thread counters.
CLI: CAND_NORMAL CAND_SLOW CTRL_NORMAL CTRL_SLOW CAND_SETUP CTRL_SETUP BYTE_REFS OUTPUT --cold259 DIR.
Original process ratios and old failure receipts remain untouched and visible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import runtime_gate_current_v4 as base

read, require, LIMITS = base.read, base.require, base.LIMITS
HERE = Path(__file__).resolve().parent
CLIENT_SHA = '60c0a5bba801aedcff99d881fce5ae3fdc15dfdc5d7dd09641b7e725f9f32066'
HELPER_SHA = '52a848ef142f15af9be3a52fbbe5c911c028b4ba8a32b2a897568f06e000e307'
RUNNER_SHA = '3d829d2a28268829ad208b7b71514f1c708211e74d04c5e6b8dacb003baa60bb'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def group(directory, expected_sha, pace, ranks=(35,200)):
    directory = Path(directory).resolve()
    m, launch, guard = (read(directory/name) for name in ('manifest.json','launcher.json','GUARD_v1.json'))
    require(m['sha256']==launch['sha256']==expected_sha==sha(m['candidate']), 'Candidate SHA mismatch')
    require(launch['client_exit']==0 and launch['server_exit']==0, 'Runtime processes did not finish cleanly')
    require(launch['handle_cap']==100000, 'Wrong handle cap')
    require(guard['completed'] and guard['whole_group_retired'] and guard['exit_code']==0 and guard['error'] is None, 'Runtime guard failed/not retired')
    require(guard['min_pressure_free_pct']>=50 and guard['max_rss_bytes']<=7_500_000_000 and guard['max_footprint_bytes']<=7_500_000_000, 'Runtime memory limit exceeded')
    require(m['client_flopscope'].split('+')[0]=='0.12.3', 'Wrong runtime SDK')
    require(m['setup_slowdown']==pace and m['ranks']==list(ranks), 'Wrong pace/rank order')
    require(not m['profiled'] and not m['sites'], 'Profiling/site census is disallowed')
    require(m['diagnostic_harness_sha256']==CLIENT_SHA and m['thread_helper_sha256']==HELPER_SHA, 'Wrong dual-counter harness')
    h=HERE/'dual_thread_boundary_v3'
    require(sha(h/'boundary_client_dual_v3.py')==CLIENT_SHA and sha(h/'thread_counts_v3.py')==HELPER_SHA and sha(h/'boundary_run_dual_v3.py')==RUNNER_SHA, 'Frozen harness changed')
    require(str(h/'boundary_run_dual_v3.py') in guard['command'], 'Wrong guarded runtime launcher')
    require(m['dual_counter']['dispatch_window']=='thread outer, process inner' and m['dual_counter']['top_level_span_count_recorded'], 'Wrong counter window')
    require(not m['setup']['failed'] and not any(x[0]=='restored after failure' for x in m['setup']['pb_log']), 'Runtime setup failed/rolled back')
    values=[json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
    require([r['rank'] for r in values]==list(ranks), 'Runtime row order/integrity failure')
    rows={r['rank']:r for r in values}
    for r in values:
        require(r['nonfinite']==0 and r['n_out']==16384, 'Invalid prediction')
        require(r['fast_rescue_control']['layer'] is None and r['fast_rescue_stats']['calls']==0, 'Rescue path used')
        require(not r.get('census_instrumented'), 'Science census instrumented runtime')
        require(r['thread_residual_instructions']>0 and r['top_level_dispatch_spans']>0, 'Missing thread counters')
        require(r['thread_total_instructions']-r['thread_dispatch_instructions']==r['thread_residual_instructions'], 'Thread instruction arithmetic mismatch')
    return dict(directory=str(directory), manifest=m, rows=rows, guard=guard)

def repeatability(normal, slow):
    answer={}
    for rank in (35,200):
        a,b=normal['rows'][rank],slow['rows'][rank]
        for key in ('requests','free_trips','freed_handles','flops','operations','top_level_dispatch_spans'):
            require(a[key]==b[key], f'Within-source work changed at rank{rank}: {key}')
        require(sha(Path(normal['directory'])/f'pred_{rank}.bin')==sha(Path(slow['directory'])/f'pred_{rank}.bin'), 'Normal/slow prediction mismatch')
        answer[rank]={key:dict(normal=a[key],slow=b[key],spread_pct=100*(b[key]/a[key]-1))
                      for key in ('thread_residual_instructions','residual_instructions')}
    return answer

def evaluate(normal,slow,control_normal,control_slow,candidate_setup,control_setup,byte_references,cold259,setup_guard=None):
    setup=base.setup_gate(candidate_setup,control_setup,candidate_guard=setup_guard)
    candidate_sha=setup['candidate_sha256']
    cn,cs=group(normal,candidate_sha,1.0),group(slow,candidate_sha,1.3)
    rn,rs=group(control_normal,base.V61_SHA,1.0),group(control_slow,base.V61_SHA,1.3)
    rep=dict(candidate=repeatability(cn,cs),control=repeatability(rn,rs))
    byte_references=Path(byte_references).resolve()
    reference=read(byte_references)
    require(reference['source_sha256']==candidate_sha and reference['science_n']==32, 'Need this exact candidate own32 byte references')
    require(sha(reference['science_result_path'])==reference['science_result_sha256'], 'Science completion receipt changed')
    require(sha(reference['science_grade_path'])==reference['science_grade_sha256'], 'Science32 grade receipt changed')
    require(len(reference['rows'])==32 and len({r['rank'] for r in reference['rows']})==32, 'Incomplete own32 byte references')
    for r in reference['rows']:
        require(Path(r['path']).stat().st_size==65536 and sha(r['path'])==r['sha256'], 'Frozen science prediction changed')
    hashes={r['rank']:r['sha256'] for r in reference['rows']}
    cold=group(cold259,candidate_sha,1.0,ranks=(259,))
    require(sha(Path(cold['directory'])/'pred_259.bin')==hashes[259], 'Cold259 differs from newscience32 bytes')
    for g in (cn,cs):
        for rank in (35,200):
            path=Path(g['directory'])/f'pred_{rank}.bin'
            require(path.stat().st_size==65536 and sha(path)==hashes[rank], 'Candidate differs from scientific32 prediction bytes')
    tele,_=base.public_telemetry()
    record=dict(base_sid=336867,candidate_sha256=candidate_sha,limits=LIMITS,setup=setup,modes={},repeatability=rep,
                method='v3 current-thread total minus outer-thread dispatch spans; fixed separate normal/slow paired controls; no pooling',
                scientific32_byte_references=str(byte_references),
                scientific32_source_sha256=candidate_sha,
                cold259=dict(source=cold['directory'],bytes_equal=True,guard_clean=True,max_rss_bytes=cold['guard']['max_rss_bytes']))
    checks=[setup['status']=='PASS']
    for mode,c,r in [('normal',cn,rn),('slow13',cs,rs)]:
        cand,ctrl=c['rows'],r['rows']
        ratios={k:cand[k]['thread_residual_instructions']/ctrl[k]['thread_residual_instructions'] for k in cand}
        process_ratios={k:cand[k]['residual_instructions']/ctrl[k]['residual_instructions'] for k in cand}
        residual=[t['residual_wall_time_s']*ratios[35 if t.get('participant_setup_wall_s') is not None else 200]for t in tele]
        wall_ratio=max(cand[k]['wall_s']/ctrl[k]['wall_s']for k in cand)
        projected=dict(residual_median_s=statistics.median(residual),residual_max_s=max(residual),
                       roundtrip_max_s=max(t['participant_roundtrip_wall_s']for t in tele)*wall_ratio)
        process_residual=[t['residual_wall_time_s']*process_ratios[35 if t.get('participant_setup_wall_s') is not None else 200]for t in tele]
        passed=all(projected[k]<=LIMITS[k]for k in projected)
        checks.append(passed)
        record['modes'][mode]=dict(passed=passed,projected=projected,thread_instruction_ratios=ratios,
            process_instruction_ratios_diagnostic=process_ratios,
            process_projected_diagnostic=dict(residual_median_s=statistics.median(process_residual),residual_max_s=max(process_residual)),
            wall_ratio=wall_ratio,requests={k:cand[k]['requests']for k in cand},
            spans={k:dict(candidate=cand[k]['top_level_dispatch_spans'],control=ctrl[k]['top_level_dispatch_spans'])for k in cand},
            source=c['directory'],reference=r['directory'],guards='all four groups clean',scientific32_bytes_equal=True)
    record['status']='PASS' if all(checks) else 'FAIL'
    record['limitation']='Local paired instruction/wall projection, not a platform guarantee. Both modes must pass unchanged limits. Prior process-only failed receipts are not superseded or deleted; process diagnostics and repeatability remain explicit.'
    return record

def main():
    p=argparse.ArgumentParser()
    for k in ('normal','slow','control_normal','control_slow','candidate_setup','control_setup','byte_references','output'):p.add_argument(k,type=Path)
    p.add_argument('--cold259',type=Path,required=True)
    p.add_argument('--setup-guard',type=Path,default=None)
    a=p.parse_args();require(not a.output.exists(),'Never overwrite decision receipt')
    try:result=evaluate(a.normal,a.slow,a.control_normal,a.control_slow,a.candidate_setup,a.control_setup,a.byte_references,a.cold259,a.setup_guard)
    except Exception as exc:result=dict(status='FAIL',reason=repr(exc))
    result['gate_source_sha256']=sha(__file__)
    result['setup_gate_source_sha256']=sha(base.__file__)
    with a.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
