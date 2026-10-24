"""Release gate requiring completed five-worker setup plus normal/slow runtime.

setup CANDIDATE_SETUP_DIR V61_SETUP_DIR OUTPUT_JSON
full NORMAL_DIR SLOW13_DIR CANDIDATE_SETUP_DIR V61_SETUP_DIR OUTPUT_JSON

Reads existing receipts only; never runs an SDK or model. Setup uses BOTH
max(candidate)/min(v61) completed-bound and full-start-envelope ratios against
the maximum successful public v61 setup. Request counts are not a setup gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

O = Path(__file__).resolve().parents[3]
R = O/'lead/runs'
V61_SHA = 'ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
LIMITS = dict(residual_median_s=.3452224775, residual_max_s=.390682602,
              roundtrip_max_s=118., setup_max_s=4.60)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def completed_five(directory, guard_path=None):
    directory = Path(directory).resolve()
    plan, summary = read(directory/'PLAN_v1.json'), read(directory/'SUMMARY_v1.json')
    require(plan['versions'] == dict(whestbench='0.16.1', flopscope_client='0.12.3', flopscope_server='0.12.3'), 'Wrong setup SDK versions')
    require(plan['workers'] == 5 and plan['setup_timeout_s'] == 5., 'Require five workers under5s start deadline')
    require(plan['setup_budget'] is True and plan['no_predict'] is True, 'Require real setup BudgetContext')
    require(plan['initial_memory_free_pct'] >= 50, 'Setup launch memory free below50%')
    require(hashlib.sha256(Path(plan['candidate']).read_bytes()).hexdigest() == plan['sha256'], 'Setup candidate file/hash changed')
    require(summary['status']=='PASS' and summary['error'] is None and summary['server_processes_retired'], 'Setup group failed or not retired')
    if guard_path is None:
        prefix, version = directory.name.rsplit('_v', 1)
        guard_path = directory.with_name(prefix+'_guard_v'+version+'.json')
    guard_path = Path(guard_path).resolve()
    guard = read(guard_path)
    require(str(Path(plan['candidate']).resolve()) in guard['command'] and str(directory) in guard['command'], 'Setup guard command does not match candidate/output paths')
    require(guard['completed'] and guard['whole_group_retired'] and guard['error'] is None, 'Setup guard did not complete cleanly')
    require(guard['min_pressure_free_pct'] >= 50 and guard['max_rss_bytes'] <= 7_500_000_000, 'Setup memory guard exceeded')
    workers = [read(directory/f'worker{i}.json') for i in range(5)]
    require(summary['records'] == workers, 'Worker receipts differ from group summary')
    values = []
    for i, worker in enumerate(workers):
        require(worker['worker']==i and worker['status']=='PASS' and worker['worker_retired'], 'Worker failed or not retired')
        require(worker['setup_timeout_s']==5. and 0 < worker['start_envelope_s'] <= 5., 'Invalid full start envelope')
        require(len(worker['start_responses'])==1, 'Missing or duplicate start response')
        response = worker['start_responses'][0]
        require(response['status']=='ok', 'Official start response not ok')
        diagnostic = response['diagnostic']
        require(diagnostic['setup_completed'] is True and diagnostic['setup_budget'] is True, 'Setup not actually completed in budget')
        require(diagnostic['bind_errors'] == [], 'Setup had a caught prebind exception')
        require(diagnostic['setup_flops'] > 0, 'Setup did no measured work')
        logs = diagnostic['pb_log']
        require(not any(row[0]=='restored after failure' for row in logs), 'Silent setup rollback')
        bound = [row for row in logs if row[0]=='bound']
        require(len(bound)==1 and bound[0][1]>0 and 0 < bound[0][-1] <= worker['start_envelope_s'], 'Missing/invalid completed bound marker')
        stages = [json.loads(line[6:]) for line in worker['worker_stderr'] if line.startswith('STAGE ')]
        boot = [s for s in stages if s.get('stage')=='boundary_bootstrap']
        require(len(boot)==1 and boot[0]['numpy_available'] is False and boot[0]['flopscope']=='0.12.3', 'Worker is not verified NumPy-free0.12.3 boundary')
        values.append(dict(worker=i, bound_s=float(bound[0][-1]), envelope_s=worker['start_envelope_s']))
    return dict(directory=str(directory), sha256=plan['sha256'], values=values,
                guard=str(guard_path), completed_workers=5, rollback_count=0)


def public_telemetry():
    public = read(O/'fire/opus_runtime_v61j_ac93b10f/eval_336867.json')['evaluation']['results']
    tele = [row['telemetry'] for row in public['per_mlp']]
    require(len(tele)==100 and all(t['residual_wall_time_s']>0 for t in tele), 'v61 public control is incomplete')
    setups = public['setup_executions']
    require(len(setups)==5 and all(s['error_code']=='SUCCESS' for s in setups), 'v61 public setups not all successful')
    return tele, max(s['wall_s'] for s in setups)


def setup_gate(candidate_dir, reference_dir, candidate_guard=None):
    candidate, reference = completed_five(candidate_dir, candidate_guard), completed_five(reference_dir)
    require(reference['sha256']==V61_SHA, 'Setup reference must be actual graded v61')
    _, public_max = public_telemetry()
    ratios = {key:max(v[key] for v in candidate['values'])/min(v[key] for v in reference['values'])
              for key in ('bound_s','envelope_s')}
    projected = {key:public_max*ratio for key,ratio in ratios.items()}
    return dict(status='PASS' if all(v <= LIMITS['setup_max_s'] for v in projected.values()) else 'FAIL',
        candidate_sha256=candidate['sha256'], control_sid=336867, candidate=candidate, reference=reference,
        max_candidate_over_min_reference=ratios, public_control_setup_max_s=public_max,
        projected_setup_s=projected['bound_s'], projected_envelope_s=projected['envelope_s'],
        limit_s=LIMITS['setup_max_s'],
        limitation='Calibrated completed-setup and full-start ratios; no claim of identical hosted startup or known remote traceback.')


def runtime_gate(normal, slow, setup):
    tele, _ = public_telemetry()
    base = R/'opus_fix336687_v1/jsk'
    record = dict(base_sid=336867, limits=LIMITS, setup=setup, modes={},
                  candidate_sha256=setup['candidate_sha256'])
    checks = [setup['status']=='PASS']
    for mode, directory, reference in [('normal',Path(normal),base/'bnd_v61j_normal'),
                                       ('slow13',Path(slow),base/'bnd_v61j_slow13')]:
        manifest, ref_manifest = read(directory/'manifest.json'), read(reference/'manifest.json')
        require(manifest['client_flopscope'].split('+')[0]=='0.12.3' and ref_manifest['client_flopscope'].split('+')[0]=='0.12.3', 'Runtime SDK mismatch')
        require(ref_manifest['sha256']==V61_SHA, 'Wrong runtime control lineage')
        require(manifest['sha256']==setup['candidate_sha256'], 'Setup/runtime candidate SHA mismatch')
        require(manifest['setup_slowdown']==(1.0 if mode=='normal' else 1.3), 'Wrong setup pace mode')
        rows = lambda p: {r['rank']:r for r in map(json.loads,(p/'rows.jsonl').read_text().splitlines())}
        cand, ctrl = rows(directory), rows(reference)
        require(set(cand)==set(ctrl)=={35,200}, 'Require paired cold35/warm200 runtime rows')
        require(not manifest['profiled'] and not manifest['sites'] and all(not r.get('census_instrumented') for r in cand.values()), 'Runtime must be uninstrumented')
        clean = not manifest['setup']['failed'] and all(r['nonfinite']==0 and
            r['fast_rescue_control']['layer'] is None and r['fast_rescue_stats']['calls']==0 for r in cand.values())
        require(not any(row[0]=='restored after failure' for row in manifest['setup'].get('pb_log',[])), 'Runtime setup rollback')
        ratios = {k:cand[k]['residual_instructions']/ctrl[k]['residual_instructions'] for k in cand}
        residual = [t['residual_wall_time_s']*ratios[35 if t.get('participant_setup_wall_s') is not None else 200] for t in tele]
        wall_ratio = max(cand[k]['wall_s']/ctrl[k]['wall_s'] for k in cand)
        projected = dict(residual_median_s=statistics.median(residual), residual_max_s=max(residual),
                         roundtrip_max_s=max(t['participant_roundtrip_wall_s'] for t in tele)*wall_ratio)
        passed = clean and all(projected[k] <= LIMITS[k] for k in projected)
        checks.append(passed)
        record['modes'][mode] = dict(clean=clean, passed=passed, projected=projected,
            instruction_ratios=ratios, wall_ratio=wall_ratio, requests={k:cand[k]['requests'] for k in cand},
            reference=str(reference), source=str(directory))
    record['status'] = 'PASS' if all(checks) else 'FAIL'
    record['limitation'] = 'Paired local instruction/wall projections plus independently completed five-worker setup; future platform noise remains unobserved.'
    return record


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='mode',required=True)
    s=commands.add_parser('setup')
    f=commands.add_parser('full')
    f.add_argument('normal',type=Path);f.add_argument('slow',type=Path)
    for p in (s,f):
        p.add_argument('candidate_setup',type=Path);p.add_argument('reference_setup',type=Path)
        p.add_argument('output',type=Path)
    a=parser.parse_args()
    require(not a.output.exists(),'Never overwrite a decision receipt')
    try:
        result=setup_gate(a.candidate_setup,a.reference_setup)
        if a.mode=='full':result=runtime_gate(a.normal,a.slow,result)
    except Exception as exc:
        result=dict(status='FAIL',reason=repr(exc),mode=a.mode)
    result['gate_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
