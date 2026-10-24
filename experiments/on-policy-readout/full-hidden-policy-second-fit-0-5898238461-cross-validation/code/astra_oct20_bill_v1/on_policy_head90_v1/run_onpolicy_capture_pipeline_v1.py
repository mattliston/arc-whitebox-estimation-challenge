"""Six frozen target-free capture families; pass-only anchors, original guards.

Root review is mandatory. Optional frozen fit is a separate explicit flag and
never exports a candidate or launches native validation. No automatic fire.
"""
import concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
A = HERE.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1'
PORT = A/'cloud_linux_capacity_v7'
L = A/'cloud_linux_markline_v1'
ORIG = HERE.parent/'truth_response95_v1'
sys.path.insert(0, str(PORT))
from common_linux_v7 import thread_env
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
PY = '/home/arcworker/arc-envs/judge/bin/python'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
READY = '71e56701238bd6ff1919c78977eb6a5351710fca8566ba57086a82e259fb43a4'
PREREG = '0c94b0dc84bd17f0542df28e9ae56d7486c8944fe3cd08db74e6c91a76c24443'
MANIFEST = 'd6eaeb233f5f8109247d46cc68672848fb10477bd18a4076829fd267a4613136'

def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')

def guardcheck(path, threads):
    g = read(path)
    assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
    assert g['guard_sha256'] == GUARD and g['thread_provenance']['blas_threads'] == threads
    assert g['max_owned_swap_bytes'] == 0 and g['cap_bytes'] == 6500000000 and g['reserve_bytes'] == 19000000000
    assert g['max_active'] == 36
    for name in ('oom', 'oom_kill', 'oom_group_kill'):
        assert g['cgroup_events_final'][name] == g['cgroup_events_initial'][name]
    return sha(path)

def start(command, tag, threads, evidence):
    log = RUNS/(tag+'.log')
    launch = HERE/'pipeline_launches_v1'/(tag+'_LAUNCH_v1.json')
    assert not log.exists() and not launch.exists()
    with log.open('x') as output:
        p = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=dict(os.environ, **thread_env(threads), ARC_BLAS_THREADS=str(threads)))
    rec = dict(utc=utc(), pid=p.pid, command=command, log=str(log), evidence=evidence,
               controller_sha256=sha(__file__), root_review_sha256=ROOT_REVIEW_SHA, no_auto_fire=True)
    write(launch, rec)
    print(json.dumps(dict(utc=rec['utc'], tag=tag, pid=p.pid)), flush=True)
    return p

def native(family, stage, jobs):
    item = family['outputs'][stage]
    out = Path(item['out'])
    assert not out.exists()
    plan = ROOT/item['plan']['path']
    assert sha(plan) == item['plan']['sha256']
    runner = L/'run_science_linux_markline_v1.py' if stage == 'control' else ORIG/'run_capture_v1.py'
    cmd = [PY, '-B', str(runner)]
    if stage != 'control':
        recipe = ROOT/item['recipe']['path']
        assert sha(recipe) == item['recipe']['sha256']
        cmd += ['--recipe', str(recipe)]
    cmd += ['--root', str(ROOT), '--plan', str(plan), '--candidate', 'teacher', '--out', str(out),
            '--state-dir', str(RUNS/'admission'), '--client-python', '/home/arcworker/arc-envs/client/bin/python',
            '--server-python', '/home/arcworker/arc-envs/server/bin/python', '--jobs', str(jobs),
            '--max-active', '36', '--blas-threads', '4', '--setup', 'paced',
            '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5']
    tag = 'onpolicy_head90_'+family['key']+'_'+stage+'_v1'
    return start(cmd, tag, 4, dict(plan_sha256=sha(plan), runner_sha256=sha(runner), source_sha256=family['teacher']['sha256']))

def audit(family, stage):
    item = family['outputs'][stage]
    receipt = Path(item['audit_guard_path'])
    assert not receipt.exists()
    auditor = HERE/'capture_support_v1/audit_onpolicy_capture_v1.py'
    cmd = [PY, '-B', str(PORT/'guard_linux_v7.py'), '--receipt', str(receipt), '--state-dir', str(RUNS/'admission'),
           '--max-active', '36', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--', PY, '-B', str(auditor),
           '--family', family['key'], '--stage', stage, '--ready-sha256', READY]
    tag = 'onpolicy_head90_'+family['key']+'_'+stage+'_audit_v1'
    child = start(cmd, tag, 1, dict(auditor_sha256=sha(auditor), ready_sha256=READY))
    assert child.wait() == 0
    guardsha = guardcheck(receipt, 1)
    path = Path(item['out'])/'AUDIT_v1.json'
    report = read(path)
    assert report['successful'] and report['family'] == family['key'] and report['stage'] == stage
    assert report['source_sha256'] == family['teacher']['sha256'] and report['no_truth_arrays_read']
    if stage == 'anchor':
        assert report['all3_anchor_predictions_byte_identical']
    return dict(audit_sha256=sha(path), audit_guard_sha256=guardsha, audit_path=str(path), guard_path=str(receipt))

def one_family(family, jobs):
    control = native(family, 'control', 1)
    anchor = native(family, 'anchor', 1)
    # Preserve both complete receipts even if one endpoint fails.
    exits = [control.wait(), anchor.wait()]
    assert exits == [0, 0], (family['key'], exits)
    anchor_proof = audit(family, 'anchor')
    remaining = native(family, 'remaining', jobs)
    assert remaining.wait() == 0
    remaining_proof = audit(family, 'remaining')
    result = dict(anchor=anchor_proof, remaining=remaining_proof)
    write(HERE/'pipeline_launches_v1'/(family['key']+'_COMPLETE_v1.json'), result)
    return family['key'], result

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-review', type=Path, required=True)
    ap.add_argument('--root-review-sha256', required=True)
    ap.add_argument('--jobs-per-family', type=int, choices=[4], default=4)
    ap.add_argument('--fit-after-capture', action='store_true')
    args = ap.parse_args()
    global ROOT_REVIEW_SHA
    ROOT_REVIEW_SHA = args.root_review_sha256
    assert sha(args.root_review) == ROOT_REVIEW_SHA
    review = read(args.root_review)
    assert review['successful'] and review['capture_authorized']
    assert review['capture_ready_sha256'] == READY and review['controller_sha256'] == sha(__file__)
    assert sha(PORT/'guard_linux_v7.py') == GUARD
    ready_path = HERE/'capture_plans_v1/CAPTURE_READY_v1.json'
    assert sha(ready_path) == READY
    ready = read(ready_path)
    assert ready['prereg_sha256'] == PREREG and ready['source_manifest']['sha256'] == MANIFEST
    assert sha(HERE/'PREREG_v1.json') == PREREG and sha(HERE/'PARENT_SOURCES_v1.json') == MANIFEST
    qualified = HERE/'parent_qualification_v1/QUALIFIED_v1.json'
    assert sha(qualified) == review['parent_qualification_sha256']
    assert read(qualified)['successful'] and read(qualified)['all_six_official_cold_warm_arithmetic_identical']
    for rec in ready['harness']+ready['control_harness']+[ready['hook']]:
        assert sha(ROOT/rec['path']) == rec['sha256']
    for tag in ('on_policy_parent_qualify_v1', 'on_policy_capture_prepare_v1'):
        guardcheck(RUNS/(tag+'_GUARD_v1.json'), 1)
    fit_recipe = read(HERE/'SECOND_FIT_RECIPE_v1.json')
    assert sha(HERE/'capture_support_v1/audit_onpolicy_capture_v1.py') == fit_recipe['capture_auditor_sha256']
    if args.fit_after_capture:
        assert review['fixed_fit_authorized'] and review['fitter_sha256'] == fit_recipe['fitter_sha256']
        assert sha(HERE/'fit_onpolicy_second_step_v1.py') == fit_recipe['fitter_sha256']
        guardcheck(RUNS/'on_policy_second_fit_selftest_v1_GUARD_v1.json', 1)
    assert len(ready['outputs']) == 6 and ready['ceiling_predictions'] == 558
    assert not (HERE/'CAPTURE_FINAL_v1.json').exists()
    (HERE/'pipeline_launches_v1').mkdir(exist_ok=False)
    results, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one_family, family, args.jobs_per_family): family['key'] for family in ready['outputs']}
        for future in concurrent.futures.as_completed(futures):
            try:
                key, result = future.result()
                results[key] = result
            except BaseException as error:
                failures.append(dict(family=futures[future], error=repr(error)))
                print(json.dumps(dict(utc=utc(), failed=failures[-1])), flush=True)
    final = dict(utc=utc(), successful=not failures, families=results, failures=failures,
                 captured_predictions=540 if not failures else None, unhooked_controls=18 if not failures else None,
                 ceiling_predictions=558, ready_sha256=READY, prereg_sha256=PREREG, source_manifest_sha256=MANIFEST,
                 controller_sha256=sha(__file__), root_review_sha256=ROOT_REVIEW_SHA,
                 no_truth_arrays_read=True, no_candidate_export=True, no_native_validation=True, no_auto_fire=True)
    write(HERE/'CAPTURE_FINAL_v1.json', final)
    assert not failures, failures
    if args.fit_after_capture:
        receipt = RUNS/'on_policy_head90_second_fit_v1_GUARD_v1.json'
        assert not receipt.exists()
        script = HERE/'fit_onpolicy_second_step_v1.py'
        cmd = [PY, '-B', str(PORT/'guard_linux_v7.py'), '--receipt', str(receipt), '--state-dir', str(RUNS/'admission'),
               '--max-active', '36', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--', PY, '-B', str(script)]
        child = start(cmd, 'on_policy_head90_second_fit_v1', 1, dict(fitter_sha256=sha(script), capture_final_sha256=sha(HERE/'CAPTURE_FINAL_v1.json')))
        assert child.wait() == 0
        guardsha = guardcheck(receipt, 1)
        fitted = RUNS/'on_policy_head90_second_fit_v1/FITTED_v1.json'
        report = read(fitted)
        write(HERE/'PIPELINE_FINAL_v1.json', dict(utc=utc(), successful=True, status=report['status'], passed=report['passed'],
              fitted_sha256=sha(fitted), fit_guard_sha256=guardsha, no_candidate_export=True, no_native_launch=True, no_auto_fire=True))
    print(json.dumps(dict(utc=utc(), capture_successful=True, optional_fit_completed=args.fit_after_capture)), flush=True)

if __name__ == '__main__':
    main()
