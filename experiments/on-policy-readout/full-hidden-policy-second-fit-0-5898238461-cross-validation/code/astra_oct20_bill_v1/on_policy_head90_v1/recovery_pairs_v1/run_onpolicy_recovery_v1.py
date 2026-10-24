"""Fixed three-wave capture recovery after the priority native stage retires.

Two families per wave, unchanged four groups per family. Reuse only original
clean anchors; replay all 87 remaining networks in fresh directories. No retry.
The fixed scientific fit is optional and separately authorized by root review.
"""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
A = ORIGINAL.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1'
PORT = A/'cloud_linux_capacity_v7'
CAPTURE = ORIGINAL.parent/'truth_response95_v1'
sys.path.insert(0, str(PORT))
from common_linux_v7 import thread_env
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
PY = '/home/arcworker/arc-envs/judge/bin/python'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
READY = 'f2224fa44d4d7dea2a302ea66a43dd7f5c16f32e3433441df7dc2d56c9e99d21'
PLAN = '916e5730b5f22b32e603a1208488b0151817818be8c1dfeeec014793af914fd0'
PREREG = '0c94b0dc84bd17f0542df28e9ae56d7486c8944fe3cd08db74e6c91a76c24443'
MANIFEST = 'd6eaeb233f5f8109247d46cc68672848fb10477bd18a4076829fd267a4613136'
ORIGINAL_READY = '71e56701238bd6ff1919c78977eb6a5351710fca8566ba57086a82e259fb43a4'
ORIGINAL_AUDITOR = '4c96b9faa54048b229cdfbf96849deb0c006780178e3b4fe7bfb6f0062407856'
WAVES = (('fold0', 'fold1'), ('fold2', 'fold3'), ('fold4', 'full90'))

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
        assert g['cgroup_events_initial'][name] == g['cgroup_events_final'][name]
    return sha(path)

def native_slot_empty():
    # A preflight contract check, not an adaptive admission or runtime governor.
    lines = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    found = [line for line in lines if any(Path(word).name == 'server_linux_v7.py' for word in line.split()[1:])]
    assert not found, ('competing native stage still active', found)

def priority_native_proof(review):
    stage = review['priority_native_stage']
    custody = stage['root_custody_review']
    custody_path = ROOT/custody['path']
    assert sha(custody_path) == custody['sha256'] and read(custody_path)['successful']
    out = Path(stage['out'])
    assert out.is_relative_to(RUNS)
    complete, manifest = read(out/'COMPLETE.json'), read(out/'RUN_MANIFEST.json')
    assert complete['successful'] and not complete['pending'] and len(complete['results']) == 12
    assert all(code == 0 for code in complete['results'].values())
    assert complete['plan_sha256'] == manifest['plan_sha256'] == stage['plan_sha256']
    assert manifest['candidate_sha256'] == stage['source_sha256']
    assert manifest['groups'] == list(range(1, 13)) and not manifest['warmup_each']
    assert manifest['blas_threads'] == complete['blas_threads'] == 4
    guards = [dict(path=str(out/('part%02d_GUARD.json'%i)), sha256=guardcheck(out/('part%02d_GUARD.json'%i), 4)) for i in range(1, 13)]
    ranks = []
    for index in range(1, 13):
        part = read(out/('part%02d'%index)/'manifest.json')
        assert part['sha256'] == stage['source_sha256']
        assert part['environment']['thread_provenance']['blas_threads'] == 4
        ranks.extend(part['ranks'])
    assert len(ranks) == len(set(ranks)) == 32
    native_slot_empty()
    return dict(stage=stage, complete_sha256=sha(out/'COMPLETE.json'), manifest_sha256=sha(out/'RUN_MANIFEST.json'),
                root_custody_review_sha256=sha(custody_path), guards=guards,
                all12_original_guards_retired=True, all32_source_manifests_verified=True)

def original_anchor(family, ready):
    key = family['key']
    pin = ready['recovery']['anchor_proofs'][key]
    path = Path(pin['audit_path'])
    assert sha(path) == pin['audit_sha256']
    report = read(path)
    assert report['successful'] and report['family'] == key and report['stage'] == 'anchor'
    assert report['source_sha256'] == family['teacher']['sha256'] == pin['source_sha256']
    assert report['ready_sha256'] == ORIGINAL_READY and report['prereg_sha256'] == PREREG
    assert report['source_manifest_sha256'] == MANIFEST and report['auditor_sha256'] == ORIGINAL_AUDITOR
    assert report['all3_anchor_predictions_byte_identical'] and report['no_truth_arrays_read']
    assert guardcheck(pin['guard_path'], 1) == pin['guard_sha256']
    for guard in report['guards']:
        assert guardcheck(guard['path'], 4) == guard['sha256']
    assert len(report['prediction_pairs']) == len(report['predictions']) == 3 and len(report['files']) == 45
    for pair, prediction in zip(report['prediction_pairs'], report['predictions']):
        control, capture = Path(pair['control_path']), Path(prediction['path'])
        assert pair['seed'] == prediction['seed'] and control.stat().st_size == capture.stat().st_size == 65536
        assert control.read_bytes() == capture.read_bytes()
        assert sha(capture) == pair['prediction_sha256'] == prediction['sha256']
    for item in report['files']:
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    return dict(audit_path=pin['audit_path'], audit_sha256=pin['audit_sha256'],
                guard_path=pin['guard_path'], audit_guard_sha256=pin['guard_sha256'], original_clean_anchor_reused=True)

def start(command, tag, threads, evidence):
    log = RUNS/(tag+'.log')
    launch = HERE/'pipeline_launches_v1'/(tag+'_LAUNCH_v1.json')
    assert not log.exists() and not launch.exists()
    with log.open('x') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=dict(os.environ, **thread_env(threads), ARC_BLAS_THREADS=str(threads)))
    record = dict(utc=utc(), pid=process.pid, command=command, log=str(log), evidence=evidence,
                  controller_sha256=sha(__file__), root_review_sha256=ROOT_REVIEW_SHA, no_auto_fire=True)
    write(launch, record)
    print(json.dumps(dict(utc=record['utc'], tag=tag, pid=process.pid)), flush=True)
    return process

def one_family(family, ready):
    anchor_proof = original_anchor(family, ready)
    key = family['key']
    item = family['outputs']['remaining']
    out, plan, recipe = Path(item['out']), ROOT/item['plan']['path'], ROOT/item['recipe']['path']
    assert not out.exists() and sha(plan) == item['plan']['sha256'] and sha(recipe) == item['recipe']['sha256']
    assert len(read(plan)['groups']) == 29 and sum(map(len, read(plan)['groups'])) == 87
    runner = CAPTURE/'run_capture_v1.py'
    assert sha(runner) == '8e6a23c406c039dfeff34f8b6b135b03c6cdd40d05f586f7b6a9dee9bdb83aca'
    command = [PY, '-B', str(runner), '--recipe', str(recipe), '--root', str(ROOT), '--plan', str(plan),
               '--candidate', 'teacher', '--out', str(out), '--state-dir', str(RUNS/'admission'),
               '--client-python', '/home/arcworker/arc-envs/client/bin/python',
               '--server-python', '/home/arcworker/arc-envs/server/bin/python', '--jobs', '4',
               '--max-active', '36', '--blas-threads', '4', '--setup', 'paced', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5']
    child = start(command, 'onpolicy_head90_'+key+'_remaining_recovery_v1', 4,
                  dict(plan_sha256=sha(plan), recipe_sha256=sha(recipe), source_sha256=family['teacher']['sha256']))
    assert child.wait() == 0, key
    receipt = Path(item['audit_guard_path'])
    auditor = HERE/'audit_onpolicy_recovery_v1.py'
    assert not receipt.exists()
    command = [PY, '-B', str(PORT/'guard_linux_v7.py'), '--receipt', str(receipt), '--state-dir', str(RUNS/'admission'),
               '--max-active', '36', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--', PY, '-B', str(auditor),
               '--family', key, '--stage', 'remaining', '--ready-sha256', READY]
    child = start(command, 'onpolicy_head90_'+key+'_remaining_recovery_audit_v1', 1,
                  dict(auditor_sha256=sha(auditor), ready_sha256=READY))
    assert child.wait() == 0, key
    guardsha = guardcheck(receipt, 1)
    report_path = out/'AUDIT_v1.json'
    report = read(report_path)
    assert report['successful'] and report['family'] == key and report['remaining_uses_same_source_anchor_proof']
    assert report['no_truth_arrays_read'] and report['source_sha256'] == family['teacher']['sha256']
    result = dict(anchor=anchor_proof, remaining=dict(audit_path=str(report_path), audit_sha256=sha(report_path),
                  guard_path=str(receipt), audit_guard_sha256=guardsha))
    write(HERE/'pipeline_launches_v1'/(key+'_COMPLETE_v1.json'), result)
    return key, result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-review', type=Path, required=True)
    parser.add_argument('--root-review-sha256', required=True)
    parser.add_argument('--fit-after-capture', action='store_true')
    args = parser.parse_args()
    global ROOT_REVIEW_SHA
    ROOT_REVIEW_SHA = args.root_review_sha256
    assert sha(args.root_review) == ROOT_REVIEW_SHA
    assert args.root_review.resolve() == (HERE/'ROOT_RECOVERY_REVIEW_v1.json').resolve()
    review = read(args.root_review)
    assert review['successful'] and review['capture_authorized'] and review['execution_slot_ceiling'] == 630
    assert review['completed_prediction_ceiling'] == 606 and review['max_native_groups'] == 8
    assert review['controller_sha256'] == sha(__file__) and review['recovery_plan_sha256'] == PLAN
    assert review['capture_ready_sha256'] == READY and sha(HERE/'RECOVERY_READY_v1.json') == READY
    assert sha(HERE/'RECOVERY_PLAN_v1.json') == PLAN
    plan, ready = read(HERE/'RECOVERY_PLAN_v1.json'), read(HERE/'RECOVERY_READY_v1.json')
    assert plan['waves'] == [list(wave) for wave in WAVES] and plan['jobs_per_family'] == 4
    assert sha(PORT/'guard_linux_v7.py') == GUARD
    assert sha(ORIGINAL/'PREREG_v1.json') == PREREG and sha(ORIGINAL/'PARENT_SOURCES_v1.json') == MANIFEST
    assert sha(ORIGINAL/'capture_plans_v1/CAPTURE_READY_v1.json') == ORIGINAL_READY
    assert sha(ORIGINAL/'CAPTURE_FINAL_v1.json') == ready['recovery']['original_capture_final_sha256']
    assert not read(ORIGINAL/'CAPTURE_FINAL_v1.json')['successful']
    assert sha(ROOT/plan['root_anchor_custody_review_path']) == plan['root_anchor_custody_review_sha256']
    assert read(ROOT/plan['root_anchor_custody_review_path'])['successful']
    for item in ready['harness']+ready['control_harness']+[ready['hook']]:
        assert sha(ROOT/item['path']) == item['sha256']
    qualified = ORIGINAL/'parent_qualification_v1/QUALIFIED_v1.json'
    assert sha(qualified) == review['parent_qualification_sha256'] and read(qualified)['successful']
    for tag in ('on_policy_parent_qualify_v1', 'on_policy_capture_prepare_v1'):
        guardcheck(RUNS/(tag+'_GUARD_v1.json'), 1)
    recipe_path = HERE/'SECOND_FIT_RECIPE_v1.json'
    assert sha(recipe_path) == review['fit_recipe_sha256']
    recipe = read(recipe_path)
    assert sha(HERE/'audit_onpolicy_recovery_v1.py') == recipe['capture_auditor_sha256'] == review['auditor_sha256']
    assert sha(HERE/'fit_onpolicy_second_step_recovery_v1.py') == recipe['fitter_sha256']
    if args.fit_after_capture:
        assert review['fixed_fit_authorized'] and review['fitter_sha256'] == recipe['fitter_sha256']
        guardcheck(RUNS/'on_policy_second_fit_selftest_v1_GUARD_v1.json', 1)
        mathproof = read(HERE/'UNCHANGED_FIT_MATH_v1.json')
        assert mathproof['successful'] and mathproof['recovery_fitter_sha256'] == recipe['fitter_sha256']
        assert mathproof['original_fitter_sha256'] == sha(ORIGINAL/'fit_onpolicy_second_step_v1.py')
    assert len(ready['outputs']) == 6 and ready['ceiling_predictions'] == 558
    assert not (HERE/'CAPTURE_FINAL_v1.json').exists()
    assert not (RUNS/'on_policy_head90_second_fit_recovery_v1').exists()
    assert not (RUNS/'on_policy_head90_second_fit_v1').exists()
    for family in ready['outputs']:
        assert sha(ROOT/family['teacher']['path']) == family['teacher']['sha256']
        item = family['outputs']['remaining']
        assert not Path(item['out']).exists() and not Path(item['audit_guard_path']).exists()
        original_anchor(family, ready)
    priority = priority_native_proof(review)
    (HERE/'pipeline_launches_v1').mkdir(exist_ok=False)
    write(HERE/'PRIORITY_NATIVE_COMPLETION_v1.json', dict(utc=utc(), successful=True, **priority))
    families = {family['key']: family for family in ready['outputs']}
    results, failures = {}, []
    for index, wave in enumerate(WAVES, 1):
        native_slot_empty()
        print(json.dumps(dict(utc=utc(), wave=index, families=wave, fixed_max_native_groups=8)), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(one_family, families[key], ready): key for key in wave}
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                except BaseException as error:
                    failures.append(dict(family=futures[future], error=repr(error)))
                    print(json.dumps(dict(utc=utc(), failed=failures[-1])), flush=True)
        if failures:
            break
    final = dict(utc=utc(), successful=not failures and len(results) == 6, families=results, failures=failures,
                 not_completed=[key for key in families if key not in results],
                 captured_predictions=540 if not failures else None, unhooked_controls=18 if not failures else None,
                 ceiling_predictions=558, completed_prediction_ceiling_including_discarded=606,
                 attempted_prediction_slot_ceiling_including_interrupted=630,
                 discarded_complete_failed_predictions=48, failed_partials_not_reused=True,
                 ready_sha256=READY, prereg_sha256=PREREG, source_manifest_sha256=MANIFEST,
                 controller_sha256=sha(__file__), root_review_sha256=ROOT_REVIEW_SHA,
                 no_truth_arrays_read=True, no_candidate_export=True, no_native_validation=True, no_auto_fire=True)
    write(HERE/'CAPTURE_FINAL_v1.json', final)
    assert final['successful'], failures
    if args.fit_after_capture:
        native_slot_empty()
        tag = 'on_policy_head90_second_fit_recovery_v1'
        receipt = RUNS/(tag+'_GUARD_v1.json')
        assert not receipt.exists()
        script = HERE/'fit_onpolicy_second_step_recovery_v1.py'
        command = [PY, '-B', str(PORT/'guard_linux_v7.py'), '--receipt', str(receipt), '--state-dir', str(RUNS/'admission'),
                   '--max-active', '36', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--', PY, '-B', str(script)]
        child = start(command, tag, 1, dict(fitter_sha256=sha(script), capture_final_sha256=sha(HERE/'CAPTURE_FINAL_v1.json')))
        assert child.wait() == 0
        guardsha = guardcheck(receipt, 1)
        fitted = RUNS/tag/'FITTED_v1.json'
        report = read(fitted)
        write(HERE/'PIPELINE_FINAL_v1.json', dict(utc=utc(), successful=True, status=report['status'], passed=report['passed'],
              fitted_sha256=sha(fitted), fit_guard_sha256=guardsha, no_candidate_export=True, no_native_launch=True, no_auto_fire=True))
    print(json.dumps(dict(utc=utc(), capture_successful=True, optional_fit_completed=args.fit_after_capture)), flush=True)

if __name__ == '__main__':
    main()
