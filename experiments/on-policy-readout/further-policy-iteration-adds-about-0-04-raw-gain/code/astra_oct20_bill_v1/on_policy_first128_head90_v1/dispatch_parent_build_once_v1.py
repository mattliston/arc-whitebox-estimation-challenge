"""Execute the single exact reviewed six-parent build after live slot checks."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path('/home/arcworker/arc-workspace')
HERE = Path(__file__).resolve().parent
RUNS = Path('/home/arcworker/arc-runs')
L = HERE.parent.parent/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
flags = lambda p: {line.split()[0]: int(line.split()[1]) for line in p.read_text().splitlines()}
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
assert sys.platform == 'linux'
review_path, recipe_path = HERE/'ROOT_BUILD_REVIEW_v1.json', HERE/'PARENT_BUILD_RECIPE_v1.json'
assert sha(review_path) == '6d52b6f76b09b9feb6684e6868e8e0fffdec858a8772066cc3432465b15c6a2a'
assert sha(recipe_path) == '53fee698c0496c3ffcd676559a609c3fc101a62d52c76eddc529e3c094d39fbd'
review, recipe = read(review_path), read(recipe_path)
assert review['successful'] and review['one_parent_build_authorized']
assert review['recipe_sha256'] == sha(recipe_path)
assert review['command'] == recipe['command'] and review['environment'] == recipe['environment']
for name, digest in recipe['files'].items():
    assert sha(ROOT/name) == digest, name
ops_path = L/'OP1_PARENT_BUILD_TWO_NAMED_SMALL_CLEARANCE_v1.json'
assert sha(ops_path) == '8c771ea36a0f74acf5c6e6c82f8eb932880b79f2c22547c591ca3cf214a94b94'
ops = read(ops_path)
assert ops['root_build_review_sha256'] == sha(review_path) and ops['two_named_small_only']
assert ops['permitted_new_guard_receipt'] == recipe['guard_receipt']
assert ops['static_transition_sha256'] == review['static_transition_sha256'] == '1b522abb8321c8b7859955f020fde724d70edc55439636762b30d106e84d0e54'
assert sha(HERE/'PARENT_SELFTEST_FINAL_v1.json') == review['selftest_final_sha256']
previous = read(RUNS/'onpolicy_first128_parent_selftest_v1_GUARD_v1.json')
assert previous['successful'] and previous['whole_group_retired'] and previous['child_exit'] == 0 and not previous['cleanup_errors']
assert all(v == 0 for v in previous['cgroup_events_final'].values()) and previous['max_owned_swap_bytes'] == 0
ps = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
parsed = [(line, [Path(word).name for word in line.split()[1:]]) for line in ps]
assert not any('build_parent_sources_v1.py' in words for _, words in parsed)
servers = [line for line, words in parsed if 'server_linux_v7.py' in words]
guards = [line for line, words in parsed if 'guard_linux_v7.py' in words]
assert len(servers) <= 8 and len(guards) <= 13
counts = dict(capture_or_audit=0, truth=0, old_fit=0)
for line in guards:
    words = line.split(); child = [Path(w).name for w in words[words.index('--')+1:]]
    receipt_name = words[words.index('--receipt')+1]
    if 'run_group_capture_v1.py' in child:
        assert any(receipt_name.startswith(str(RUNS/('onpolicy_head90_'+key+'_remaining_recovery_v1'))+'/part') for key in ('fold4','full90'))
        counts['capture_or_audit'] += 1
    elif 'audit_onpolicy_recovery_v1.py' in child:
        assert receipt_name in ops['old_family_audit_guards_inside_eight_family_slots']
        counts['capture_or_audit'] += 1
    elif ops['initial4_truth_worker'] in child:
        assert receipt_name.startswith(ops['initial4_truth_guard_root']+'/')
        counts['truth'] += 1
    elif 'fit_onpolicy_second_step_recovery_v1.py' in child:
        assert receipt_name == ops['permitted_original_automatic_fit_guard']
        counts['old_fit'] += 1
    else: raise AssertionError(line)
assert counts['capture_or_audit'] <= 8 and counts['truth'] <= 4 and counts['old_fit'] <= 1
cg = Path('/sys/fs/cgroup/arc-estimator-v5')
ev, local = flags(cg/'memory.events'), flags(cg/'memory.events.local')
assert ev['max'] == 8033 and local['max'] == 128
assert all(ev[key] == 0 for key in ('oom', 'oom_kill', 'oom_group_kill'))
assert int((cg/'memory.swap.current').read_text()) == 0
live = {}
for p in cg.glob('arc-*'):
    try:
        if not (p/'cgroup.procs').read_text().strip(): continue
        v = flags(p/'memory.events')
        assert all(x == 0 for x in v.values()) and int((p/'memory.swap.current').read_text()) == 0
        live[p.name] = v
    except FileNotFoundError:
        pass
memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'SwapTotal:'))}
assert memory['MemAvailable'] > 25500000000 and memory['SwapTotal'] == 0
receipt, log, launch = map(Path, (recipe['guard_receipt'], recipe['log'], recipe['launch_receipt']))
assert not any(p.exists() for p in (receipt, log, launch, HERE/'parents_v1', HERE/'PARENT_SOURCES_v1.json'))
preflight = dict(utc=utc(), native_servers=servers, guards=guards, memory=memory, parent_events=ev, parent_local_events=local, live_child_events=live, parent_current=int((cg/'memory.current').read_text()))
with log.open('x') as stream:
    process = subprocess.Popen(recipe['command'], cwd=ROOT, env=dict(os.environ, **recipe['environment']), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
stat = Path('/proc/%d/stat'%process.pid).read_text()
start = int(stat[stat.rfind(')')+2:].split()[19])
result = dict(utc=utc(), pid=process.pid, startticks=start, command=recipe['command'], environment=recipe['environment'], script_sha256=sha(HERE/'build_parent_sources_v1.py'), launcher_sha256=sha(__file__), root_review_sha256=sha(review_path), recipe_sha256=sha(recipe_path), ops_clearance_sha256=sha(ops_path), preflight=preflight, parent_source_build_only=True,no_new_native_capture_or_second_fit=True, no_retry=True)
with launch.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps(dict(pid=process.pid,startticks=start,launch=str(launch),launch_sha256=sha(launch),utc=result['utc'])),flush=True)
