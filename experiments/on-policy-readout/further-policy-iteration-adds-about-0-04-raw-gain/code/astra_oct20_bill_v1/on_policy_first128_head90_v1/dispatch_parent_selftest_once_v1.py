"""Execute the single exact reviewed synthetic self-test after live slot checks."""
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
review_path, recipe_path = HERE/'ROOT_SELFTEST_REVIEW_v1.json', HERE/'PARENT_SELFTEST_RECIPE_v1.json'
assert sha(review_path) == '9918d52210456f07249fc85e65e5dd9b4c51b4913f75da37232c97935c556011'
assert sha(recipe_path) == '07df3aced8358801646d90669eb31b92139376f7b9fc55e43f43eaf9f4d09e86'
review, recipe = read(review_path), read(recipe_path)
assert review['successful'] and review['one_selftest_authorized'] and review['no_parent_build_authorized']
assert review['recipe_sha256'] == sha(recipe_path)
assert review['command'] == recipe['command'] and review['environment'] == recipe['environment']
for name, digest in recipe['files'].items():
    assert sha(ROOT/name) == digest, name
ops_path = L/'OP1_PARENT_SELFTEST_SMALL_SLOT_CLEARANCE_v1.json'
assert sha(ops_path) == '13611ea4954c2be019c67e398849ec5e08fd0d1144808aa9f5edc0804a2581d4'
ops = read(ops_path)
assert ops['root_review_sha256'] == sha(review_path) and ops['max_capture'] == 8 and ops['max_small'] == 1
assert ops['allowed_guard_receipt'] == recipe['guard_receipt']
assert ops['no_parent_build_fit_capture_truth_or_native_authorized']
ps = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
parsed = [(line, [Path(word).name for word in line.split()[1:]]) for line in ps]
assert not any('build_parent_sources_v1.py' in words for _, words in parsed)
servers = [line for line, words in parsed if 'server_linux_v7.py' in words]
guards = [line for line, words in parsed if 'guard_linux_v7.py' in words]
assert len(servers) <= 8 and len(guards) <= 8
for line in guards:
    words = line.split(); child = words[words.index('--')+1:]
    assert any(Path(word).name == 'run_group_capture_v1.py' for word in child), line
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
result = dict(utc=utc(), pid=process.pid, startticks=start, command=recipe['command'], environment=recipe['environment'], script_sha256=sha(HERE/'build_parent_sources_v1.py'), launcher_sha256=sha(__file__), root_review_sha256=sha(review_path), recipe_sha256=sha(recipe_path), ops_clearance_sha256=sha(ops_path), preflight=preflight, no_build_native_capture_truth_or_fit=True, no_retry=True)
with launch.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps(dict(pid=process.pid,startticks=start,launch=str(launch),launch_sha256=sha(launch),utc=result['utc'])),flush=True)
