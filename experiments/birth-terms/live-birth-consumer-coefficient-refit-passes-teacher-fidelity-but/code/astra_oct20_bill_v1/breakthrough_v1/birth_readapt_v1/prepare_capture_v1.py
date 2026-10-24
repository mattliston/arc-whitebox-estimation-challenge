"""Pin the approved four-network birth-consumer diagnostic; never launches it."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
V = HERE.parents[2] / 'astra_oct20_vm_frontier_v1'
K = HERE.parents[2] / 'fable_oct19_kc_v1'
PORT = K / 'agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import thread_env

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def rec(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    assert sys.platform == 'linux'
    base = V / 'PLAN_BANK8_ACTUAL_SHARED_v1.json'
    assert sha(base) == 'cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
    source = K / 'candidate_micro_copyall_ecut_eh14_astra_sci_v1.py'
    assert sha(source) == '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded'
    assert sha(PORT / 'guard_linux_v7.py') == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
    current = RUNS / 'full_state_response_train4_v1'
    assert json.loads((current / 'COMPLETE.json').read_text())['successful']
    old = RUNS / 'teacher_capture_v1'
    checked = json.loads((old / 'VERIFIED_BY_BILL_v1.json').read_text())
    assert checked['complete'] and checked['diagnostic_only'] and not checked['network_truth_read']
    baseline = []
    for rank in (35, 150, 200, 250):
        part = old / ('part01' if rank != 250 else 'part02')
        assert json.loads((part / 'manifest.json').read_text())['sha256'] == sha(source)
        f = part / ('pred_%d.bin' % rank)
        assert f.stat().st_size == 65536
        prior = next(x for x in checked['predictions'] if x['rank'] == rank)
        assert sha(f) == prior['sha256']
        baseline.append(dict(rank=rank, path=str(f), sha256=sha(f),
                             diagnostic_only=True, same_source=True))
    teacher = dict(rec(source), effective_sha256=sha(source))
    names = ('client_capture_v1.py', 'run_group_capture_v1.py', 'run_capture_v1.py',
             'fragment_mark_complete_line_v1.py')
    for name in names + ('birth_capture_hook_v1.py',):
        ast.parse((HERE / name).read_text())
    recipe = dict(utc=utc(), kind='ORIGINAL16_BIRTH_CONSUMER_TRAIN4', teacher=teacher,
                  capture_ranks=[35, 150, 200, 250], layers=list(range(1, 15)), rows_per_layer=1024,
                  hook=rec(HERE / 'birth_capture_hook_v1.py'), harness=[rec(HERE / n) for n in names],
                  prereg=rec(HERE / 'PREREG_v1.json'), observer_test=rec(HERE / 'OBSERVER_SELFTEST_v2.json'),
                  target_free=True, diagnostic_only=True, uses_network_truth=False,
                  reference_submission=337422, numerical_teacher='Original complete16-message09fb trajectory',
                  parity_controls=baseline, current_state_capture=str(current), root_authorized=True)
    plan = json.loads(base.read_text())
    plan.update(utc=utc(), kind=recipe['kind'], diagnostic_only=True,
                groups=[[35, 150], [200, 250]], first7=[35, 150, 200, 250], held25=[],
                rows={str(r): {'weights': plan['rows'][str(r)]['weights']} for r in (35, 150, 200, 250)},
                candidates={'teacher': teacher})
    config = RUNS / 'birth_consumer_train4_config_v1'
    config.mkdir(exist_ok=False)
    write(config / 'PLAN_v1.json', plan)
    write(config / 'RECIPE_v1.json', recipe)
    env = dict(os.environ, **thread_env(1), ARC_BLAS_THREADS='1',
               ARC_BIRTH_CONSUMER_RECIPE=str(config / 'RECIPE_v1.json'),
               ARC_BIRTH_CONSUMER_RECIPE_SHA=sha(config / 'RECIPE_v1.json'))
    client = '/home/arcworker/arc-envs/client/bin/python'
    run = subprocess.run([client, '-B', str(HERE / 'client_capture_v1.py'), '--help'],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    subprocess.run([client, '-c', "import importlib.util;assert importlib.util.find_spec('numpy') is None"],
                   env=env, check=True)
    result = dict(utc=utc(), successful=True, source_sha256=sha(source),
                  plan=str(config / 'PLAN_v1.json'), plan_sha256=sha(config / 'PLAN_v1.json'),
                  recipe=str(config / 'RECIPE_v1.json'), recipe_sha256=sha(config / 'RECIPE_v1.json'),
                  hook_sha256=sha(HERE / 'birth_capture_hook_v1.py'),
                  harness_sha256=sha(HERE / 'run_capture_v1.py'), client_preflight='PASS',
                  numpy_free=True, native_launches=0, truth_access=False, original_guard_unchanged=True)
    write(HERE / 'CAPTURE_READY_v1.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
