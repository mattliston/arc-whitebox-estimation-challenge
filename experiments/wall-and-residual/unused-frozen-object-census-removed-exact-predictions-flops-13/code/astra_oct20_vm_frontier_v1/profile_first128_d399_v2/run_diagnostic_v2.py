"""Sequential cold control/observer, each inside the unchanged original VM guard.

Review-only until root authorizes this frozen plan. No truths, score, or fire.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
PORT = HERE.parents[1] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import THREAD_KEYS, sha, utc, verified, write_new


def main():
    recipe = json.loads((HERE / 'RECIPE_v1.json').read_text())
    for record in recipe['files']:
        verified(ROOT, record)
    plan = HERE / 'PLAN_LINUX_v1.json'
    assert sha(plan) == recipe['plan_sha256']
    assert sha(PORT / 'guard_linux_v7.py') == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
    output = RUNS / 'profile_first128_d399_v2'
    output.mkdir(exist_ok=False)
    env = dict(os.environ, **{key: '4' for key in THREAD_KEYS}, ARC_BLAS_THREADS='4',
               ARC_D399_CPROFILE_RECIPE_SHA=sha(HERE / 'RECIPE_v1.json'))
    commands = []
    for mode in ('control', 'observe'):
        guard_receipt = output / (mode + '_GUARD_v1.json')
        command = ['/home/arcworker/arc-envs/judge/bin/python', '-B', str(PORT / 'guard_linux_v7.py'),
                   '--receipt', str(guard_receipt), '--state-dir', str(RUNS / 'admission'),
                   '--max-active', '36', '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--',
                   '/home/arcworker/arc-envs/judge/bin/python', '-B', str(HERE / 'run_group_profile_v2.py'),
                   '--root', str(ROOT), '--plan', str(plan), '--candidate', 'first128_dcr_slr_release',
                   '--ranks', '35', '--out', str(output / mode), '--setup', 'paced',
                   '--client-python', '/home/arcworker/arc-envs/client/bin/python',
                   '--server-python', '/home/arcworker/arc-envs/server/bin/python']
        commands.append(dict(mode=mode, command=command, utc=utc()))
        with (output / (mode + '.log')).open('x') as log:
            result = subprocess.run(command, env=dict(env, ARC_D399_CPROFILE_MODE=mode),
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt = json.loads(guard_receipt.read_text())
        assert result.returncode == 0 and receipt['reason'] is None and receipt['child_exit'] == 0
        assert receipt['whole_group_retired'] and not receipt['cleanup_errors']
        assert not receipt['max_owned_swap_bytes']
    rows = {mode: json.loads((output / mode / 'diagnostic_rows.jsonl').read_text())
            for mode in ('control', 'observe')}
    predictions = {mode: output / mode / 'diagnostic_pred_35.bin' for mode in rows}
    assert predictions['control'].read_bytes() == predictions['observe'].read_bytes()
    assert rows['control']['flops'] == rows['observe']['flops']
    assert rows['control']['operations'] == rows['observe']['operations']
    for key in ('requests','free_trips','freed_handles'):
        assert rows['control'][key] == rows['observe'][key], (key, rows['control'][key], rows['observe'][key])
    diagnostic=json.loads((output/'observe/candidate_cprofile_v1.json').read_text())
    control_profile=json.loads((output/'control/candidate_cprofile_v1.json').read_text())
    assert diagnostic['started'] and diagnostic['enabled'] and diagnostic['rows']
    assert control_profile['started'] and not control_profile['enabled'] and not control_profile['rows']
    for profile in (diagnostic,control_profile):
        assert profile['stack_empty'] and profile['sdk_unmodified'] and profile['gc_policy_unmodified'] and profile['profiler_restored']
    for row in rows.values():
        assert row['fast_rescue_stats'].get('calls', 0) == 0
    write_new(output / 'COMPLETE_v1.json', dict(utc=utc(), diagnostic_only=True, successful=True,
        source_sha256=recipe['candidate']['effective_sha256'], recipe_sha256=sha(HERE / 'RECIPE_v1.json'),
        commands=commands, byte_exact=True, identical_flops=True, identical_operations=True, identical_requests_and_frees=True, profile_sha256=sha(output/'observe/candidate_cprofile_v1.json'), control_profile_sha256=sha(output/'control/candidate_cprofile_v1.json'),
        flops=rows['control']['flops'], prediction_sha256=sha(predictions['control']),
        no_truth_read=True, timing_is_not_a_release_gate=True))
    print(json.dumps(dict(utc=utc(), successful=True, output=str(output))))


if __name__ == '__main__':
    main()
