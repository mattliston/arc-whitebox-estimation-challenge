"""Root-authorized target-free analysis, under unchanged v7 resource guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PORT = HERE.parents[1] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import thread_env

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, choices=(35, 150), required=True)
    a = p.parse_args()
    assert sys.platform == 'linux'
    script = HERE / 'analyze_old_dv_v3.py'
    assert sha(script) == '5b77120c76daf3293aafbe568762102ec36cff979a573d9cac8963f566d8b55a'
    guard = PORT / 'guard_linux_v7.py'
    assert sha(guard) == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
    if a.rank == 35:
        capture = Path('/home/arcworker/arc-runs/current_old_dv_capture_v1')
        receipt = capture / 'part01/old_dv_capture_v1/SALVAGED_CAPTURE_v1.json'
        assert sha(receipt) == '457be375a42b22d1a0bf17b3eab2d2e3e7340ddd8dbaca7399920f55638533b2'
    else:
        capture = Path('/home/arcworker/arc-runs/current_old_dv_capture_rank150_v2')
        receipt = capture / 'part01/old_dv_capture_v2/CAPTURE_COMPLETE_v1.json'
        assert json.loads(receipt.read_text())['complete']
        assert json.loads((capture / 'COMPLETE.json').read_text())['successful']
        g = json.loads((capture / 'part01_GUARD.json').read_text())
        assert g['successful'] and g['child_exit'] == 0 and not g['cleanup_errors']
        assert g['max_owned_swap_bytes'] == 0 and g['whole_group_retired']
        pred = capture / 'part01/diagnostic_pred_150.bin'
        ref = Path('/home/arcworker/arc-runs/msg8_loaderfix337422_control32_markline_blas4_v1/part01/pred_150.bin')
        assert pred.read_bytes() == ref.read_bytes()
    out = Path('/home/arcworker/arc-runs/current_old_dv_analysis_rank%d_v1' % a.rank)
    assert not out.exists()
    judge = '/home/arcworker/arc-envs/judge/bin/python'
    cmd = [judge, '-B', str(guard), '--receipt', str(out) + '_GUARD.json',
           '--state-dir', '/home/arcworker/arc-runs/admission', '--max-active', '36',
           '--cgroup-root', '/sys/fs/cgroup/arc-estimator-v5', '--',
           judge, '-B', str(script), '--capture', str(capture), '--out', str(out), '--rank', str(a.rank)]
    env = dict(os.environ, **thread_env(2), ARC_BLAS_THREADS='2')
    log = Path(str(out) + '.log')
    with log.open('x') as f:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(utc=utc(), rank=a.rank, pid=proc.pid, command=cmd,
                  source_sha256=sha(script), guard_sha256=sha(guard), evidence_sha256=sha(receipt),
                  out=str(out), log=str(log), truth_access=False, native_predictor_launches=0,
                  source_bytes_preserved=True, root_authorized=True)
    with (HERE / ('ANALYSIS_LAUNCH_RANK%d_v1.json' % a.rank)).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
