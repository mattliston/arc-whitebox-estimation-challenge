"""HELD fixed TRAIN32 late-only gradients; no optimizer and no CHECK access."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
H = D.parent
plan = json.loads((D / 'PLAN_v1.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pincheck():
    for path, expected in plan['source_pins'].items():
        assert sha(path) == expected, path
    for path, expected in plan['truth_pins'].items():
        assert sha(path) == expected, path
    # Only argparse's permitted rank list changes in v2. Exact execution body,
    # current physics, late clone initialization and backward scope are pinned.
    one = (H / 'run_late_encoder_v1.py').read_text()
    two = (H / 'run_late_encoder_v2.py').read_text()
    assert two.replace('choices=[35, 200, *range(7100268, 7100300)]',
                       'choices=[35, 200, *range(7100268, 7100276)]') == one


def completed(rank, path, guard_path, reused=False):
    value = json.loads((path / 'COMPLETE_v1.json').read_text())
    guard = json.loads(guard_path.read_text())
    assert value['rank'] == rank and value['mode'] == 'grad'
    assert value['no_optimizer'] and not value['parameters_updated']
    assert guard['completed'] and guard['whole_group_retired']
    assert guard['exit_code'] == 0 and not guard['error'] and not guard['remaining_rows']
    assert guard['min_pressure_free_pct'] >= 50
    assert max(guard['max_rss_bytes'], guard['max_footprint_bytes'], value['peak_rss']) < 9_000_000_000
    assert sha(value['result']) == value['result_sha256']
    assert value['current_private_sources'] == plan['current_private_sources']
    assert value['original_encoder_hashes'] == plan['original_encoder_hashes']
    return dict(rank=rank, result=value['result'], result_sha256=value['result_sha256'],
                complete=str(path / 'COMPLETE_v1.json'), guard=str(guard_path),
                guard_sha256=sha(guard_path), real_highwater=value['peak_rss'],
                mse=value['mse'], seconds=value['seconds'], reused_exact_scope=reused)


pincheck()
assert plan['ranks'] == list(range(7100268, 7100300))
env = dict(os.environ, TRAINER_THREADS='1', TRAINER_PL='1', TRAINER_CARD='M',
           TRAINER_RELIEF='1', TRAINER_STAR_ATTACH='0', TRAINER_BIRTH_ATTACH='0')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    env[key] = '1'
rows = []
started = time.monotonic()
try:
    proof = json.loads(Path(plan['reuse268']['proof']).read_text())
    assert proof['zero_update_all16_mse_exact'] and proof['zero_update_final_mse_exact']
    assert proof['source_scope_sha256'] == sha(H / 'late_encoder_scope_v1.py')
    assert proof['runner_sha256'] == sha(H / 'run_late_encoder_v1.py')
    rows.append(completed(7100268, Path(plan['reuse268']['directory']),
                          Path(plan['reuse268']['guard']), True))
    for rank in plan['ranks'][1:]:
        pincheck()
        output = D / f'gradient_{rank}_v1'
        guard = D / f'GUARD_{rank}_v1.json'
        command = [plan['launcher_python'], '-B', plan['guard'],
                   '--record', str(guard), '--seconds', '180',
                   '--footprint-limit', '9000000000', '--rss-backstop', '9000000000',
                   '/usr/bin/nice', '-n', '10', plan['torch_python'], '-B',
                   str(H / 'run_late_encoder_v2.py'), '--rank', str(rank),
                   '--mode', 'grad', '--output', str(output),
                   '--w3module', plan['w3module']]
        write(D / f'COMMAND_{rank}_v1.json', command)
        print(json.dumps(dict(event='START', rank=rank)), flush=True)
        with (D / f'LOG_{rank}_v1.txt').open('x') as log:
            result = subprocess.run(command, cwd=plan['repository'], env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (rank, result.returncode)
        row = completed(rank, output, guard)
        rows.append(row)
        print(json.dumps(dict(event='COMPLETE', rank=rank, seconds=row['seconds'],
                              highwater=row['real_highwater'])), flush=True)
    pincheck()
    write(D / 'GRADIENT_MANIFEST_v1.json', dict(
        status='FIXED_TRAIN32_COMPLETE', ranks=plan['ranks'], rows=rows,
        source_pins=plan['source_pins'], original_encoder_hashes=plan['original_encoder_hashes'],
        gradient_keys=plan['gradient_keys'], gradient_scale=100000000.0,
        native_branch_sha256=plan['native_branch_sha256'],
        parameters_updated=False, optimizer=False, CHECK_read=False,
        offline_guard_bytes=9000000000, elapsed_seconds=time.monotonic() - started))
    print('TRAIN32_COMPLETE', flush=True)
except BaseException as error:
    write(D / 'STOP_v1.json', dict(error=repr(error), completed=[row['rank'] for row in rows],
                                  elapsed_seconds=time.monotonic() - started,
                                  optimizer=False, parameters_updated=False))
    raise
