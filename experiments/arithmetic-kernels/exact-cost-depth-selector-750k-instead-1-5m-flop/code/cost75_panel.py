"""One frozen candidate on the same14 nets; reuse its sealed rank35 and100 cold calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NETS = (35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505)
CANDIDATE_SHA = '5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c'


def main():
    assert json.loads((HERE/'cost75_dev_score.json').read_text())['development_gate_passed']
    out = HERE/'cost75_panel_v1'
    out.mkdir(exist_ok=False)
    candidate = HERE/'cost_rule75_scalar.py'
    records = []
    for rank in NETS:
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == CANDIDATE_SHA
        if rank == 35:
            target = HERE/'cost75_dev_v1'
            watch_path = HERE/'cost75_dev_watch.json'
        elif rank == 100:
            target = HERE/'cost75_cold100_v1'
            watch_path = HERE/'cost75_cold100_watch.json'
        else:
            target = out/f'candidate_{rank}'
            watch_path = out/f'candidate_{rank}_watch.json'
            command = [sys.executable, str(HERE/'run_guarded.py'), '--record', str(watch_path),
                       '--seconds', '100', str(RUNTIME), '-W', 'ignore', str(HERE/'native_screen.py'),
                       str(candidate), str(target), '--ranks', str(rank)]
            with (out/f'candidate_{rank}.log').open('x') as log:
                done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            assert done.returncode == 0, (rank, done.returncode)
        watch = json.loads(watch_path.read_text())
        assert watch['completed'] and watch['whole_group_retired']
        receipt = json.loads((target/'result.json').read_text())
        assert receipt['candidate_sha256'] == CANDIDATE_SHA and not receipt['truth_accessed']
        call = receipt['records'][0]
        assert call['rank'] == rank and call['includes_initialization']
        assert hashlib.sha256(Path(call['output']).read_bytes()).hexdigest() == call['output_sha256']
        records.append(dict(rank=rank, result=str(target/'result.json'), watch=str(watch_path), call=call))
        print(json.dumps(dict(rank=rank, flops=call['flops'], wall_s=call['wall_s'],
                              max_rss_bytes=watch['max_rss_bytes'])), flush=True)
    with (out/'completed.json').open('x') as f:
        json.dump(dict(completed=True, records=records, new_calls=12,
                       candidate_sha256=CANDIDATE_SHA, truth_accessed=False), f, indent=2)


if __name__ == '__main__':
    main()
