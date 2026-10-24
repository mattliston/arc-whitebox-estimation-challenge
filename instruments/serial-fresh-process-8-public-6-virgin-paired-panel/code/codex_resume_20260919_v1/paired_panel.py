"""Serial fresh-process 8+6 paired panel, no fitting or candidate changes."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NETS = (35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505)
ARMS = {'baseline': HERE.parent/'fable_wall_repair_v1/cand_c2_wallrepair_I95.py',
        'scalar': HERE/'scalar_distill_I95.py'}


def main():
    out = HERE/'paired14_v1'
    out.mkdir(exist_ok=False)
    for rank in NETS:
        for arm, candidate in ARMS.items():
            target = out/f'{arm}_{rank}'
            command = [sys.executable, str(HERE/'run_guarded.py'), '--record',
                       str(out/f'{arm}_{rank}_watch.json'), '--seconds', '100',
                       str(RUNTIME), '-W', 'ignore', str(HERE/'native_screen.py'),
                       str(candidate), str(target), '--ranks', str(rank)]
            with (out/f'{arm}_{rank}.log').open('x') as log:
                p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            assert p.returncode == 0, (arm, rank, p.returncode)
            watch = json.loads((out/f'{arm}_{rank}_watch.json').read_text())
            assert watch['completed'] and watch['whole_group_retired']
            record = json.loads((target/'result.json').read_text())['records'][0]
            print(json.dumps(dict(arm=arm, rank=rank, flops=record['flops'],
                                  wall_s=record['wall_s'], max_rss_bytes=watch['max_rss_bytes'])), flush=True)
    with (out/'completed.json').open('x') as stream:
        json.dump(dict(completed=True, networks=list(NETS), calls=28, truth_accessed=False), stream)
    print('PAIRED_PANEL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
