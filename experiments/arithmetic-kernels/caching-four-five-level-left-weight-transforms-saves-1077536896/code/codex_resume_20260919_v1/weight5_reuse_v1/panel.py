"""Serial cold checks of the fixed14 networks, authenticating saved Read5 outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
PARENT = D / 'read5_packet_v1'
RUNTIME = D.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NETS = (35, 100, 150, 200, 250, 300, 400, 500, 3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
EXPECTED = 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert read(HERE / 'native_comparison.json')['native_gate_passed']
    assert read(HERE / 'boundary_comparison.json')['boundary_gate_passed']
    parent = {r['rank']: r for r in read(PARENT / 'panel/completed.json')['records']}
    out = HERE / 'panel'
    out.mkdir(exist_ok=False)
    candidate = HERE / 'candidate.py'
    records = []
    for rank in NETS:
        assert sha(candidate) == EXPECTED
        if rank == 35:
            target, watch_path = HERE / 'dev', HERE / 'dev_watch.json'
        elif rank == 100:
            target, watch_path = HERE / 'cold100', HERE / 'cold100_watch.json'
        else:
            target = out / f'candidate_{rank}'
            watch_path = out / f'candidate_{rank}_watch.json'
            command = [sys.executable, str(D / 'run_guarded.py'), '--record', str(watch_path),
                       '--seconds', '100', str(RUNTIME), '-B', '-W', 'ignore', str(HERE / 'native.py'),
                       str(candidate), str(target), '--ranks', str(rank)]
            with (out / f'candidate_{rank}.log').open('x') as log:
                done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            assert done.returncode == 0, (rank, done.returncode)
        watch = read(watch_path)
        assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
        receipt = read(target / 'result.json')
        assert receipt['candidate_sha256'] == EXPECTED and not receipt['truth_accessed']
        call = receipt['records'][0]
        old = parent[rank]['call']
        assert call['rank'] == rank and call['includes_initialization']
        assert sha(call['output']) == sha(old['output']) == call['output_sha256'] == old['output_sha256']
        assert call['weights_sha256'] == old['weights_sha256']
        cache = call['weight5_cache']
        assert cache['completed'] and cache['numeric_cache_retired']
        saving = cache['removed_flops'] - cache['copied_flops']
        assert saving > 0 and old['flops'] - call['flops'] == saving
        row = dict(rank=rank, result=str(target / 'result.json'), watch=str(watch_path), call=call,
                   parent_result=parent[rank]['result'], all16_byte_exact=True,
                   saved_flops=saving, flop_ratio=call['flops'] / old['flops'])
        records.append(row)
        with (out / f'verified_{rank}.json').open('x') as stream:
            json.dump(row, stream, indent=2)
        print(json.dumps(dict(rank=rank, flops=call['flops'], saved_flops=saving, all16_byte_exact=True,
                              wall_s=call['wall_s'], max_rss_bytes=watch['max_rss_bytes'])), flush=True)
    with (out / 'completed.json').open('x') as stream:
        json.dump(dict(completed=True, records=records, new_calls=12,
                       candidate_sha256=EXPECTED, truth_accessed=False), stream, indent=2)


if __name__ == '__main__':
    main()
