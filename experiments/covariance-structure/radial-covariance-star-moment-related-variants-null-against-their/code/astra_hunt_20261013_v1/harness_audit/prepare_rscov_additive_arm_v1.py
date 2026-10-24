"""Stdlib source-only pinning; does not alter existing models or arm requests."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    additive_path = HERE / 'additive_control_manifest_v1.json'
    additive = json.loads(additive_path.read_text())
    original_path = HERE / 'rscov_manifest_v1.json'
    original = json.loads(original_path.read_text())
    pins = dict(additive['source_pins'])
    pins[str(additive_path)] = sha(additive_path)
    pins[str(HERE / 'allow_rscov_v1.txt')] = sha(HERE / 'allow_rscov_v1.txt')
    names = ('rscov_additive_gate_v1.py', 'screen_entry_rscov_additive_arm_v1.py',
             'compare_rscov_additive_arm_v1.py', 'launch_rscov_additive_arm_v1.sh',
             'prepare_rscov_additive_arm_v1.py')
    for name in names:
        path = HERE / name
        if path.suffix == '.py':
            ast.parse(path.read_text())
        pins[str(path)] = sha(path)
    for row in original['rows']:
        for field in ('base', 'base_request'):
            pins[row[field]] = row[field + '_sha256']
        assert sha(row['arm_request']) == row['arm_request_sha256']
    for path, expected in pins.items():
        assert sha(path) == expected, path
    run = HERE / 'run_rscov_arm_v1'
    assert not list((run / 'claimed').iterdir()) and not list((run / 'results').iterdir())
    manifest = dict(version=1, status='source_only_prepared_not_run',
                    run=str(run), worker=str(HERE / 'worker_astra_v1.py'),
                    runtime=str(HERE / 'otr_cut_runtime_astra_v1.py'),
                    entry=str(HERE / 'screen_entry_rscov_additive_arm_v1.py'),
                    comparator=str(HERE / 'compare_rscov_additive_arm_v1.py'),
                    additive_control_manifest=str(additive_path),
                    original_rscov_manifest=str(original_path),
                    original_control_result_path_intentionally_unused=original['control_result'],
                    control_gate='completed additive comparison plus independent all16-MSE/FP64-byte recheck, pinned at launch and rechecked after arm',
                    rule=original['rule'], threshold_mean_log_percent=original['threshold_mean_log_percent'],
                    arm_requests={row['arm_request']: row['arm_request_sha256'] for row in original['rows']},
                    source_pins=pins)
    out = HERE / 'rscov_additive_arm_manifest_v1.json'
    with out.open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status='PREPARED_SOURCE_ONLY', manifest=str(out), sha256=sha(out), pins=len(pins))))


if __name__ == '__main__':
    main()
