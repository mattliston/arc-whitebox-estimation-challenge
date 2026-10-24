"""Fresh arm entry: pin a completed additive control before the unchanged worker."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from rscov_additive_gate_v1 import HERE, control_reference, sha, verify_pins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-comparison', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    artifact.relative_to(HERE)
    assert not artifact.exists()
    manifest = verify_pins()
    reference = control_reference(args.control_comparison)
    run = Path(manifest['run'])
    assert not list((run / 'claimed').iterdir()) and not list((run / 'results').iterdir())
    assert {str(path) for path in (run / 'queue').glob('*.json')} == set(manifest['arm_requests'])
    for path, expected in manifest['arm_requests'].items():
        assert sha(path) == expected
    pin_path = artifact.with_suffix('.control_reference.json')
    with pin_path.open('x') as handle:
        json.dump(reference, handle, indent=2, allow_nan=False)
        handle.write('\n')
    sys.argv = [str(HERE / 'screen_entry_v2.py'),
                '--worker', manifest['worker'], '--run', str(run), '--index', '0',
                '--comparator', manifest['comparator'], '--artifact', str(artifact),
                '--compare-arg', str(pin_path), '--compare-arg', sha(pin_path)]
    runpy.run_path(str(HERE / 'screen_entry_v2.py'), run_name='__main__')


if __name__ == '__main__':
    main()
