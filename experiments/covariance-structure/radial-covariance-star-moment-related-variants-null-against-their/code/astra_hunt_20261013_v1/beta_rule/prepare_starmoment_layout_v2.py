"""Stdlib-only worker-directory preparation; no request or source edits."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'starmoment_manifest_v2.json').read_text())
    requests = [(manifest['control_request'], manifest['control_request_sha256'])]
    requests += [(row['arm_request'], row['arm_request_sha256'])
                 for row in manifest['rows']]
    for filename, expected in requests:
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    for name in ('run_starmoment_control_v2', 'run_starmoment_arm_v2'):
        run = HERE / name
        assert run.is_dir() and (run / 'queue').is_dir()
        for directory in ('claimed', 'results'):
            path = run / directory
            assert not path.exists() or not list(path.iterdir()), path
            path.mkdir(exist_ok=True)
    print('V2 worker directories ready; all nine queued request pins unchanged.')


if __name__ == '__main__':
    main()
