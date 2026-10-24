"""Stdlib verification of the completed additive no-op control and its pins."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / 'rscov_additive_arm_manifest_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for path, expected in manifest['source_pins'].items():
        assert sha(path) == expected, ('Changed pinned source/reference', path)
    return manifest


def control_reference(comparison_path):
    manifest = verify_pins()
    comparison_path = Path(comparison_path).resolve()
    comparison_path.relative_to(HERE)
    control_manifest_path = Path(manifest['additive_control_manifest'])
    control_manifest = json.loads(control_manifest_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    assert comparison['completed'] is True
    assert comparison['worker'] == control_manifest['worker']
    assert comparison['run'] == control_manifest['run']
    assert comparison['comparator'] == control_manifest['comparator']
    assert comparison['compare_args'] == []
    table = comparison['comparison']
    assert table['status'] == 'PASS' and table['no_arm'] is True
    assert table['parity'] == 'all16 MSE and final1024 FP64 bytes'
    assert table['diagnostic_layers'] == 7
    control_path = Path(control_manifest['result'])
    control = json.loads(control_path.read_text())
    sq = json.loads(Path(control_manifest['sq_result']).read_text())
    base = json.loads(Path(control_manifest['t134_result']).read_text())
    assert control['rank'] == sq['rank'] == base['rank'] == 7100584
    assert control['params'] == sq['params'] == base['params']
    assert control['mse'] == sq['mse'] == base['mse']
    assert len(control['mse_layers']) == 16
    assert control['mse_layers'] == sq['mse_layers'] == base['mse_layers']
    assert len(control['final_prediction']) == len(sq['final_prediction']) == 1024
    actual = struct.pack('<1024d', *control['final_prediction'])
    assert actual == struct.pack('<1024d', *sq['final_prediction'])
    assert hashlib.sha256(actual).hexdigest() == table['prediction_sha256']
    telemetry_path = Path(control_manifest['telemetry'])
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert table['telemetry'] == str(telemetry_path) and table['rows'] == rows
    assert [row['layer'] for row in rows] == list(range(8, 15))
    assert all(row['request_id'] == control['id'] and row['no_arm'] and row['rdc'] == 2.0 for row in rows)
    return dict(comparison_artifact=str(comparison_path), comparison_sha256=sha(comparison_path),
                control_manifest=str(control_manifest_path), control_manifest_sha256=sha(control_manifest_path),
                control_result=str(control_path), control_result_sha256=sha(control_path),
                telemetry=str(telemetry_path), telemetry_sha256=sha(telemetry_path),
                prediction_sha256=table['prediction_sha256'],
                parity='all16 MSE and final1024 FP64 bytes', status='PASS')
