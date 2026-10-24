"""Root-only stdlib post-screen report; no evaluation and no frozen-pin edits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def check(comparison_path):
    # The imported gate helper contains only stdlib verification and has no
    # import-time numerical work; that imported source is in the frozen pins.
    from rscov_additive_gate_v1 import control_reference, sha, verify_pins
    manifest = verify_pins()
    old = json.loads(Path(manifest['original_rscov_manifest']).read_text())
    comparison = json.loads(comparison_path.read_text())
    assert comparison['completed'] is True, 'Arm comparison did not complete'
    assert comparison['worker'] == manifest['worker']
    assert comparison['run'] == manifest['run']
    assert comparison['comparator'] == manifest['comparator']
    args = comparison['compare_args']
    assert len(args) == 2
    pin_path = Path(args[0]).resolve()
    pin_path.relative_to(HERE)
    assert sha(pin_path) == args[1], 'Control-reference hash mismatch'
    reference = json.loads(pin_path.read_text())
    assert control_reference(reference['comparison_artifact']) == reference
    assert comparison['comparison']['control_reference'] == reference
    assert comparison['comparison']['control_reference_pin_sha256'] == args[1]
    rows, pairs = [], []
    for source in old['rows']:
        claimed = Path(manifest['run']) / 'claimed' / (source['arm_id'] + '.w0.json')
        assert sha(claimed) == source['arm_request_sha256']
        result = json.loads((Path(manifest['run']) / 'results' / (source['arm_id'] + '.json')).read_text())
        base = json.loads(Path(source['base']).read_text())
        assert result['rank'] == base['rank'] == source['seed']
        assert result['params'] == base['params']
        assert len(result['mse_layers']) == len(base['mse_layers']) == 16
        values = [*result['mse_layers'], *base['mse_layers']]
        assert all(math.isfinite(value) and value >= 0 for value in values), 'Invalid layer MSE'
        assert all(math.isfinite(value) and value > 0 for value in (result['mse'], base['mse'])), 'Invalid final log-ratio MSE'
        mismatch = [layer for layer in range(8) if result['mse_layers'][layer] != base['mse_layers'][layer]]
        rows.append(dict(seed=source['seed'], log_percent=100*math.log(result['mse']/base['mse']),
                         early_layers_exact=not mismatch, early_mismatched_layers=mismatch))
        pairs.append((result, base))
    assert len(rows) == 8 and [row['seed'] for row in rows] == list(range(7100584, 7100592))
    logs = [row['log_percent'] for row in rows]
    mean = sum(logs)/8
    se = math.sqrt(sum((value-mean)**2 for value in logs)/7/8)
    valid = all(row['early_layers_exact'] for row in rows)
    layer_numerators = [sum(result['mse_layers'][layer] for result,base in pairs) for layer in range(16)]
    layer_denominators = [sum(base['mse_layers'][layer] for result,base in pairs) for layer in range(16)]
    return dict(status='VALID' if valid else 'INVALID_EARLY_LAYER_MISMATCH',
                validity=valid, completed_seeds=8,
                exact_layers0_through7_all8=valid,
                mean_log_percent=mean, se_percent=se,
                pooled_percent=100*(sum(result['mse'] for result,base in pairs)/sum(base['mse'] for result,base in pairs)-1),
                improved=sum(value < 0 for value in logs), per_network=rows,
                pooled_layers_percent=[100*(numerator/denominator-1) if denominator > 0 else None
                                       for numerator,denominator in zip(layer_numerators, layer_denominators)],
                zero_baseline_layer_denominators=[layer for layer,value in enumerate(layer_denominators) if value == 0],
                threshold_mean_log_percent=old['threshold_mean_log_percent'],
                passes_predeclared_gate=valid and mean <= old['threshold_mean_log_percent'],
                control_parity=reference['parity'],
                control_reference_pin_sha256=args[1],
                source_comparison=str(comparison_path), source_comparison_sha256=sha(comparison_path),
                activation_telemetry='not recorded in frozen arm; fallback/near-identity attribution unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    comparison = args.comparison.resolve()
    comparison.relative_to(HERE)
    if args.out is not None:
        args.out = args.out.resolve()
        args.out.relative_to(HERE)
        assert not args.out.exists(), 'Output must be fresh'
    try:
        report = check(comparison)
    except Exception as error:
        report = dict(status='INVALID_OR_INCOMPLETE', validity=False,
                      reason=type(error).__name__ + ': ' + str(error),
                      source_comparison=str(comparison))
    report['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out is not None:
        with args.out.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['validity'] else 1


if __name__ == '__main__':
    sys.exit(main())
