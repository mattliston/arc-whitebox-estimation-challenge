"""Report paired adjusted gains against the current best graded submission.

The frozen v4a qualification evidence is retained separately; this reporter
does not authorize submission and does not change any original release gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

from compare_relative_v1 import compare

HERE = Path(__file__).resolve().parent

def report(candidate):
    baseline = json.loads((HERE/'BEST_GRADED_REFERENCE_v1.json').read_text())
    reference = Path(baseline['science_run'])
    source = Path(baseline['science_source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == baseline['science_sha256']
    out = compare(candidate.resolve(), reference)
    out.pop('science_bar_cleared')
    out['reference_submission_id'] = baseline['submission_id']
    out['reference_platform_score'] = baseline['platform_score']
    out['interpretation'] = 'Paired local32 gain versus current graded best; this is not a measured platform gain.'
    old = compare(candidate.resolve(), HERE/'science32_rel_v4a_m15')
    out['frozen_v4a_qualification'] = {k:old[k] for k in ['all32','heldout25','science_bar_cleared']}
    out['submission_authorized'] = False
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = report(args.candidate)
    print('Reference: best graded #%s, platform score %.15g' %
          (result['reference_submission_id'], result['reference_platform_score']))
    for name in ['seven','heldout25','all32']:
        row = result[name]
        print('%s: adjusted %+.6f%% (SE %.6f, lower %s/%s)' %
              (name, row['adjusted_pct'], row['adjusted_se'], row['lower'],row['n']))
    if args.output:
        assert not args.output.exists(), args.output
        args.output.write_text(json.dumps(result, indent=2)+'\n')
