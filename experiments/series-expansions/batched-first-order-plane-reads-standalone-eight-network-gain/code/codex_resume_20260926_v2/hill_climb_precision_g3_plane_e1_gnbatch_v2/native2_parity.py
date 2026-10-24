"""Saved TRAIN2 all-16 byte parity against E1; root-only, no estimator or targets.
Run NATIVE -B native2_parity.py ENABLED_NATIVE_CONFIG_SHA under 120s/7.5GB.
This receipt and its successful retired guard are required before boundary admission.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
ENGINE_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
CANDIDATE_SHA = '06538586dd306df2b00986f43cc85afd39d515a12c8977f4da6fa9c539ed91b1'
BUILD_SHA = '332ca00b0a8bc2d80c2a50fd5339d2e5dd1786296e1c0c79a12f0ec6e63ce751'
OLD_CONFIG = {'path': '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_v1/native_screen.json', 'sha256': '01899419158e3320da12b41fd96e75057999728cec1c217c8d039c701e1e991d', 'bytes': 23913}
OLD_RECEIPT = {'path': '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_v1/native2_assessment.json', 'sha256': 'bdc4056b3290939f65d8a7e471abeb899d1b201fdd492e909bd8236100131452', 'bytes': 16088}
OLD_GUARD = {'path': '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_v1/native2_assess_guard.json', 'sha256': '152ea74d48f6ffa0dc94610ff98118003937f06bb921e5ad24ca9bd6b0084212', 'bytes': 1092}


def main():
    assert len(sys.argv) == 2
    engine = H2/'native_screen_engine_v1/screen.py'
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == ENGINE_SHA
    spec = importlib.util.spec_from_file_location('gnbatch_parity_screen', engine)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    assert Path(sys.executable).absolute() == q.NATIVE and q.np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in q.THREADS)
    cp, c = q.configure(HERE/'native_screen.json', sys.argv[1])
    assert c['arms']['candidate']['source'] == q.pin(HERE/'candidate.py')
    assert c['arms']['candidate']['source']['sha256'] == CANDIDATE_SHA
    assert q.pin(HERE/'build.json')['sha256'] == BUILD_SHA
    target = HERE/'native2_parity.json'; assert not target.exists()
    try:
        prior_pin = q.pin(HERE/'native2_assessment.json')
        prior = json.loads(q.checked(prior_pin))
        assert prior['passed_integrity'] is True and prior['admitted'] is True
        assert prior['n'] == 2 and prior['ranks'] == [35, 100]
        assert prior['config'] == cp and prior['engine'] == c['engine']
        assert prior['candidate'] == c['arms']['candidate']['source']
        q.guard(c['stages']['2']['assessment_guard'], c['stages']['2']['assessment_limits'],
                [str(q.NATIVE), '-B', str(engine), cp['path'], cp['sha256'], '2'])
        for record in prior['evidence']: q.checked(record)
        old = json.loads(q.checked(OLD_RECEIPT))
        oc = json.loads(q.checked(OLD_CONFIG)); q.checked(OLD_GUARD)
        assert old['passed_integrity'] is True and old['admitted'] is True and old['n'] == 2
        assert old['config'] == OLD_CONFIG and old['engine'] == c['engine']
        assert old['candidate'] == oc['arms']['candidate']['source']
        assert old['candidate']['sha256'] == '6ac194efda6491b48f86eca074493c503f9312c6fd9e667105f378ee7637d6f9'
        q.guard(OLD_GUARD['path'], oc['stages']['2']['assessment_limits'],
                [str(q.NATIVE), '-B', str(engine), OLD_CONFIG['path'], OLD_CONFIG['sha256'], '2'])
        parent = copy.deepcopy(oc['arms']['candidate'])
        parent['blocks']['2']['artifacts'] = old['blocks'][0]['arms']['candidate']
        child_rows, child, child_pins = q.panel(c['arms']['candidate'], 2, 'candidate')
        assert child_pins == prior['blocks'][0]['arms']['candidate']
        parent_rows, saved, parent_pins = q.panel(parent, 2, 'parity_parent')
        for rank in (35, 100):
            assert child[rank].tobytes() == saved[rank].tobytes(), ('all16_bytes', rank)
            assert next(x['mse'] for x in child_rows if x['rank'] == rank) == next(x['mse'] for x in parent_rows if x['rank'] == rank)
        for record in list(q.EVIDENCE.values()): q.checked(record)
        result = dict(passed=True, receipt_checks_passed=True, n=2, ranks=[35,100],
            candidate=c['arms']['candidate']['source'], parent=parent['source'], assessor=q.pin(__file__),
            config=cp, engine=c['engine'], native2=prior_pin, parent_native2=OLD_RECEIPT,
            parent_native_config=OLD_CONFIG, parent_assessment_guard=OLD_GUARD,
            child=child_pins, saved_parent=parent_pins, all16_parent_bytes_exact=True,
            all16_predictions_finite=True, same_mse=True,
            bills_required_equal=False, evidence=list(q.EVIDENCE.values()),
            required_before_boundary=True, no_parent_six_acquisition=True,
            targets_read=False, model_executed=False, release_qualified=False)
    except Exception as exc:
        q.once(target, dict(passed=False, receipt_checks_passed=False, candidate=c['arms']['candidate']['source'],
            assessor=q.pin(__file__), config=cp, error=dict(type=type(exc).__name__,message=str(exc))))
        raise
    q.once(target, result)
    print(json.dumps(dict(passed=True, n=2, all16_parent_bytes_exact=True)))


if __name__ == '__main__':
    main()
