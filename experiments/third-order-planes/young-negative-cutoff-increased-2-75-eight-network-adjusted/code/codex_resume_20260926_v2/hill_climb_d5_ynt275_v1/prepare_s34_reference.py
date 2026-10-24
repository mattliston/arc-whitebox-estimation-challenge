"""Source-only S34 reference refresh config; root --enable after review."""
from pathlib import Path
import copy
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_e1_ynt_psdb_d5_v1'


def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def once(p, obj):
    with p.open('x') as f: json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def main():
    assert sys.argv[1:] in ([], ['--enable'])
    enabled = sys.argv[1:] == ['--enable']
    bank = json.loads((H/'official_bank_S34_v1.json').read_text())
    assert pin(H/'official_bank_S34_v1.json')['sha256'] == '8b2cba1ddc5381b4ce70948107511bb888fe86fb3f8766d80abac4417f37d0e2'
    root_selection = pin(D/'s34_reference_root_admission.json')
    assert root_selection['sha256'] == '001ee5b05c07cb60cc1bcd9f2935d219f06839f68f3b4471c477fc05036b1949'
    original = json.loads((D/'native_screen_continuation.json').read_text())
    c = dict(schema='s34-residual-reference-v1', prospective_authorization=enabled,
        engine=pin(D/'boundary_assess_s34_reference.py'), candidate=pin(D/'candidate.py'), parent=pin(P/'candidate.py'),
        original_assessor=pin(D/'boundary_assess_official_limit.py'), original_config=pin(D/'boundary_config.json'),
        original_failure=pin(D/'boundary_assessment_extension_v2.json'),
        original_assessment_guard=pin(D/'boundary_assess_guard_extension_v2.json'),
        parent_boundary=pin(P/'boundary_assessment_extension_v2.json'), s34_bank=pin(H/'official_bank_S34_v1.json'),
        root_selection=root_selection,
        s34_evaluation=bank['evaluation_receipt'], native_config_path=str(D/'native_screen_s34_reference.json'),
        policy='One outcome-informed reassessment justified by newly graded current winner S34. Same first candidate and parent censuses, unchanged4x multiplier, strict0.4 residual cap. Original S32 wall/CPU/setup/memory checks remain unchanged. No timing/model rerun or retry; original failure remains false; no broad/serving authority.')
    c['provenance'] = [pin(__file__), c['engine'], *[v for v in c.values() if isinstance(v, dict) and 'sha256' in v]]
    boundary_path = D/('boundary_s34_reference_config.json' if enabled else 'boundary_s34_reference_config_draft.json')
    once(boundary_path, c)
    nc = copy.deepcopy(original)
    nc['prospective_authorization'] = enabled
    nc['stages']['2']['result'] = str(D/'native2_s34_reference_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_s34_reference_assess_guard.json')
    req = nc['boundary_before8']
    req['assessor'] = c['engine']
    req['receipt'] = str(D/'boundary_s34_reference_assessment.json')
    req['assessment_guard'] = str(D/'boundary_s34_reference_assess_guard.json')
    req['assessment_command'] = [str(H.parents[4]/'.venv/bin/python'), '-B', c['engine']['path'],
        str(D/'boundary_s34_reference_config.json'), pin(boundary_path)['sha256']]
    nc['provenance'] += [pin(__file__), c['engine'], pin(boundary_path), pin(D/'native_screen_continuation.json'),
        pin(D/'native2_continuation_assessment.json'), pin(D/'native2_continuation_assess_guard.json'),
        c['original_failure'], c['original_assessment_guard'], c['s34_bank'], c['s34_evaluation']]
    nc['interpretation'] = 'Saved-N2 reassessment for a new current-winner S34 residual-reference receipt. Original pilot/continuation/firstS32failure remain immutable. Same arms, source, predictions, driver, fixed ranks, parity and score gates. No candidate or control timing rerun. Only a genuine passing separate reference receipt permits original native8 assessment; no broad/bank authority.'
    nc['prospective_scope'] = 'saved_native2_reassessment_then_s34_reference_assessment_then_fixed6_and_native8'
    assert all(nc[k] == original[k] for k in ('arms', 'engine', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert all(nc['stages'][s]['gates'] == original['stages'][s]['gates'] for s in ('2', '8'))
    output = D/('native_screen_s34_reference.json' if enabled else 'native_screen_s34_reference_draft.json')
    once(output, nc)
    print(json.dumps(dict(enabled=enabled, engine=c['engine'], boundary_config=pin(boundary_path), native_config=pin(output), builder=pin(__file__))))


if __name__ == '__main__': main()
