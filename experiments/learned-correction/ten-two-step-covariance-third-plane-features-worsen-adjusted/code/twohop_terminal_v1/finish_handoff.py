"""Advance the versioned continuation record after all experiment groups retire."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    close = read(HERE / 'closeout.json')
    assert close['resources']['all_owned_groups_retired']
    for name, expected in close['artifacts_sha256'].items():
        assert sha(HERE / name) == expected, name
    for name, expected in close['external_artifacts_sha256'].items():
        assert sha(D / name) == expected, name
    result = read(HERE / 'information/result.json')
    bank = read(D / 'official_best_331599.json')
    assert bank['grading_status'] == 'graded' and bank['adjusted'] < 4e-9
    passed = result['gate_passed']
    report = 'INFORMATION_RESULT.md' if passed else 'NEGATIVE_RESULT.md'
    disposition = ('The fixed information gate passes; no production candidate is qualified. '
                   'A separate fixed fit, CHECK evaluation and complete native/client/panel qualification are required.'
                   if passed else
                   'The fixed information gate fails and this ten-feature family is closed. '
                   'No CHECK/full fit, operator/layer/penalty sweep, native port, archive or upload follows.')
    section = f'''## 2026-09-19 — official sub-4e-9 bank; terminal two-step information test complete

- AIcrowd #331599 is GRADED: adjusted {bank['adjusted']:.16g}, raw {bank['raw']:.16g}. Official improvement over #331584 is {bank['relative_adjusted_improvement']*100:.6f}%. The4e-9 milestone is confirmed;3e-9,2e-9 and approximately1e-9 remain unmet. Exact approved archive SHA594aa6e9debf9f0e370c29afea9ccb33b475af6d27f7454bbd85abfeb8a5c8f4 was uploaded once. No approval or grading job is pending.
- New fixed twohop_terminal_v1 completed: ten full-matrix covariance/third-plane features, TRAIN20 five-fold network CV, TRAIN-only weighted projection off the existing65-column affine head. Extended/unchanged adjusted ratio {result['extended_to_unchanged']:.12g}; extended/control {result['extended_to_control']:.12g}; {result['fold_wins']}/5 fold wins against both. These are reused research networks, not independent generalization evidence. {disposition}
- All three observed qualification outputs and parent FLOP bills exactly match Lazy; features repeat byte-for-byte. Independent explicit-matrix and weighted-ridge certificates pass. Features cost6489753600F per call, charged with own complete cold parent bills plus2MF readout reserve; no qualified production bill is claimed. Submitted parent and earlier negative results remain unchanged.
-22 full calls and22 guarded stages completed serially, peak {close['resources']['peak_rss_bytes']}B including self-reports, zero added swap; all owned groups retired. Qualification session75767 and TRAIN capture session67262 are TERMINAL, as are the scorer's guarded group and all prior sessions. No numerical/upload workers remain. Stored artifacts before closeout: {close['artifact_bytes_before_closeout']}B.
- See twohop_terminal_v1/{report}, resources.json, source_review.json and closeout.json; official_best_331599.json, sub331599_poll_09.json and lazy_rotation_objective_v1/OFFICIAL_RESULT.md. All211 original Lazy closeout artifacts and four external bindings were preserved. Latest checkpoint continuation_checkpoint_v21.json supersedes the interim v21 handle records. The standing approximately1e-9 goal remains active.

'''
    handoff = ROOT / 'HANDOFF_CURRENT.md'
    prior = handoff.read_text()
    with (D / 'HANDOFF_CURRENT_before_twohop_closeout.md').open('x') as stream:
        stream.write(prior)
    handoff.write_text(section + prior)
    for name in ['RECEIPTS.md', 'REEXAMINATION_INDEX_20260919.md']:
        with (D / name).open('a') as stream:
            stream.write('\n\n' + section)
    checkpoint = read(D / 'continuation_checkpoint_v21_interim.json')
    checkpoint.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      current_handoff_sha256=sha(handoff), numerical_workers_active=False,
                      active_numerical_session=None, upload_worker_active=False,
                      local_work=close['status'] + '; official4e-9 bank confirmed.',
                      prior_checkpoint='continuation_checkpoint_v20.json',
                      prior_checkpoint_sha256=sha(D / 'continuation_checkpoint_v20.json'))
    checkpoint['twohop_terminal'] = dict(status=close['status'], gate_passed=passed,
        extended_to_unchanged=result['extended_to_unchanged'],
        extended_to_control=result['extended_to_control'], fold_wins=result['fold_wins'],
        qualification_passed=True, algebra_certificate_passed=True,
        production_candidate_qualified=False, reference_means_opened=True,
        new_full_prediction_calls=22, all_owned_groups_retired=True,
        peak_rss_bytes=close['resources']['peak_rss_bytes'], additional_swap_bytes=0,
        result=f'twohop_terminal_v1/{report}', closeout='twohop_terminal_v1/closeout.json',
        closeout_sha256=sha(HERE / 'closeout.json'))
    checkpoint['official_objective_gap'] = read(D / 'official_lazy_objective_gap_v1.json')
    with (D / 'continuation_checkpoint_v21.json').open('x') as stream:
        json.dump(checkpoint, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'checkpoint':'continuation_checkpoint_v21.json',
                      'checkpoint_sha256':sha(D / 'continuation_checkpoint_v21.json'),
                      'handoff_sha256':sha(handoff), 'closeout_sha256':sha(HERE / 'closeout.json')}))


if __name__ == '__main__':
    main()
