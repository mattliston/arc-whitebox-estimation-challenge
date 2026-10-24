from common import (HERE, D, P2, CANDIDATE, CANDIDATE_SHA, PARENT_SHA,
                    NETS, read, sha, pin, write, verify_pin, verify_call)

assert sha(CANDIDATE) == CANDIDATE_SHA
counts = {}
check = read(D / 'continuation_checkpoint_v60_verified.json')
for name in ('checkpoint', 'handoff', 'index', 'closeout'):
    verify_pin(check[name])
for relative in ('terminal_third_covariance_sketch_v1/closeout.json',
                 'full_lower_moment_support_v1/closeout.json',
                 'full_lower_moment_objective_v1/closeout.json'):
    obj = read(D / relative)
    count = 0
    for key in ('artifacts', 'external_pins', 'own_source_pins', 'sources'):
        items = obj.get(key, [])
        if isinstance(items, dict):
            items = items.values()
        for item in items:
            if isinstance(item, dict) and 'path' in item and 'sha256' in item:
                verify_pin(item)
                count += 1
    counts[relative] = count
prior_score = read(D / 'full_lower_moment_objective_v1/dev_score.json')
assert not prior_score['development_gate_passed']
assert prior_score['repeat35_exact'] and prior_score['warm_cold100_exact']
assert prior_score['pooled_adjusted_ratio'] < 1
parents = read(D / 'lazy_rotation_objective_v1/panel/completed.json')
assert parents['candidate_sha256'] == PARENT_SHA
saved = {row['rank']: row['candidate'] for row in
         read(D / 'full_lower_moment_objective_v1/predictions_sealed.json')['pairs']}
old_scores = {r['rank']: r for r in
              read(D / 'lazy_rotation_objective_v1/panel_score.json')['rows']}
rows = []
for entry in parents['records']:
    rank = entry['rank']
    parent = entry['call']
    verify_call(parent)
    truth = P2 / 'public_g19_attribution_v1' / (
        f'fresh_{rank}_truth4M.npz' if rank >= 3016500
        else f'mps_truth_rank{rank:04d}_indep4M.npz')
    assert sha(truth) == old_scores[rank]['truth_sha256']
    if rank in saved:
        verify_call(saved[rank])
        assert saved[rank]['weights_sha256'] == parent['weights_sha256']
    rows.append(dict(rank=rank, parent=parent, truth=pin(truth),
                     saved_candidate=saved.get(rank)))
assert tuple(r['rank'] for r in rows) == NETS
write(HERE / 'inputs.json', dict(rows=rows, candidate=pin(CANDIDATE),
    parent_sha256=PARENT_SHA, prior_verification_counts=counts,
    prior_goal_turn_classification='numerical progress', references_decoded=False))
paths = [HERE / name for name in ('PROTOCOL.md', 'common.py', 'prepare.py',
                                 'panel.py', 'score.py', 'inputs.json')]
paths += [CANDIDATE, D / 'run_guarded.py', D / 'weight5_reuse_v1/native.py',
          D / 'continuation_checkpoint_v60_verified.json',
          D / 'full_lower_moment_objective_v1/dev_score.json',
          D / 'lazy_rotation_objective_v1/panel_score.json']
write(HERE / 'sources_sealed.json', dict(pins=[pin(p) for p in paths],
    candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA,
    new_predictions=0, references_decoded=False))
print(dict(authenticated=counts, rows=len(rows), saved_calls=len(saved)))

