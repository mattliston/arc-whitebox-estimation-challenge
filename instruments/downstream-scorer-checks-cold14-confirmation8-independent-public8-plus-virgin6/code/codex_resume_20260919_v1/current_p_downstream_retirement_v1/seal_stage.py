import argparse
from records import *
ap = argparse.ArgumentParser()
ap.add_argument('stage',choices=['panel','confirmation','independent'])
stage = ap.parse_args().stage
stage_ready(stage)
meta = read(HERE/'inputs.json')
roles, pins = {}, [pin(HERE/'admission.json'),pin(HERE/'inputs.json'),pin(HERE/'sources_sealed.json'),
    pin(HERE/'control_qualification.json'),pin(HERE/'qualify_control_watch.json')]
if stage == 'independent':
    ranks = INDEPENDENT
    for role in ['parent','trained']:
        rows,items = verify_run(f'independent_{role}',stage,role,ranks+[35])
        roles[role] = rows[:-1]
        pins.extend(items)
else:
    ranks = PANEL if stage == 'panel' else CONFIRM
    roles = dict(parent=[],trained=[])
    for rank,baseline in zip(ranks,meta['baselines'][stage]):
        assert baseline['rank'] == rank
        rows,items = verify_run(f'{stage}_{rank}',stage,'trained',[rank])
        roles['trained'].extend(rows)
        pins.extend(items)
        parent = baseline['parent']
        for key in ['prediction','receipt','source']:
            check(parent[key]); pins.append(parent[key])
        check(baseline['parent_watch']); pins.append(baseline['parent_watch'])
        clean_watch(baseline['parent_watch']['path'])
        roles['parent'].append(dict(rank=rank,**parent))
write(HERE/f'{stage}_predictions_sealed.json',dict(completed=True,stage=stage,ranks=ranks,roles=roles,
    pins=list({p['path']:p for p in pins}.values()),all_predictions_and_bills_before_targets=True,
    repeats_excluded=True,own_complete_bills=True,independent_truth_files_opened=False))
print(dict(sealed=True,stage=stage,networks=len(ranks),truths_decoded=False),flush=True)
