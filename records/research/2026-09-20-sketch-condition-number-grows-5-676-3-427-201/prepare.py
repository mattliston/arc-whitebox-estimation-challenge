from datetime import datetime, timezone
from common import HERE, D, P2, OLD, PARENT, RUNTIME, read, write, pin, verify

build = read(OLD / 'build.json')
controls = read(OLD / 'response_qualified.json')
base = next(x for x in read(OLD / 'inputs.json')['qualification_parent'] if x['rank'] == 38)
rows = [dict(label='base', source=pin(PARENT), output=base['output'], flops=base['flops'], receipt=base['receipt'])]
for label in ['plus', 'minus', 'double']:
    variant = next(x for x in build['variants'] if x['coordinate'] == 0 and x['label'] == label)
    control = next(x for x in controls['rows'] if x['rank'] == 38 and x['coordinate'] == 0 and x['label'] == label and not x['repeat'])
    rows.append(dict(label=label, source=variant['source'], output=control['output'],
                     flops=control['flops'], receipt=pin(OLD / 'response_qualified.json')))
for row in rows:
    for key in ['source', 'output', 'receipt']:
        verify(row[key])
write(HERE / 'inputs.json', dict(rank=38, coordinate=0, epsilon=.0625, controls=rows, targets_opened=False))
np_source = RUNTIME.parent.parent / 'lib/python3.11/site-packages/numpy/linalg/_linalg.py'
backend_review = P2 / 'astra_phase2_reexamination_v1/current_comp_shared_old384_v1/BACKEND_DECISION_20260912.md'
paths = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md', HERE / 'inputs.json', D / 'run_guarded.py',
         D / 'official_best_331684.json', OLD / 'build.json', OLD / 'response_qualified.json', np_source, backend_review]
paths.extend(x[k]['path'] for x in rows for k in ['source', 'output', 'receipt'])
unique = sorted({str(__import__('pathlib').Path(p).resolve()) for p in paths})
write(HERE / 'sources_sealed.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
     pins=[pin(p) for p in unique], targets_opened=False, previous_turn='progress'))
print(dict(sealed=len(unique), controls=4))
