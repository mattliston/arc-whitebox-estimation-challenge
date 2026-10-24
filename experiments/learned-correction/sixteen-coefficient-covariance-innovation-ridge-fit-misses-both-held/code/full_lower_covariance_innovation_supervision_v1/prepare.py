import ast
from common import *
assert sha(SOURCE)==PARENT_SHA
for f in HERE.glob('*.py'):ast.parse(f.read_text())
inputs=[SOURCE,D/'run_guarded.py',D/'full_lower_multilayer_mean_observer_v1/parent_triple/result.json']
baseline=read(inputs[-1])
inputs.extend(Path(r['output']) for r in baseline['records'])
for rank in TRAIN+VALID:
    assert weights(rank).exists() and truth(rank).exists()
    # Hash existing labels without decoding them; no reference value inspection.
    inputs.extend([weights(rank),truth(rank)])
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.glob('*.py'))]+[pin(HERE/'PROTOCOL.md')]+[pin(p) for p in inputs],parent_sha256=PARENT_SHA,targets_decoded=False))
print('Sources, roster and unchanged controls sealed; no numerical job launched.')
