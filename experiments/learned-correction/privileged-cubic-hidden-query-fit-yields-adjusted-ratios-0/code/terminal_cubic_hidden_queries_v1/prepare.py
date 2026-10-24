import ast
from common import *
assert sha(SOURCE)==PARENT_SHA
for p in HERE.glob('*.py'):ast.parse(p.read_text())
triple=D/'full_lower_multilayer_mean_observer_v1/parent_triple/result.json';cold=D/'full_lower_multilayer_mean_observer_v1/parent_cold100/result.json'
rec=read(triple)['records'];cr=read(cold)['records'][0];assert [r['rank'] for r in rec]==[35,100,35] and cr['rank']==100
write(HERE/'controls.json',{'35':rec[0],'100':cr})
inputs=[SOURCE,D/'run_guarded.py',triple,cold]+[Path(r['output']) for r in rec]+[Path(cr['output'])]
for rank in (35,100):inputs.extend([weights(rank),truth(rank)])
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.glob('*.py'))]+[pin(HERE/'PROTOCOL.md'),pin(HERE/'controls.json')]+[pin(p) for p in inputs],targets_decoded=False,parent_sha256=PARENT_SHA))
print('Fixed source/query/readout and original controls sealed before numerics.')
