import ast
from common import *
verify();assert read(HERE/'unit_result.json')['passed']
names=['targets.py','data.py','train.py','judge.py','run_fit.py','prepare_fit.py']
for n in names:ast.parse((HERE/n).read_text())
write(HERE/'fit_sources_sealed.json',dict(pins=[pin(HERE/n) for n in names],new_fourth_fields_decoded=False,model_trained=False,protocol=pin(HERE/'PROTOCOL.md'),no_design_change=True))
print(dict(fit_sources_sealed=len(names)))
