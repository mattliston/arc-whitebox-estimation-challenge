import ast
from common import *
verify();verify('runtime_repair_sealed.json')
names=['freeze_inputs.py','download.py','qualify_targets.py','train.py','judge.py','run_fit.py','prepare_fit.py']
for n in names:ast.parse((HERE/n).read_text())
write(HERE/'fit_sources_sealed.json',dict(pins=[pin(HERE/n) for n in names],new_target_values_decoded=0,model_trained=False,
      protocol=pin(HERE/'PROTOCOL.md'),no_design_change=True))
print(dict(fit_sources_sealed=len(names)))
