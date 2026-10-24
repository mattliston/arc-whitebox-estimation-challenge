import ast
from common import *
verify();verify('fit_sources_sealed.json')
for n in ('replay.py','prepare_replay.py','run_replay.py'):ast.parse((HERE/n).read_text())
write(HERE/'replay_sources_sealed.json',dict(pins=[pin(HERE/n) for n in ('replay.py','prepare_replay.py','run_replay.py')],
      target_relative_tolerance=1e-9,loss_relative_tolerance=5e-11,does_not_change_fit_or_gates=True))
