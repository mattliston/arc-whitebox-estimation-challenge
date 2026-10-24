import ast
from common import *
verify();verify('fit_sources_sealed.json')
for name in ('replay_judgment.py','prepare_replay.py'):ast.parse((HERE/name).read_text())
write(HERE/'replay_sources_sealed.json',dict(pins=[pin(HERE/n) for n in ('replay_judgment.py','prepare_replay.py','DATA_PROVENANCE.md')],
      independent_centering_and_two_GEMM_response=True,changes_to_protocol_or_fit=False))
