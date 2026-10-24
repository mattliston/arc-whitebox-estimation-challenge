import ast
from datetime import datetime,timezone
from common import *
assert sha(SOURCE)==PARENT_SHA and shutil.disk_usage(HERE).free>=480_000_000_000
assert not (HERE/'sources_sealed.json').exists()
for p in HERE.glob('*.py'):ast.parse(p.read_text())
external=[SOURCE,D/'run_guarded.py',D/'continuation_checkpoint_v92.json',OLD/'closeout.json',OLD/'PROTOCOL.md',OLD/'RESULT.md',
          OLD/'head.py',OLD/'DATA_PROVENANCE.md',OLD/'DIRECTED_RESPONSE_CORRECTION.md',OLD/'directed_response_audit.py',
          A/'FOURTH_PLANE_PAIR_SPAN_V1_CLOSED.md',A/'CURRENT_PAIR_ORACLE_ATTRIBUTION_PARTIAL_V2.md',
          A/'ANISOTROPIC_FOURTH_SOURCE_ROUND_V1_CLOSED.md',A/'FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md',
          A/'strategic_reset_20260909_v1/accuracy.md',D/'frozen_covariance_readout_composition_v1/upload_review_rejection.json']
for ordinal in TRAIN+VALID:
    external.extend([OLD/f'capture_mini{ordinal:04d}.json',OLD/f'inputs_mini{ordinal:04d}.json',OLD/f'targets_mini{ordinal:04d}.json',OLD/f'download_mini{ordinal:04d}.json'])
write(HERE/'sources_sealed.json',dict(utc=datetime.now(timezone.utc).isoformat(),pins=[pin(p) for p in sorted(HERE.glob('*')) if p.is_file()]+[pin(p) for p in external],
      new_fourth_fields_decoded=False,training_started=False,reused_validation=True))
print(dict(sealed=True,free_bytes=shutil.disk_usage(HERE).free))
