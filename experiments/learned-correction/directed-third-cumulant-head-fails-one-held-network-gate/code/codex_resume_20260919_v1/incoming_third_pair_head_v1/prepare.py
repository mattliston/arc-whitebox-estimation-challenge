import ast,shutil
from datetime import datetime,timezone
from common import *
assert not (HERE/'sources_sealed.json').exists()
assert shutil.disk_usage(HERE).free>=480_000_000_000
assert sha(SOURCE)==PARENT_SHA
external=[SOURCE,A/'public_p2_moments_metadata_v1.json',A/'PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md',
          A/'public_third_centering_v1.py',A/'public_third_centering_certificate_v1.json',
          P2/'data_gate/target_blind_mini_v2.py',P2/'data_gate/authority.json',
          D/'run_guarded.py',D/'continuation_checkpoint_v91.json',
          D/'frozen_pair_history_endpoint_v1/RESULT.md',D/'canonical_selector_response_v1/RESULT.md',
          D/'fp64_sketch_response_v1/RESULT.md',D/'fp64_sketch_endpoint_v1/RESULT.md',
          D/'frozen_covariance_readout_composition_v1/upload_review_rejection.json']
for p in HERE.glob('*.py'):ast.parse(p.read_text())
pins=[pin(p) for p in sorted(HERE.glob('*')) if p.is_file()]+[pin(p) for p in external]
write(HERE/'sources_sealed.json',dict(utc=datetime.now(timezone.utc).isoformat(),pins=pins,
      current_parent=PARENT_SHA,train=list(TRAIN),heldout=list(HELDOUT),new_target_values_decoded=0))
print(dict(sealed=len(pins),free_bytes=shutil.disk_usage(HERE).free))
