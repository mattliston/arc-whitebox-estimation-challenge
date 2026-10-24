"""Version an all-old local approximation, separately from submitted S31."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = HERE/'candidate_old_both_tail_trace_bounded_v6.py'
parent_sha = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_sha
submitted = json.loads((HERE.parent/'fable_fire_v1/fire_candS31/submission_receipt_v1.json').read_text())
assert submitted['candidate_sha256'] == parent_sha and submitted['submission_id'] == 332517
fragment = HERE/'old_all_trace_research_fragment_v1.py'
text = parent.read_text() + '\n\n' + fragment.read_text()
target = HERE/'candidate_old_all_trace_v1.py'
compile(text, str(target), 'exec')
with target.open('x') as f:
    f.write(text)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate=str(target),
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(), parent=str(parent), parent_sha256=parent_sha,
              fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),
              changed='At L7-14 replace all old local rows with zero plus analytical diagonal/radial repair.',
              preserved='Full propagated Q/F carrier, all young columns, trained parameters and setup keys.',
              approximation='Squared left operand in the response-plane contraction also becomes zero. The complete old response plane is not restored.',
              status='Unqualified research; training screen only',
              protocol=dict(initial_ranks=[35,100], baseline='both_tail_train2_v6',
                            expand_only_if='Meaningful adjusted gain on TRAIN2; no validation for a failed screen'))
with (HERE/'old_all_trace_manifest_v1.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
