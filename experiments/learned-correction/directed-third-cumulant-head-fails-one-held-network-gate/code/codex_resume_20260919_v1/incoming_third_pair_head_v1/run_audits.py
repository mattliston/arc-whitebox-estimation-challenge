from common import *
verify();verify('replay_sources_sealed.json');verify('directed_audit_sealed.json');verify('audit_controller_sealed.json')
assert read(HERE/'fit_completed.json')['completed']
guarded('judgment_replay',600,'replay_judgment.py')
guarded('directed_audit',600,'directed_response_audit.py')
write(HERE/'audits_completed.json',dict(completed=True,original_replay=pin(HERE/'judgment_replay.json'),directed_audit=pin(HERE/'directed_response_audit.json')))
