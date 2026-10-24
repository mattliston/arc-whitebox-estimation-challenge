from common import *
seal_check();guarded('build',60,'build.py');guarded('legacy_replay',180,'legacy_replay.py')
write(HERE/'preparation_completed.json',dict(completed=True,guards=['build','legacy_replay'],ordinary_1B_targets_read=False))
print(dict(preparation_completed=True),flush=True)
