from common import *
verify();verify('replay_sources_sealed.json');assert read(HERE/'fit_completed.json')['completed']
guarded('replay',600,'replay.py')
