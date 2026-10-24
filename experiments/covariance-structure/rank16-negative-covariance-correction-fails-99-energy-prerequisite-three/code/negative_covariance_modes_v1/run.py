from common import *

verify();guarded('unit', 60, 'unit.py');guarded('spectrum', 120, 'spectrum.py')
spectrum = read(HERE/'spectrum_result.json')
write(HERE/'geometry_sealed.json', dict(pins=[pin(HERE/'spectrum_result.json')]+[r['file'] for r in spectrum['rows']], targets_decoded=False))
if spectrum['passed']:
    guarded('score', 90, 'score.py')
    passed = read(HERE/'score_result.json')['passed'];stage = 'covariance_and_response'
else:
    passed = False;stage = 'rank16_spectrum'
write(HERE/'completed.json', dict(completed=True, passed=passed, stage=stage,
      targets_opened=spectrum['passed'], goal_achieved=False, candidate_created=False))
