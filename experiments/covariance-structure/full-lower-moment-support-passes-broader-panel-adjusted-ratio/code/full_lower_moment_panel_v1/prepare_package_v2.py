import tarfile
import json
from common import (HERE,CANDIDATE,CANDIDATE_SHA,read,write,pin,sha,verify,verify_pin)
from whestbench.packaging import package_submission
from whestbench.validation import validate_package

verify()
for item in read(HERE/'boundary_sources_sealed.json')['pins']:
    verify_pin(item)
assert read(HERE/'panel_score.json')['panel_advancement_passed']
assert read(HERE/'confirmation_score.json')['confirmation_advancement_passed']
assert read(HERE/'matched_runtime_comparison.json')['calibrated_runtime_gate_passed']
assert not read(HERE/'boundary_comparison.json')['boundary_gate_passed']
contract=read(HERE/'contract_watch.json')
assert contract['completed'] and contract['whole_group_retired']
assert contract['max_swap_growth_bytes']==0
copy=HERE/'candidate.py'
with copy.open('x') as stream:
    stream.write(CANDIDATE.read_text())
assert sha(copy)==CANDIDATE_SHA
archive=HERE/'submission-full-lower-moments-20260920.tar.gz'
assert not archive.exists()
package_submission(copy,class_name='Estimator',output_path=archive)
validation=validate_package(archive)
assert validation.ok,validation.issues
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getnames()
    assert sorted(members)==['estimator.py','manifest.json']
    assert tar.extractfile('estimator.py').read()==CANDIDATE.read_bytes()
    manifest=json.load(tar.extractfile('manifest.json'))
assert manifest['files']==[dict(name='estimator.py',sha256=CANDIDATE_SHA)]
write(HERE/'package.json',dict(archive=pin(archive),candidate=pin(copy),
    validated=True,members=members,manifest=manifest,uploaded=False,
    local_nc14_adjusted=read(HERE/'panel_score.json')['groups']['all14']['candidate']['adjusted_nc'],
    old_development_gate_passed=False,qualified_by_broader_study=True, original_runtime_gate_passed=False, calibrated_runtime=True))
print(dict(archive=pin(archive),validated=True),flush=True)
