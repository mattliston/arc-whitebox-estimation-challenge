import json
import tarfile
from whestbench.packaging import package_submission
from whestbench.validation import validate_package
from common import HERE, read, write, pin, verify, sha, seal_check

seal_check('release_sources_sealed.json')
assert read(HERE/'native_qualified.json')['passed'] and read(HERE/'boundary_qualified.json')['passed']
g = read(HERE/'contract_watch.json')
assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['max_swap_growth_bytes']==0
candidate = HERE/'candidate.py'
verify(read(HERE/'build.json')['candidate'])
archive = HERE/'submission-gaussian-gate-reuse-20260920.tar.gz'
assert not archive.exists()
package_submission(candidate,class_name='Estimator',output_path=archive)
result = validate_package(archive)
assert result.ok, result.issues
with tarfile.open(archive,'r:gz') as tar:
    members = tar.getnames()
    assert sorted(members) == ['estimator.py','manifest.json']
    assert tar.extractfile('estimator.py').read() == candidate.read_bytes()
    manifest = json.load(tar.extractfile('manifest.json'))
assert manifest['files'] == [dict(name='estimator.py',sha256=sha(candidate))]
write(HERE/'package.json',dict(validated=True,archive=pin(archive),candidate=pin(candidate),
    members=members,manifest=manifest,uploaded=False,native=pin(HERE/'native_qualified.json'),
    boundary=pin(HERE/'boundary_qualified.json'),parent=read(HERE/'build.json')['parent']))
print(dict(validated=True,archive=pin(archive)),flush=True)
