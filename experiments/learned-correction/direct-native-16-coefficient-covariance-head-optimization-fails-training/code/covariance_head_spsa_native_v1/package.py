import json,tarfile
from whestbench.packaging import package_submission
from whestbench.validation import validate_package
from common import HERE,read,write,pin,verify,sha
for p in read(HERE/'release_sources_sealed.json')['pins']:verify(p)
assert read(HERE/'release_qualified.json')['passed']
g=read(HERE/'contract_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
candidate=HERE/'candidate.py';archive=HERE/'submission-native-spsa-covariance-20260920.tar.gz'
assert not archive.exists();package_submission(candidate,class_name='Estimator',output_path=archive)
valid=validate_package(archive);assert valid.ok,valid.issues
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getnames();assert sorted(members)==['estimator.py','manifest.json']
    assert tar.extractfile('estimator.py').read()==candidate.read_bytes()
    manifest=json.load(tar.extractfile('manifest.json'))
assert manifest['files']==[dict(name='estimator.py',sha256=sha(candidate))]
write(HERE/'package.json',dict(validated=True,archive=pin(archive),candidate=pin(candidate),members=members,manifest=manifest,uploaded=False))
print(dict(validated=True,archive=pin(archive)),flush=True)
