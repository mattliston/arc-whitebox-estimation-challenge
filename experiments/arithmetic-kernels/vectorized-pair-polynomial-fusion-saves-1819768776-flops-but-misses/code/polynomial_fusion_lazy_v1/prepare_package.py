"""Package only after the separate fixed objective study and client checks pass."""
import hashlib
import json
from pathlib import Path
import tarfile
from whestbench.packaging import package_submission
from whestbench.validation import validate_package

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
candidate=HERE/'candidate.py'
expected='29e9211b4136bde7a74c9f554e697bd0e7e0c77e69326e5c365bafedfa9d8b07'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected
assert read(HERE/'build.json')['whole_source_inverse_byte_exact']
assert read(HERE/'unit_v1/result.json')['passed']
assert read(HERE/'dev_score.json')['development_gate_passed']
panel=read(HERE/'panel_score.json')
assert panel['panel_gate_passed'] and panel['official_bank_ratio_projection']<4e-9
assert read(HERE/'confirmation_score.json')['confirmation_gate_passed']
assert read(HERE/'boundary_comparison.json')['boundary_gate_passed']
contract=read(HERE/'contract_watch.json')
assert contract['completed'] and contract['whole_group_retired'] and contract['max_swap_growth_bytes']==0
archive=HERE/'submission-polynomial-fusion-20260919.tar.gz'
assert not archive.exists()
package_submission(candidate,class_name='Estimator',output_path=archive)
validation=validate_package(archive)
assert validation.ok,validation.issues
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getnames()
    assert sorted(members)==['estimator.py','manifest.json']
    assert tar.extractfile('estimator.py').read()==candidate.read_bytes()
    manifest=json.load(tar.extractfile('manifest.json'))
assert manifest['files']==[dict(name='estimator.py',sha256=expected)]
result=dict(archive=str(archive.resolve()),bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),candidate_sha256=expected,
            manifest=manifest,members=members,validated=True,uploaded=False,
            local_nc14_adjusted=panel['groups']['all14']['candidate']['adjusted_nc'],
            official_parent_ratio_projection=panel['official_bank_ratio_projection'],
            old_component_gate_remains_failed=True,qualified_on_current_lazy_parent=True,
            prior_lazy_archive_unchanged=True,standing_submission_preauthorization=True)
with (HERE/'package.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
