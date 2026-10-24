"""Prepare the separate qualified cache archive; never upload or change Read5."""
import hashlib
import json
from pathlib import Path
import tarfile
from whestbench.packaging import package_submission
from whestbench.validation import validate_package

HERE = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())
candidate = HERE / 'candidate.py'
expected = 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected
build = read(HERE / 'build.json')
assert build['candidate_sha256'] == expected and build['parent_byte_prefix_exact']
assert read(HERE / 'unit_v1/result.json')['passed']
assert read(HERE / 'native_comparison.json')['native_gate_passed']
panel = read(HERE / 'panel_score.json')
boundary = read(HERE / 'boundary_comparison.json')
contract = read(HERE / 'contract_watch.json')
assert contract['completed'] and contract['whole_group_retired']
assert panel['panel_gate_passed'] and panel['official_bank_ratio_projection'] < 4e-9
assert boundary['candidate_sha256'] == expected and boundary['boundary_gate_passed']
archive = HERE / 'submission-weight5-20260919.tar.gz'
assert not archive.exists()
package_submission(candidate, class_name='Estimator', output_path=archive)
validation = validate_package(archive)
assert validation.ok, validation.issues
with tarfile.open(archive, 'r:gz') as file:
    names = file.getnames()
    assert sorted(names) == ['estimator.py', 'manifest.json']
    payload = file.extractfile('estimator.py').read()
    manifest = json.load(file.extractfile('manifest.json'))
assert payload == candidate.read_bytes()
assert manifest['files'] == [dict(name='estimator.py', sha256=expected)]
receipt = dict(archive=str(archive.resolve()), bytes=archive.stat().st_size,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), candidate_sha256=expected,
               manifest=manifest, validated=True, members=names,
               local_nc14_adjusted=panel['groups']['all14']['candidate']['adjusted_nc'],
               official_parent_ratio_projection=panel['official_bank_ratio_projection'],
               uploaded=False, pending_read5_archive_unchanged=True,
               upload_approval_required=True, upload_approval_requested=False)
with (HERE / 'package.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
