"""Create a pinned single-file package only after all fixed qualification gates."""
import hashlib
import json
from pathlib import Path
import tarfile
from whestbench.packaging import package_submission
from whestbench.validation import validate_package

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
candidate = HERE / 'candidate.py'
expected = 'd7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected
assert read(HERE / 'build.json')['candidate_sha256'] == expected
assert read(HERE / 'source_audit.json')['parent_prefix_byte_exact']
assert read(HERE / 'unit_v1/result.json')['passed']
assert read(HERE / 'dev_score.json')['development_gate_passed']
panel = read(HERE / 'panel_score.json')
boundary = read(HERE / 'boundary_comparison.json')
contract = read(HERE / 'contract_watch.json')
assert contract['completed'] and contract['whole_group_retired']
assert panel['panel_gate_passed'] and panel['official_bank_ratio_projection'] < 4e-9
assert boundary['candidate_sha256'] == expected
assert all(boundary[k] for k in ('native_byte_exact', 'native_flops_exact', 'repeat_byte_exact',
                                 'numpy_free_client', 'boundary_gate_passed'))
archive = HERE / 'submission-read5-20260919.tar.gz'
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
rec = dict(archive=str(archive.resolve()), bytes=archive.stat().st_size,
           archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), candidate_sha256=expected,
           manifest=manifest, validated=True, members=names,
           local_nc14_adjusted=panel['groups']['all14']['candidate']['adjusted_nc'],
           official_parent_ratio_projection=panel['official_bank_ratio_projection'],
           uploaded=False, upload_approval_required=True)
with (HERE / 'package.json').open('x') as stream:
    json.dump(rec, stream, indent=2)
print(json.dumps(rec), flush=True)
