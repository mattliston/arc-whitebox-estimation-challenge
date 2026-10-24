"""Restore declared grad/thread state after a legacy import; preserve failed v1."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
def fixed(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

repair_path = HERE / 'runtime_repair_v2.json'
admission = json.loads((HERE / 'runtime_v2_root_admission.json').read_text())
assert admission['root_admitted'] is True and admission['repair'] == fixed(repair_path)
assert admission['original_protocol'] == fixed(HERE / 'protocol.json')
repair = json.loads(repair_path.read_text())
for record in repair['pins'] + repair['preserved_failure']:
    assert fixed(record['path']) == record, record['path']
source = HERE / 'train.py'
assert fixed(source)['sha256'] == '1584494b6e19334bdbaf7aed90a96bd053433abf6707f8a7fd4c6a93fa89130c'
text = source.read_text()
changes = [
    ("OUT = HERE / 'training'", "OUT = HERE / 'training_runtime_v2'", 1),
    ('spec.loader.exec_module(LC)',
     'spec.loader.exec_module(LC)\n'
     'torch.set_num_threads(1)\n'
     'torch.set_grad_enabled(True)\n'
     'assert torch.is_grad_enabled() and torch.get_num_threads() == 1', 1),
    ('trainer=pin(__file__),',
     "trainer=pin(__file__), original_trainer=pin(HERE / 'train.py'), "
     "runtime_repair=pin(HERE / 'runtime_repair_v2.json'),", 2),
]
for old, new, count in changes:
    assert text.count(old) == count, old
    text = text.replace(old, new)
exec(compile(text, str(source) + ':runtime_v2', 'exec'), globals())
