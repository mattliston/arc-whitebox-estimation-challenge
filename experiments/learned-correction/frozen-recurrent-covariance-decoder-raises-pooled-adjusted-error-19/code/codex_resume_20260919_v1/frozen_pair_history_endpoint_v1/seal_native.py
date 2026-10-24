import ast
from common import *
seal_check();done('build');done('legacy_replay');assert read(HERE/'legacy_replay.json')['passed']
names=('unit.py','native.py','qualify.py','score.py','run_native.py','seal_native.py')
for name in names:ast.parse((HERE/name).read_text(),filename=name)
write(HERE/'native_sources_sealed.json',dict(pins=[pin(HERE/n) for n in names]+[
    pin(HERE/n) for n in ('candidate.py','parameters.npz','build.json','legacy_replay.json','preparation_completed.json')],
    before_units_and_current_native_calls=True,before_ordinary_1B_targets=True,scientific_protocol_unchanged=True))
print(dict(native_sources_sealed=True),flush=True)
