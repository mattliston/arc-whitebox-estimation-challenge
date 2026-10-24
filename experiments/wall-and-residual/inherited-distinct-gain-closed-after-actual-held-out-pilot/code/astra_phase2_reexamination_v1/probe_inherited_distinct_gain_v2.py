"""Same frozen one-direction pilot, separately qualified local derivative.

The original v1 script and its failed global prerequisite are preserved.
Only output version names and the explicit prerequisite change. All train,
holdout, step, fidelity, actual-selection and raw-improvement gates remain.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'probe_inherited_distinct_gain_v1.py').read_text()
before="assert json.loads((HERE/'inherited_distinct_gain_fidelity_v1.json').read_text())['passed']"
after="""assert json.loads((HERE/'inherited_distinct_gain_fidelity_v1.json').read_text())['passed'] is False
assert json.loads((HERE/'inherited_gain_local_branch_v2.json').read_text())['passed']
assert not (HERE/'inherited_distinct_gain_pilot_v1.json').exists()"""
assert source.count(before)==1
source=source.replace(before,after)
for stem in ('pilot','control','progress'):
    source=source.replace('inherited_distinct_gain_'+stem+'_v1',
                          'inherited_distinct_gain_'+stem+'_v2')
exec(compile(source,str(HERE/'probe_inherited_distinct_gain_v1.py'),'exec'),globals())
