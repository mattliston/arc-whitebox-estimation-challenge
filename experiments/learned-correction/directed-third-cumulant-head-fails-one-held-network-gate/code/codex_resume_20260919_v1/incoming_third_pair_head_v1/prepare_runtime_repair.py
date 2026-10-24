import ast
from common import *
verify()
old=(HERE/'unit.py').read_text();a='from scipy.special import ndtr';b='import math\ndef ndtr(x):return np.array([.5*math.erfc(-float(v)/math.sqrt(2.)) for v in x],dtype=np.float64)'
assert old.count(a)==1;new=old.replace(a,b,1);assert new.replace(b,a,1)==old
with (HERE/'unit_v2.py').open('x') as f:f.write(new)
failure=read(HERE/'unit_watch.json');assert failure['exit_code']==1 and failure['whole_group_retired'] and failure['max_swap_growth_bytes']==0
assert 'ModuleNotFoundError: No module named \'scipy\'' in (HERE/'unit.log').read_text()
write(HERE/'runtime_repair.json',dict(reason='Official-compatible runtime has NumPy and flopscope, no SciPy or PyArrow.',
      unit_change='Independent normal CDF uses Python math.erfc in place of unavailable SciPy ndtr; equations, tolerances and all numerical stages unchanged.',
      unit_exact_inverse=True,weight_change='Same original get_weights function runs separately under existing data_gate PyArrow environment, projected allowed columns only; original native arithmetic remains in official-compatible environment.',
      original_failure=pin(HERE/'unit_watch.json'),new_parameters_or_targets=False))
names=['unit_v2.py','extract_weights.py','weights_dispatch.py','capture_v2.py','run_inputs_v2.py','prepare_runtime_repair.py','runtime_repair.json']
for n in names:
    if n.endswith('.py'):ast.parse((HERE/n).read_text())
write(HERE/'runtime_repair_sealed.json',dict(pins=[pin(HERE/n) for n in names]))
print(read(HERE/'runtime_repair.json'))
