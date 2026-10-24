"""Small retired-result validation, no network/model or parameter update."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
student, teacher = map(Path, sys.argv[1:3])
complete = json.loads((student/'COMPLETE_v1.json').read_text())
record = torch.load(complete['result'],map_location='cpu',weights_only=False)
with np.load(teacher/'TEACHER_TARGET_v1.npz',allow_pickle=False) as data:
    parent = data['prediction'].copy()
prediction = np.asarray(record['prediction'],dtype=np.float64)
assert prediction.shape == parent.shape == (16,1024)
assert np.array_equal(prediction[:7],parent[:7]), 'Frozen prefix changed'
expected = np.mean((prediction[-1]-parent[-1])**2)
assert np.isclose(expected,record['distillation_mse'],rtol=1e-12,atol=0)
assert record['scale']==1e8 and len(record['grads'])==6
assert all(torch.isfinite(g).all() for g in record['grads'].values())
assert any(torch.count_nonzero(g) for g in record['grads'].values())
with (student/'RESULT_CHECK_v1.json').open('x') as f:
    json.dump(dict(status='PASS',prefix_L0_6_byte_exact=True,
                   explicit_teacher_loss_verified=True,six_finite_gradients=True,
                   parameters_updated=False),f,indent=2)
print('retired result checks PASS')
