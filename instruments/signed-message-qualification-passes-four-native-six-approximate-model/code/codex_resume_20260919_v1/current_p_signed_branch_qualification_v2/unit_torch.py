from common import *
admission()
import numpy as np
import torch
import model_parent as parent
import model_branch as branch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
p0, a0, c0, e0 = parent.torch_payload(torch, np, False)
p1, a1, c1, e1 = branch.torch_payload(torch, np, False)
p1['ungated_w'].requires_grad_(True)
env0 = parent.source_environment(parent.TorchNP(torch), None, e0, torch)
env1 = branch.source_environment(branch.TorchNP(torch), None, e1, torch)
rng = np.random.default_rng(9409214)
layer = 1
x = p0['centers'][layer] + .2 * p0['scales'][layer] * torch.tensor(rng.normal(size=(8,30)), dtype=torch.float32)
w = torch.tensor(rng.normal(size=(8,8)) * .2, dtype=torch.float32)
state = torch.tensor(rng.normal(size=(8,64)) * .1, dtype=torch.float32)
sigma = torch.full((8,), .7)
left = env0['gnn_step'](layer,x,sigma,w,state,p0)
right = env1['gnn_step'](layer,x,sigma,w,state,p1)
assert all(parent.tensor_sha(a) == branch.tensor_sha(b) for a,b in zip(left,right))
cot = torch.tensor(rng.normal(size=(8,64)), dtype=torch.float64) / 128
loss = (right[1].to(torch.float64) * cot).sum()
loss.backward()
grad = p1['ungated_w'].grad.detach().clone()
assert bool(torch.isfinite(grad).all()) and float(grad.norm()) > 0
checks = []
for index in range(3):
    direction = torch.tensor(rng.choice([-1., 1.], size=(128,48)), dtype=torch.float32)
    direction = direction / direction.norm()
    values = []
    for sign in [-1, 1]:
        pp = dict(p1)
        pp['ungated_w'] = sign * .01 * direction
        output = env1['gnn_step'](layer,x,sigma,w,state,pp)[1]
        values.append(float((output.to(torch.float64) * cot).sum()))
    fd = (values[1] - values[0]) / .02
    ad = float((grad.to(torch.float64) * direction.to(torch.float64)).sum())
    error = abs(fd - ad)
    assert error <= 5e-7 + .05 * abs(ad), (index,fd,ad,error)
    checks.append(dict(index=index, finite_difference=fd, autograd=ad, absolute_error=error))
assert int(torch.count_nonzero(p1['ungated_w'])) == 0
write(HERE / 'unit_torch_result.json', dict(passed=True, zero_branch_bytes_exact=True,
    gradient_norm=float(grad.norm()), directional_checks=checks, epsilon=.01,
    gradient_scope='small synthetic encoder only', native_derivative_certified=False,
    targets_decoded=False, optimizer_updates=0))
print(dict(passed=True, branch_gradient_norm=float(grad.norm()), directional_checks=checks), flush=True)
