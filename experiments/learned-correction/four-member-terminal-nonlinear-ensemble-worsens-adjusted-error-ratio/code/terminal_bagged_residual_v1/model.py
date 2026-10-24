"""Small terminal-only model and train-only weighted affine projection."""
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1


class ResidualMLP(torch.nn.Module):
    def __init__(self, payload, dtype=torch.float32):
        super().__init__()
        for name in ('w', 'b', 'v', 'c'):
            self.register_parameter(name, torch.nn.Parameter(torch.tensor(payload[name], dtype=dtype, device='cpu')))

    def forward(self, x):
        z = x @ self.w.T + self.b
        h = z / (1 + torch.exp(-z))
        return h @ self.v + self.c

    def export(self):
        return {name: value.detach().cpu().numpy().copy() for name, value in self.named_parameters()}


def numpy_forward(payload, x):
    z = x @ payload['w'].T + payload['b']
    h = z / (1 + np.exp(-z))
    return h @ payload['v'] + payload['c']


def weighted_rms(x, weight):
    return np.sqrt(np.sum(x * x * weight[:, None], axis=0) / np.sum(weight))


def control_fit(x, y, weight):
    scale = np.maximum(weighted_rms(x, weight), 1e-12)
    xn = x / scale
    gram = xn.T @ (xn * weight[:, None])
    gram = (gram + gram.T) * .5
    gram.flat[::len(gram) + 1] += 1e-4 * np.sum(weight)
    chol = np.linalg.cholesky(gram)
    beta = np.linalg.solve(chol.T, np.linalg.solve(chol, xn.T @ (weight * y)))
    root = np.sqrt(weight)
    u, singular, vt = np.linalg.svd(root[:, None] * xn, full_matrices=False)
    keep = singular > 1e-7 * singular[0]
    pinv = (vt[keep].T / singular[keep]) @ (u[:, keep].T * root[None, :])
    return dict(scale=scale, beta=beta, xn=xn, pinv=pinv,
                singular=singular, rank=int(keep.sum()))


def project_torch(q, xn, pinv):
    coefficient = pinv @ q.to(torch.float64)
    return q.to(torch.float64) - xn @ coefficient, coefficient


def certificate():
    rng = np.random.default_rng(190919699)
    n, inputs, width = 23, 5, 4
    x = rng.normal(size=(n, inputs))
    t = np.linspace(-1, 1, n)
    affine = np.stack([np.ones(n), t, np.cos(2 * t)], axis=1)
    weight = 1.3 + .2 * t
    control = control_fit(affine, np.sin(3 * t), weight)
    assert control['rank'] == 3
    payload = dict(w=rng.normal(size=(width, inputs)) * .2,
                   b=rng.normal(size=width) * .1,
                   v=rng.normal(size=width) * .2, c=np.array(.04))
    net = ResidualMLP(payload, dtype=torch.float64)
    xt = torch.tensor(x, dtype=torch.float64)
    xn = torch.tensor(control['xn'], dtype=torch.float64)
    pinv = torch.tensor(control['pinv'], dtype=torch.float64)
    sigma = torch.tensor(.8 + .1 * t, dtype=torch.float64)
    wt = torch.tensor(weight, dtype=torch.float64)
    target = torch.tensor(.005 * np.sin(4 * t), dtype=torch.float64)

    def loss():
        q = sigma * net(xt)
        delta, _ = project_torch(q, xn, pinv)
        return torch.sum(wt * (delta - target)**2) / wt.sum()

    value = loss()
    value.backward()
    directions = [torch.tensor(rng.normal(size=tuple(p.shape)), dtype=torch.float64)
                  for p in net.parameters()]
    automatic = sum(float(torch.sum(p.grad * direction))
                    for p, direction in zip(net.parameters(), directions))
    saved = [p.detach().clone() for p in net.parameters()]
    eps = 1e-6
    sides = []
    with torch.no_grad():
        for sign in (1, -1):
            for p, base, direction in zip(net.parameters(), saved, directions):
                p.copy_(base + sign * eps * direction)
            sides.append(float(loss()))
        for p, base in zip(net.parameters(), saved):
            p.copy_(base)
        q = sigma * net(xt)
        delta, coefficient = project_torch(q, xn, pinv)
    finite = (sides[0] - sides[1]) / (2 * eps)
    relative = abs(finite - automatic) / max(1e-12, abs(finite), abs(automatic))
    assert relative < 1e-7, (finite, automatic, relative)
    d = delta.numpy()
    orthogonality = float(np.max(np.abs(control['xn'].T @ (weight * d))) / weight.sum())
    assert orthogonality < 1e-12
    assert np.max(np.abs(net(xt).detach().numpy() - numpy_forward(payload, x))) < 1e-13
    duplicate = control['xn'] @ np.array([.3, -.6, .2])
    assert np.max(np.abs(duplicate - control['xn'] @ (control['pinv'] @ duplicate))) < 1e-12
    zero = dict(payload, v=np.zeros(width), c=np.array(0.))
    z = ResidualMLP(zero, dtype=torch.float64)(xt)
    dz, _ = project_torch(sigma * z, xn, pinv)
    assert torch.count_nonzero(dz).item() == 0
    predictions = rng.normal(size=(4, n))
    truth = rng.normal(size=n)
    lhs = np.mean((predictions.mean(axis=0) - truth)**2)
    rhs = np.mean((predictions - truth)**2) - np.mean(np.var(predictions, axis=0))
    identity_error = float(abs(lhs - rhs))
    assert identity_error < 1e-14
    return dict(passed=True, directional_derivative_relative_error=relative,
                weighted_orthogonality=orthogonality, affine_duplicate_removed=True,
                zero_head_is_exact_null=True, torch_numpy_double_forward_exact_within=1e-13,
                ensemble_mse_identity_abs_error=identity_error, target_files_opened=False)
