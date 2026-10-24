"""One shared-gradient/local-response information test, no optimization sweep."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from conditional_birth_autodiff_v1 import ConditionalBirthCard

HERE = Path(__file__).resolve().parent
OUT = HERE / 'conditional_birth_information_v1.json'
PROGRESS = HERE / 'conditional_birth_information_progress_v1'
assert not OUT.exists() and not PROGRESS.exists()
assert json.loads((HERE / 'conditional_birth_autodiff_r35_v1.json').read_text())['fidelity_pass']
assert json.loads((HERE / 'conditional_birth_cells_r100_v1.json').read_text())['fixed_cell_fidelity_pass']
for r in (35, 100):
    check = json.loads((HERE / f'conditional_birth_autodiff_r{r}_v1.json').read_text())
    assert check['zero_head_cpu_max_abs'] < 2e-6
    assert max(row['response_relative'] for row in check['rows']) <= .10
PROGRESS.mkdir()
VALID = (37, 39, 42, 48)
TRAIN = tuple(r for r in range(35, 59) if r not in VALID)
torch.set_num_threads(3)
start = time.time()


def label(rank):
    return torch.tensor(np.load(HERE.parent / 'public_g22_operator_features_v1' /
                               'truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'],
                        dtype=torch.float64)


def save(stage, rank, rec, **arrays):
    rec = dict(stage=stage, rank=rank, **rec, wall_s=time.time() - start,
               max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    path = PROGRESS / f'{stage}_r{rank:04d}'
    assert not path.with_suffix('.json').exists() and not path.with_suffix('.npz').exists()
    np.savez(path.with_suffix('.npz'), **arrays)
    path.with_suffix('.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec), flush=True)


gradients, training_errors = [], {}
for rank in TRAIN:
    assert time.time() - start < 900, '15-minute information-test deadline'
    theta = torch.nn.Parameter(torch.zeros((3, 64), dtype=torch.float32))
    card = ConditionalBirthCard(rank, theta)
    pred = card.run(training=True)[0].clamp_min(0)
    error = pred - label(rank)
    mse = (error * error).mean(1)
    (mse[-1] / 2e-8).backward()
    grad = theta.grad.detach().numpy().copy()
    assert np.isfinite(grad).all()
    gradients.append(grad)
    training_errors[rank] = error[-1].detach().numpy().copy()
    save('gradient', rank, dict(final_mse=float(mse[-1].detach()),
                               grad_norm=float(np.linalg.norm(grad))),
         gradient=grad, prediction=pred.detach().numpy())
    del theta, card, pred, error, mse
gradient = np.mean(gradients, axis=0)
norm = np.linalg.norm(gradient)
assert np.isfinite(norm) and norm > 1e-12, norm
direction = torch.tensor(-gradient / norm, dtype=torch.float32)
np.savez(PROGRESS / 'direction.npz', direction=direction.numpy(), gradients=np.stack(gradients))


def response(rank):
    assert time.time() - start < 900, '15-minute information-test deadline'
    card = ConditionalBirthCard(rank)
    with torch.no_grad(), forward_ad.dual_level():
        scalar = forward_ad.make_dual(torch.zeros((), dtype=torch.float32),
                                      torch.ones((), dtype=torch.float32))
        card.theta = scalar * direction
        output = card.run()[0].clamp_min(0)
        pred, tangent = forward_ad.unpack_dual(output)
        error = (pred - label(rank)).numpy()
        tangent = tangent.numpy()
    assert np.isfinite(tangent).all()
    rec = dict(a=float(np.mean(error[-1] ** 2)),
               b=float(np.mean(error[-1] * tangent[-1])),
               c=float(np.mean(tangent[-1] ** 2)))
    if rank in training_errors:
        rec['repeated_baseline_error_max_abs'] = float(np.max(np.abs(error[-1] - training_errors[rank])))
        assert rec['repeated_baseline_error_max_abs'] < 1e-12
    save('response', rank, rec, error=error, tangent=tangent)
    return dict(rank=rank, **rec)


train = [response(r) for r in TRAIN]
a, b, c = (float(np.mean([r[k] for r in train])) for k in ('a', 'b', 'c'))
step = float(np.clip(-b / max(c, 1e-30), 0, .25))
train_ratio = (a + 2 * step * b + step * step * c) / a
selection = dict(train=TRAIN, valid=VALID, parameters=192, mean_gradient_norm=float(norm),
                 trust_radius=.25, step=step, train_quadratic=dict(a=a, b=b, c=c),
                 train_predicted_ratio=train_ratio)
(PROGRESS / 'selected_step.json').write_text(json.dumps(selection, indent=2) + '\n')
print(json.dumps(selection), flush=True)
valid = []
if train_ratio <= .90:
    for rank in VALID:
        rec = response(rank)
        rec['predicted_mse'] = rec['a'] + 2 * step * rec['b'] + step * step * rec['c']
        rec['predicted_ratio'] = rec['predicted_mse'] / rec['a']
        valid.append(rec)
valid_ratio = None if not valid else float(np.mean([r['predicted_mse'] for r in valid]) / np.mean([r['a'] for r in valid]))
passed = bool(valid and valid_ratio <= .90 and max(r['predicted_ratio'] for r in valid) <= .95)
result = dict(**selection, train_rows=train, valid_rows=valid, valid_predicted_mean_ratio=valid_ratio,
              information_gate_pass=passed, wall_s=time.time() - start,
              max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              qualification='Local linear-response information test with one fit-side-selected direction and radius-bounded quadratic step. Not an actual nonlinear gain, candidate, adjusted score or submission. Four holdouts are excluded only from this new direction and step selection.')
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
