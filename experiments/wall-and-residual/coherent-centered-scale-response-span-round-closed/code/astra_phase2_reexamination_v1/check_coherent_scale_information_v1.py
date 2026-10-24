"""Complete small Jacobian-span test for coherent centered-state rescaling."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from coherent_central_scale_autodiff_v1 import CoherentScaleCard

HERE = Path(__file__).resolve().parent
OUT = HERE/'coherent_scale_information_v1.json'
PROGRESS = HERE/'coherent_scale_information_progress_v1'
assert not OUT.exists() and not PROGRESS.exists()
for rank in (35,100):
    check = json.loads((HERE/f'coherent_scale_autodiff_r{rank}_v1.json').read_text())
    assert check['zero_action_cpu_max_abs'] < 2e-6
    assert max(r['response_relative'] for r in check['rows']) <= .10
    if not check['fidelity_pass']:
        assert json.loads((HERE/f'coherent_scale_fixed_cell_r{rank}_v1.json').read_text())['branch_fidelity_pass']
PROGRESS.mkdir()
TRAIN = (35,36,38,40)
VALID = (37,39,42,48)
torch.set_num_threads(3)
start = time.time()


def truth(rank):
    path = HERE.parent/'public_g22_operator_features_v1'/'truthpost'/f'truthpost_rank{rank:04d}.npz'
    return np.load(path)['mc_post'][-1].astype(np.float64)


def directional(card, direction):
    assert time.time()-start < 900, '15-minute information-test deadline'
    with torch.no_grad(),forward_ad.dual_level():
        scalar = forward_ad.make_dual(torch.zeros((),dtype=torch.float64),torch.ones((),dtype=torch.float64))
        card.theta = scalar*torch.tensor(direction,dtype=torch.float64)
        pred,tangent = forward_ad.unpack_dual(card.run()[0].clamp_min(0))
        pred,tangent = pred.numpy().copy(),tangent.numpy().copy()
    assert np.isfinite(pred).all() and np.isfinite(tangent).all()
    return pred,tangent


def write_progress(name, receipt, **arrays):
    path = PROGRESS/name
    assert not path.with_suffix('.json').exists() and not path.with_suffix('.npz').exists()
    receipt = dict(**receipt,wall_s=time.time()-start,
                   max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if arrays:
        np.savez(path.with_suffix('.npz'),**arrays)
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


bases = np.eye(12).reshape(12,3,4)
train = []
errors,matrices = [],[]
for rank in TRAIN:
    card = CoherentScaleCard(rank)
    columns,baseline = [],None
    for j in range(12):
        pred,tangent = directional(card,bases[j])
        if baseline is None:
            baseline = pred
        else:
            assert np.max(np.abs(pred-baseline)) < 1e-12
        columns.append(tangent[-1])
        write_progress(f'jacobian_r{rank:04d}_c{j:02d}',
                       dict(stage='jacobian',rank=rank,column=j,
                            response_rms=float(np.sqrt(np.mean(tangent[-1]**2)))),
                       tangent=tangent)
    J = np.stack(columns,1)
    error = baseline[-1]-truth(rank)
    errors.append(error)
    matrices.append(J)
    row = dict(rank=rank,parent_mse=float(np.mean(error**2)))
    train.append(row)
    write_progress(f'baseline_r{rank:04d}',dict(stage='baseline',**row),baseline=baseline,error=error,jacobian=J)
    del card,columns,baseline,pred,tangent,J,error
J,e = np.concatenate(matrices),np.concatenate(errors)
H,g = J.T@J/len(e),J.T@e/len(e)
diagonal = np.maximum(np.diag(H),1e-30)
regularizer = .001*np.diag(diagonal)
matrix = H+regularizer
radius = .25
unconstrained = np.linalg.solve(matrix,-g)
mu = 0.
if np.linalg.norm(unconstrained) > radius:
    lo,hi = 0.,float(np.linalg.norm(g)/radius)
    for _ in range(80):
        mu = (lo+hi)/2
        trial = np.linalg.solve(matrix+mu*np.eye(12),-g)
        if np.linalg.norm(trial) > radius:
            lo = mu
        else:
            hi = mu
    mu = hi
theta = np.linalg.solve(matrix+mu*np.eye(12),-g)
assert np.isfinite(theta).all() and np.linalg.norm(theta) <= radius*(1+1e-10)
for row,Jr,er in zip(train,matrices,errors):
    row['predicted_mse'] = float(np.mean((er+Jr@theta)**2))
    row['predicted_ratio'] = row['predicted_mse']/row['parent_mse']
train_ratio = float(np.mean([r['predicted_mse'] for r in train])/np.mean([r['parent_mse'] for r in train]))
normalized = H/np.sqrt(diagonal[:,None]*diagonal[None,:])
eigenvalues = np.linalg.eigvalsh(normalized)
selection = dict(train=TRAIN,valid=VALID,parameters=12,ridge=.001,trust_radius=radius,
                 selected_norm=float(np.linalg.norm(theta)),lagrange_multiplier=mu,
                 unbounded_regularized_norm=float(np.linalg.norm(unconstrained)),
                 normalized_response_eigenvalues=eigenvalues.tolist(),
                 train_rows=train,train_predicted_mean_ratio=train_ratio)
write_progress('selected_update',dict(stage='selection',**selection),theta=theta.reshape(3,4),H=H,g=g)
valid = []
if train_ratio <= .80:
    for rank in VALID:
        card = CoherentScaleCard(rank)
        baseline,delta = directional(card,theta.reshape(3,4))
        error = baseline[-1]-truth(rank)
        a = float(np.mean(error**2))
        predicted = float(np.mean((error+delta[-1])**2))
        row = dict(rank=rank,parent_mse=a,predicted_mse=predicted,predicted_ratio=predicted/a)
        valid.append(row)
        write_progress(f'holdout_r{rank:04d}',dict(stage='holdout',**row),baseline=baseline,delta=delta)
        del card,baseline,delta,error
valid_ratio = None if not valid else float(np.mean([r['predicted_mse'] for r in valid])/np.mean([r['parent_mse'] for r in valid]))
passed = bool(valid and valid_ratio <= .80 and max(r['predicted_ratio'] for r in valid) <= .90)
result = dict(**selection,valid_rows=valid,valid_predicted_mean_ratio=valid_ratio,
              information_gate_pass=passed,wall_s=time.time()-start,
              max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              qualification='Fit-only complete12-column local-response solve with fixed regularization and radius. Not an actual nonlinear gain or score. Four holdouts excluded from this update selection, not from every historical experiment. No candidate, bill, panel or submission claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
