from common import *
verify();guarded('unit',120,'unit.py')
for rank in TRAIN+VALID:guarded(f'data_{rank:04d}',120,'data_prep.py',rank)
records=[read(HERE/f'data_{rank:04d}.json') for rank in TRAIN+VALID]
write(HERE/'data_sealed.json',dict(pins=[r['arrays'] for r in records]+[pin(HERE/f'data_{rank:04d}.json') for rank in TRAIN+VALID]+[pin(HERE/'bank_parameters.npz')],
    prepared_before_fits=True,train=list(TRAIN),valid=list(VALID)))
for arm in ('output_only','nonlinear'):guarded(f'fit_{arm}',1800,'fit.py',arm)
write(HERE/'models_sealed.json',dict(pins=[pin(HERE/f'{arm}_{name}') for arm in ('output_only','nonlinear') for name in ('parameters.npz','fit.json')],
    before_held_prediction=True,final_epochs_only=True))
guarded('predict',180,'predict.py');guarded('score',180,'score.py')
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'result.json')['passed'],
    guarded_names=['unit']+[f'data_{rank:04d}' for rank in TRAIN+VALID]+['fit_output_only','fit_nonlinear','predict','score']))
print(dict(completed=True,passed=read(HERE/'result.json')['passed']),flush=True)
