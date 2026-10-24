"""Fixed eight-feature matched final-head fit, independently versioned."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'train_final_variance_source_probe_v1.py').read_text()
source = source.replace('final_variance_source', 'gaussian_shadow_disagreement')
for before, after in (
    ('==(14,)', '==(8,)'),
    ('torch.manual_seed(930062)', 'torch.manual_seed(930073)'),
    ('torch.nn.Linear(140,128)', 'torch.nn.Linear(134,128)'),
    ('parent_ratio<=.90 and control_ratio<=.92', 'parent_ratio<=.65 and control_ratio<=.70'),
    ('14 variance-source channels', '8 independent Gaussian-trajectory disagreement channels'),
):
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(HERE/'train_final_variance_source_probe_v1.py'), 'exec'),
     dict(__file__=str(Path(__file__).resolve()), __name__='__main__'))
