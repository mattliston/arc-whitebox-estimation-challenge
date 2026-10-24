"""New32-source matched pilot; reuse fixed training/parity machinery."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'train_final_variance_source_probe_v1.py').read_text()
source = source.replace('variance_source', 'cohort_source')
for before, after in (
    ('==(14,)', '==(32,)'),
    ('torch.manual_seed(930062)', 'torch.manual_seed(930065)'),
    ('torch.nn.Linear(140,128)', 'torch.nn.Linear(158,128)'),
    ('14 variance-source channels', '32 signed-pair cohort/star/trace channels'),
):
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(HERE/'train_final_variance_source_probe_v1.py'), 'exec'),
     dict(__file__=str(HERE/'train_final_cohort_source_probe_v1.py'), __name__='__main__'))
