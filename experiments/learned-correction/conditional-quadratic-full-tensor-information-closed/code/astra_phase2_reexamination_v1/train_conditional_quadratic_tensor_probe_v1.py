"""Fixed new-observable information gate; old closed experiment unmodified."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here/'train_conditional_skew_energy_probe_v1.py').read_text()
replacements = {
    'conditional_skew_energy_probe_verdict_v1.json': 'conditional_quadratic_tensor_probe_verdict_v1.json',
    'conditional_skew_energy_probe_models_v1.pt': 'conditional_quadratic_tensor_probe_models_v1.pt',
    'conditional_skew_energy_probe_rows_v1.json': 'conditional_quadratic_tensor_probe_rows_v1.json',
    "here/'rows_conditional_skew_energy_probe_v1'/f'rank_{rank:04d}.npy'":
        "here/'rows_conditional_quadratic_tensor_probe_v1'/f'row_{rank:04d}'/'features.npy'",
    '5290723': '951059',
    'torch.nn.Linear(130,64)': 'torch.nn.Linear(134,64)',
}
for old, new in replacements.items():
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
exec(compile(source, __file__, 'exec'), globals())
