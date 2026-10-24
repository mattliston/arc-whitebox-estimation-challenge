"""One matched200-row/40-epoch round after both fixed feedback screens pass."""
import json
from pathlib import Path

here=Path(__file__).resolve().parent
for rank in (35,100):
    trial=json.loads((here/f'centered_feedback_r{rank}_v1.json').read_text())
    assert trial['passes_predeclared_10pct_gate'] and trial['early_layer0_max_abs']==0
runner=here/'chain_radial_training.py'
source=runner.read_text()
edits={
    'rows_radial_covmetric_v1':'rows_centered_feedback_v1',
    'ASTRA_RADIALCOV_V1_C64_E40':'ASTRA_CENTERED_FEEDBACK_V1_C64_E40',
    'chain_radial_training_v1.json':'chain_centered_feedback_training_v1.json',
    'payload_radialcov_v1.npz':'payload_centered_feedback_v1.npz',
    'str(HERE/"generate_radial_rows.py")':'str(HERE/"generate_centered_feedback_rows_v1.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
        'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_compact.py","radial_compact_radialfeed_v1.py","radial_mehler.py","radial_features.py","radial_centered_feedback_v1.py","generate_radial_rows.py","generate_centered_feedback_rows_v1.py")',
    'row_seconds = time.time()-started':
        'profile=[sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\n'
        'assert 1e-10<profile[-1]<2*profile[10],("unhealthy honest rows",profile)\n'
        'print("CENTERED_FEEDBACK_HONEST_PROFILE",profile,flush=True)\n'
        'row_seconds = time.time()-started',
}
for old,new in edits.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(runner),'exec'),dict(__file__=str(runner),__name__='__main__'))
