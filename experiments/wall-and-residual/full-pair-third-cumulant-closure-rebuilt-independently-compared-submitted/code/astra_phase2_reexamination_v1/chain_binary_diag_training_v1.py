"""One bounded coadaptation of the better oracle-profile feedback rule."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/"chain_radial_training.py").read_text()
replacements={
    'rows_radial_covmetric_v1':'rows_binary_diagfeedback_v1',
    'ASTRA_RADIALCOV_V1_C64_E40':'ASTRA_BINARY_DIAGFEEDBACK_V1_C64_E40',
    'chain_radial_training_v1.json':'chain_binary_diag_training_v1.json',
    'payload_radialcov_v1.npz':'payload_binary_diagfeedback_v1.npz',
    'str(HERE/"generate_radial_rows.py")':'str(HERE/"generate_binary_diag_rows.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
    'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_cubic.py","radial_binary_diagfeedback.py","radial_features.py","generate_radial_rows.py","generate_binary_diag_rows.py")',
    'row_seconds = time.time()-started':
    'profile = [sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\nassert 1e-10 < profile[-1] < 2*profile[10], ("unhealthy honest-row profile",profile)\nprint("BINARY_DIAG_HONEST_PROFILE",profile,flush=True)\nrow_seconds = time.time()-started'
}
for old,new in replacements.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(HERE/"chain_radial_training.py"),"exec"),dict(__file__=str(HERE/"chain_radial_training.py"),__name__="__main__"))
