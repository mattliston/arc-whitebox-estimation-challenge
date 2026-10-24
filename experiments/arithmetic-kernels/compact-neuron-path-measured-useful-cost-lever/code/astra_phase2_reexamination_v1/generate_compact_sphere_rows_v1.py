"""Honest angular-trajectory rows with labels kept in Gaussian output units.

Features are captured on the angular state before truth feedback. Host and
sigma are both multiplied by the same known c_n, so the unchanged trainer
uses Gaussian truths while learning the exact same dimensionless angular
correction. At inference its sigma input is angular and the final output
is multiplied by c_n. No truth or future correction enters any feature.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "generate_radial_rows.py"
source = path.read_text()
edits = {
    "from k3_radial_numpy import predict":
        """from radial_compact_sphere_v1 import predict as angular_predict, gaussian_radius_mean

def predict(W, **kwargs):
    return angular_predict(W, gate_floor=.01, **kwargs)
""",
    "xs,hosts,sigmas = [],[],[]":
        "mean_scale = gaussian_radius_mean(1024)\nxs,hosts,sigmas = [],[],[]",
    '    hosts.append(m.copy()); sigmas.append(aux["sigma"].copy())':
        '    hosts.append(mean_scale*m.copy()); sigmas.append(mean_scale*aux["sigma"].copy())',
    "    return truth[l]-m if l>=1 else np.zeros_like(m)":
        "    return truth[l]/mean_scale-m if l>=1 else np.zeros_like(m)",
    '(out/"result.json").write_text':
        'result.update(internal_trajectory="angular", saved_host_and_sigma_frame="Gaussian", gaussian_radius_mean=mean_scale)\n(out/"result.json").write_text',
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), "exec"), dict(__file__=str(path), __name__="__main__"))
