"""Versioned coherent-8M protocol; retain v1 implementation and artifacts."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here / 'generate_covariance_innovation_probe_v1.py').read_text()
old = "mean_truth=np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)\ncov_truth=np.load(g19/f'mps_cov2_rank{args.rank:04d}.npz')"
new = "cov_truth=np.load(g19/f'mps_cov2_rank{args.rank:04d}.npz')\nassert int(cov_truth['P']) == 8000000\nmean_truth=cov_truth['mc_post'].astype(np.float64)"
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace('rows_covariance_innovation_probe_v1', 'rows_covariance_innovation_probe_v2')
source = source.replace("provenance='Current input matrices", "mean_and_covariance_samples=8000000, protocol='coherent_8m_v2',\n    provenance='Current input matrices")
exec(compile(source, __file__, 'exec'))
