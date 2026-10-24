"""Compare symmetric covariance actions, as required by the frozen target.

Version1 stopped before output: raw finite covariance response differs by
1.16e-12 because the parent's float32 covariance push is not bit-symmetric.
The source-v2 target and analytic response are explicitly symmetrized.
An antisymmetric matrix has zero w.T E w for every real w. Version2 checks
the matching symmetric finite action without loosening the1e-12 gate.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'check_double_skew_tree_value_v1.py').read_text()
changes = (
    ("double_skew_tree_value_r35_v1.json", "double_skew_tree_value_r35_v2.json"),
    ("        cov_error = float(np.max(np.abs((finite[1]-C)-dc)))",
     "        finite_dc = finite[1]-C\n"
     "        asymmetric = float(np.max(np.abs(finite_dc-finite_dc.T)))\n"
     "        finite_dc = (finite_dc+finite_dc.T)/2\n"
     "        cov_error = float(np.max(np.abs(finite_dc-dc)))"),
    ("                   finite_cov_max_error=cov_error, baseline_covariance_energy=bce,",
     "                   finite_cov_max_error=cov_error, raw_finite_cov_asym_max=asymmetric, baseline_covariance_energy=bce,"),
    ("Finite pair-map response certified against full recomputation.",
     "Symmetric finite pair-map response certified against symmetrized full recomputation, matching the symmetric target; raw float32-induced antisymmetry is separately reported and has zero quadratic action."),
)
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(HERE/'check_double_skew_tree_value_v1.py'), 'exec'),
     dict(__file__=str(HERE/'check_double_skew_tree_value_v2.py'), __name__='__main__'))
