"""Judge-side observer only: plain NumPy host export, no estimator arithmetic."""
from pathlib import Path
import numpy as np

def make_observer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    def observe(record, p):
        layer = int(record['layer']); old = int(p['old_columns'])
        path = directory / ('L%02d.npz' % layer)
        assert not path.exists()
        # np.savez streams members without retaining previous layers. No copies
        # or projections are added to the estimator's FNP numerical path.
        r = p['refine']
        values = dict(left_young=np.asarray(p['left'])[:, old:],
                      right_young=np.asarray(p['right']),
                      lambda_young=np.asarray(p['lam'])[old:],
                      rows=np.asarray(p['rows']), cols=np.asarray(p['cols']),
                      mu=np.asarray(p['mu']), vc=np.asarray(p['vc']),
                      gate=np.asarray(p['gate']), Q=np.asarray(r['Q']),
                      prefix=np.asarray(r['prefix']),
                      c15=np.asarray(record['c15']), c2=np.asarray(record['c2']))
        if r['old_inner'] is not None:
            values.update(old_inner=np.asarray(r['old_inner']),
                          old_right=np.asarray(r['old_right']))
        np.savez(path, **values)
    return observe
