"""Use pre-extracted target-blind weights; original numerical capture unchanged."""
from common import *
import capture
def load_weights(ordinal):
    import numpy as np
    r=read(HERE/f'weights_mini{ordinal:04d}.json');assert r['passed'] and r['target_values_decoded']==0
    check(r['weights']);W=np.load(r['weights']['path'],allow_pickle=False)
    assert W.shape==(16,1024,1024) and W.dtype==np.float32
    return W,Path(r['weights']['path']),r['provenance']
verify('runtime_repair_sealed.json')
capture.get_weights=load_weights
capture.main()
