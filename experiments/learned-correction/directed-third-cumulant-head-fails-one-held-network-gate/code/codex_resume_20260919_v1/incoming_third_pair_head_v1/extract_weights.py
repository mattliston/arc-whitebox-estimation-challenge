import resource
from common import *
from capture import get_weights
import numpy as np
from threadpoolctl import threadpool_info
verify();verify('runtime_repair_sealed.json')
assert all(x['num_threads']==1 for x in threadpool_info())
ordinal=int(sys.argv[1]);W,wp,provenance=get_weights(ordinal)
write(HERE/f'weights_mini{ordinal:04d}.json',dict(passed=True,weights=pin(wp),provenance=provenance,
      target_values_decoded=0,python=sys.version,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(ordinal=ordinal,weights=pin(wp)),flush=True)
