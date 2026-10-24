import runpy
from endpoint_common import *
endpoint_verify()
done('seal_predictions')
seal = read(HERE/'lifecycle_receipts_sealed.json')
assert seal['completed'] and seal['before_validation_targets'] and seal['all_85_closed_context_calls_qualified']
for p in seal['pins']:
    check(p)
runpy.run_path(str(HERE/'score_native40.py'),run_name='__main__')
