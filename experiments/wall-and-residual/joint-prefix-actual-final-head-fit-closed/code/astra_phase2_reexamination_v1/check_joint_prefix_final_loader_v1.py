"""Target-free constructor lineage/parity check before fixed joint training."""
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard
from joint_prefix_final_data_v1 import HERE,ManifestFullCard,weight_path,sha


def main():
    out = HERE/'joint_prefix_final_loader_v1.json'
    assert not out.exists()
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    torch.set_num_threads(3)
    started = time.time()
    records = []
    for rank in (35,100):
        oldpath = HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'weights.npy'
        newpath = weight_path(rank)
        equal_weights = np.array_equal(np.load(oldpath,mmap_mode='r'),np.load(newpath,mmap_mode='r'))
        old = ExplicitBoundBankFullCard(rank,'mps',audit_selections=True)
        new = ManifestFullCard(rank,old.P0,old.Q0,audit_selections=True)
        fixed_equal = torch.equal(old.fixed,new.fixed)
        with torch.no_grad():
            a,_ = old.run()
            b,_ = new.run()
            equal_pred = torch.equal(a,b)
            maximum = float((a-b).abs().max().cpu())
        selections_equal = len(old.selections)==len(new.selections)==30 and all(
            np.array_equal(oi,ni) and np.array_equal(om,nm)
            for (oi,om),(ni,nm) in zip(old.selections,new.selections))
        record = dict(rank=rank,weights_equal=bool(equal_weights),fixed_features_equal=bool(fixed_equal),
                      predictions_equal=bool(equal_pred),maximum_mean_error=maximum,
                      selections_equal=bool(selections_equal),weight_file_sha256=sha(newpath))
        record['passed'] = all((equal_weights,fixed_equal,equal_pred,selections_equal))
        records.append(record)
        print(json.dumps(record),flush=True)
        del old,new,a,b
        gc.collect();torch.mps.empty_cache()
    result = dict(passed=all(r['passed'] for r in records),records=records,target_access=False,
                  utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-started,
                  source_sha256=sha(__file__),loader_sha256=sha(HERE/'joint_prefix_final_data_v1.py'),
                  protocol_sha256=sha(HERE/'JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md'),
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['passed']


if __name__=='__main__':
    main()
