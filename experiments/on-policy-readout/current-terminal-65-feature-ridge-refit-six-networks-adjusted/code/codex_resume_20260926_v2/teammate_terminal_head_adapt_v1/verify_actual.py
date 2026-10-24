"""Post-score invariants; does not change the generic bank advance gates."""
import numpy as np
from common import *


def main():
    verify();fit=read(HERE/'fit_result.json')
    verify_pin(fit['candidate']);verify_pin(fit['disabled_control'])
    assert pin(HERE/'disabled_control.py')['sha256']==pin(COMPONENT)['sha256']
    rows=[]
    for suffix,ranks,thin in [('train2',(35,100),'thin_star_route_train2_v1'),
                              ('train6',(36,45,52,58,64,70),'thin_star_route_train6_v1')]:
        directory=H2/f'{HERE.name}_terminal_head_{suffix}_v1'
        if not directory.exists():continue
        manifest=read(directory/'manifest.json')
        assert manifest['sha256']==fit['candidate']['sha256'] and manifest['ranks']==list(ranks)
        target_rows={r['rank']:r for r in map(__import__('json').loads,(directory/'rows.jsonl').read_text().splitlines())}
        old_rows={r['rank']:r for r in map(__import__('json').loads,(H2/thin/'rows.jsonl').read_text().splitlines())}
        assert set(target_rows)==set(ranks)
        for rank in ranks:
            candidate=np.load(directory/f'pred_{rank}.npy',allow_pickle=False)
            parent=np.load(H2/thin/f'pred_{rank}.npy',allow_pickle=False)
            assert candidate[:15].tobytes()==parent[:15].tobytes()
            assert target_rows[rank]['flops']==old_rows[rank]['flops']
            rows.append(dict(rank=rank,first15_exact=True,complete_bill_equal_to_thin=True))
    assert len(rows) in (2,8)
    once(HERE/'actual_invariants.json',dict(passed=True,rows=rows,
          generic_result=pin(HERE/'result.json'),official_bank_gates_unchanged=True,
          zero_added_prediction_arithmetic_verified=True))


if __name__=='__main__':main()
