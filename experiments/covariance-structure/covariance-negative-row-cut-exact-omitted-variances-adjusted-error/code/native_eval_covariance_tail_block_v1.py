"""Native evaluator plus compact causal-call and untouched-layer checks."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'native_eval.py').read_text()
old='        np.save(out/f\'pred_{rank}.npy\',y)'
new='''        row['covariance_tail_block'] = dict(mod._CTB_STATS)
        row['covariance_tail_block']['kept'] = list(mod._CTB_STATS['kept'])
        bank_tag = 'both_tail_train2_v6' if rank in (35,100) else 'both_tail_train6_v6_routing_control'
        bank_y=np.load(HERE/bank_tag/f'pred_{rank}.npy')
        row['layer0_bytes_exact'] = y[0].tobytes()==bank_y[0].tobytes()
        assert row['layer0_bytes_exact']
        del bank_y
        np.save(out/f'pred_{rank}.npy',y)'''
assert s.count(old)==1;s=s.replace(old,new)
exec(compile(s,str(HERE/'native_eval.py')+':covariance_tail_block_audit','exec'))
