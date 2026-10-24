"""Native scalar-star trace screen with call census and unchanged layer0 check."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
build=json.loads((HERE/'star_residual_trace_build_v1.json').read_text())
source=(HERE/'native_eval.py').read_text()
marker='class MLP:\n'
source=source.replace(marker,"assert ranks in [[35,100],[36,45,52,58,64,70]]\nassert hashlib.sha256(candidate.read_bytes()).hexdigest()==build['candidate_sha256']\n"+marker)
marker="        np.save(out/f'pred_{rank}.npy',y)"
assert source.count(marker)==1
insert='''        row['star_residual_trace_calls'] = mod._LRST_STATS['calls']
        assert row['star_residual_trace_calls'] == 15*(ranks.index(rank)+1)
        base_tag = 'both_tail_train2_v6' if rank in [35,100] else 'both_tail_train6_v6_routing_control'
        parent_layer0 = np.load(HERE/base_tag/f'pred_{rank}.npy')[0]
        row['layer0_bytes_exact'] = y[0].tobytes() == parent_layer0.tobytes()
        assert row['layer0_bytes_exact']
'''
source=source.replace(marker,insert+marker)
exec(compile(source,str(HERE/'native_eval.py'),'exec'),globals())
