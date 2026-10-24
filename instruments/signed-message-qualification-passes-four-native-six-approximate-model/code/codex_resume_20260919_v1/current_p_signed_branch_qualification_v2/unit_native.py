from common import *
admission()
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import model_parent as parent
import model_branch as branch

old, a, c, edge = parent.parameters(np)
new, aa, cc, ee = branch.parameters(np)
assert set(new) == set(old) | {'ungated_w'}
assert all(new[k].tobytes() == v.tobytes() for k, v in old.items())
assert new['ungated_w'].shape == (128, 48) and not np.count_nonzero(new['ungated_w'])
assert a.tobytes() == aa.tobytes() and c.tobytes() == cc.tobytes()
assert all(edge[k].tobytes() == ee[k].tobytes() for k in edge)
rng = np.random.default_rng(9409213)
fixtures = []
with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
    op = {k:fnp.asarray(v) for k,v in old.items()}
    bp = {k:fnp.asarray(v) for k,v in new.items()}
    env0 = parent.source_environment(fnp, flops.stats.norm.cdf, None)
    env1 = branch.source_environment(fnp, flops.stats.norm.cdf, None)
    for dtype in [np.float32, np.float64]:
        for layer in [0, 1, 15]:
            x = fnp.asarray((old['centers'][layer] + .2 * old['scales'][layer] * rng.normal(size=(8,30))).astype(dtype))
            w = fnp.asarray((rng.normal(size=(8,8)) * .2).astype(dtype))
            state = fnp.asarray((rng.normal(size=(8,64)) * .1).astype(np.float32))
            sigma = fnp.asarray(np.full(8, .7, dtype=dtype))
            left = env0['gnn_step'](layer, x, sigma, w, state, op)
            right = env1['gnn_step'](layer, x, sigma, w, state, bp)
            assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x,y in zip(left,right))
            fixtures.append(dict(layer=layer, dtype=str(np.dtype(dtype)), output_and_state_bytes_exact=True))
    # A synthetic representation fixture, not a real-network error claim.
    layer = 1
    x = fnp.asarray(np.broadcast_to(old['centers'][layer], (8,30)).copy())
    w = fnp.asarray((rng.normal(size=(8,8)) * .2).astype(np.float32))
    state = fnp.asarray((rng.normal(size=(8,64)) * .1).astype(np.float32))
    sigma = fnp.asarray(np.full(8, .7, dtype=np.float32))
    op['msg_b'] = fnp.zeros_like(op['msg_b'])
    bp['msg_b'] = fnp.zeros_like(bp['msg_b'])
    bp['ungated_w'] = fnp.asarray((rng.normal(size=(128,48)) * .05).astype(np.float32))
    baseline_plus = env0['gnn_step'](layer,x,sigma,w,state,op)
    baseline_minus = env0['gnn_step'](layer,x,sigma,w,-state,op)
    branch_plus = env1['gnn_step'](layer,x,sigma,w,state,bp)
    branch_minus = env1['gnn_step'](layer,x,sigma,w,-state,bp)
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x,y in zip(baseline_plus,baseline_minus))
    assert np.asarray(branch_plus[1]).tobytes() != np.asarray(branch_minus[1]).tobytes()
    assert all(np.isfinite(np.asarray(x)).all() for x in branch_plus + branch_minus)
write(HERE / 'unit_native_result.json', dict(passed=True, fixtures=fixtures,
    zero_branch_preserves_old_parameters=True, synthetic_zero_gate_sign_distinguished=True,
    representation_fixture_is_not_real_error_evidence=True, complete_unit_bill=ctx.summary_dict(),
    targets_decoded=False, candidate_changed=False))
print(dict(passed=True, fixtures=len(fixtures), synthetic_sign_fixture=True), flush=True)
