import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE, write, seal_check
from snapshot import clone_fields, factored_reference

seal_check()
rng = np.random.default_rng(2026092069)
rows = []
for n, rank, strided, ill in [(9, 4, False, False), (9, 4, True, False), (12, 5, False, True)]:
    raw = rng.normal(size=(n, rank * (2 if strided else 1))).astype(np.float32)
    if ill:
        raw *= np.logspace(0, -7, rank).astype(np.float32)
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        source = fnp.asarray(raw)
        y = source[:, ::2] if strided else source
        before = fnp.copy(y)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        copied = clone_fields(dict(Y=y))['Y']
    copy_flops = ctx.summary_dict()['flops_used']
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        q, r = fnp.linalg.qr(y, mode='reduced')
        qc, rc = fnp.linalg.qr(copied, mode='reduced')
        fnp.fill_diagonal(y, -1000.)
    yy, qq, rr = (np.asarray(x).copy() for x in [copied, q, r])
    assert not np.shares_memory(np.asarray(y), np.asarray(copied))
    assert yy.tobytes() == np.asarray(before).tobytes()
    assert qq.tobytes() == np.asarray(qc).tobytes() and rr.tobytes() == np.asarray(rc).tobytes()
    q64, r64 = np.linalg.qr(yy.astype(np.float64), mode='reduced')
    assert qq.tobytes() == q64.astype(np.float32).tobytes()
    assert rr.tobytes() == r64.astype(np.float32).tobytes()
    fields = dict(J=rng.normal(size=(n, 7)).astype(np.float32), weight=rng.random(7).astype(np.float32),
                  Omega=rng.normal(size=(n, rank)).astype(np.float32),
                  Qp=rng.normal(size=(n, rank)).astype(np.float32), core=np.eye(rank, dtype=np.float32))
    ref = factored_reference(fields, np)
    j, w, om, qp, c = [fields[k].astype(np.float64) for k in ['J', 'weight', 'Omega', 'Qp', 'core']]
    direct = ((j * w) @ j.T + qp @ c @ qp.T) @ om
    rel = float(np.linalg.norm(ref - direct) / np.linalg.norm(direct))
    assert rel < 1e-12
    perm = rng.permutation(n)
    a, _ = np.linalg.qr(ref)
    b, _ = np.linalg.qr(ref[perm]); b = b[np.argsort(perm)]
    err = float(np.linalg.norm(a @ a.T - b @ b.T) / np.sqrt(rank))
    assert err < 1e-12
    rows.append(dict(n=n, rank=rank, strided=strided, ill_conditioned=ill, copy_flops=copy_flops,
                     copy_storage_independent=True, QR_rounding_exact=True, gram_identity_relative=rel,
                     row_permutation_projector_relative=err))
write(HERE / 'unit.json', dict(passed=True, fixtures=rows, targets_opened=False))
print(dict(passed=True, fixtures=rows), flush=True)
