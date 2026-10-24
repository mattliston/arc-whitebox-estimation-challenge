"""Five target-free common-cubic calls, using the unchanged RD lifetime guard."""
import ast
import gc
import hashlib
import io
from pathlib import Path
import tokenize

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_receiver_directions_v1/qualification.py'
OWNER_SHA = 'a964c0f7a0ff5f1725da2d68983452d4fbae6e58e3579a31f2bf0179f4a66dd9'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def common_tiny(np, torch, model, fixture, save):
    """Only the new function; compare its plane with complete stored cubes."""
    rng = np.random.Generator(np.random.PCG64DXSM(9309102))
    n = 8
    bn = (.25*rng.standard_normal((n, n))).astype(np.float32)
    np.fill_diagonal(bn, 0.)
    bn[[2, 6], :] = 0.
    hn = (.2*rng.standard_normal((n, 64))).astype(np.float32)
    an = np.asarray([.1, .3, 1e-7, .2, 1., .01, .6, 0.], dtype=np.float32)
    tn = np.asarray([2.5, -.3, 0., -8., 1e-21, 1., 0., 1e-8], dtype=np.float32)
    sn = np.asarray([1.4, -.7, 0., -8.15, 1e-21, -.2, 0., -1e-8], dtype=np.float32)
    mask = np.asarray([1, 0, 1, 1, 0, 1, 0, 1.], dtype=np.float32)
    chosen = np.flatnonzero(mask).astype(np.int64)
    plane_cot = rng.standard_normal((n, n))
    plus_cot, minus_cot = (rng.standard_normal(n) for _ in range(2))
    records = {}
    cpu_gradient = cpu_plane = cpu_plus = cpu_minus = None
    for device in ('cpu', 'mps'):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        t32 = lambda value, grad=False: torch.tensor(value, dtype=torch.float32,
            device=device, requires_grad=grad)
        array = lambda value: value.detach().cpu().contiguous().numpy().copy()
        B, H = t32(bn, True), t32(hn, True)
        a, t, s = t32(an, True), t32(tn, True), t32(sn, True)
        births = torch.tensor(mask, dtype=dtype, device=device)
        selected = torch.tensor(chosen, dtype=torch.int64, device=device)
        pc = torch.tensor(plane_cot, dtype=dtype, device=device)
        ac, bc = (torch.tensor(value, dtype=dtype, device=device)
                  for value in (plus_cot, minus_cot))
        zero = torch.zeros(65, dtype=torch.float32, device=device, requires_grad=True)
        response = model.common_response(H, zero)
        p0, m0 = model.common_weights(a, t, s, response)
        k0 = model.common_plane(B, a, t, s, p0, m0, selected, births, dtype)
        require(np.array_equal(array(p0), an) and np.array_equal(array(m0), -an)
            and np.all(array(k0) == 0.), 'Live zero action has original weight values and zero plane')
        nz = an != 0
        require(array(p0)[nz].tobytes() == an[nz].tobytes()
            and array(m0)[nz].tobytes() == (-an[nz]).tobytes(),
            'Nonzero original weight bytes unchanged; signed zero follows prescribed live arithmetic')
        loss0 = ((k0*pc).sum()+.01*((p0.to(dtype)*ac).sum()+(m0.to(dtype)*bc).sum()))/(n*n)
        loss0.backward(); zero_gradient = array(zero.grad)
        require(np.isfinite(zero_gradient).all() and np.linalg.norm(zero_gradient) > 0,
            'Zero-Q formula has a connected finite gradient, not a bypass')
        for value in (B, H, a, t, s):
            value.grad = None
        Q = t32(fixture, True)
        response1 = model.common_response(H, Q)
        plus, minus = model.common_weights(a, t, s, response1)
        plane = model.common_plane(B, a, t, s, plus, minus, selected, births, dtype)
        loss = ((plane*pc).sum()+.01*((plus.to(dtype)*ac).sum()+(minus.to(dtype)*bc).sum()))/(n*n)
        loss.backward()
        gradients = {name: array(value.grad) for name, value in
            (('Q', Q), ('H', H), ('B', B), ('a', a), ('t', t), ('s', s))}
        require(all(np.isfinite(value).all() and np.linalg.norm(value) > 0
            for value in gradients.values()), 'Connected finite Q/H/B/a/t/s gradients')
        p, m, k = array(plus), array(minus), array(plane)
        require(all(np.isfinite(value).all() for value in (p, m, k)), 'Finite small/zero weight and t=s cases')
        limit = .25*np.abs(an.astype(np.float64)) + 4*np.finfo(np.float32).eps*np.abs(an.astype(np.float64))
        require(np.all(np.abs(p.astype(np.float64)-an) <= limit)
            and np.all(np.abs(m.astype(np.float64)+an) <= limit), 'Bounded stored weight increments including rounding')
        dplus = p.astype(np.float64)-an.astype(np.float64)
        dminus = m.astype(np.float64)+an.astype(np.float64)
        ds = [(dplus*tn.astype(np.float64)**r+dminus*sn.astype(np.float64)**r)*mask
              for r in range(4)]
        pcol, mcol = bn.astype(np.float64).copy(), bn.astype(np.float64).copy()
        np.fill_diagonal(pcol, tn); np.fill_diagonal(mcol, sn)
        changed = np.zeros((n, n, n), dtype=np.float64)
        for j in chosen:
            changed += dplus[j]*np.einsum('i,j,k->ijk', pcol[:, j], pcol[:, j], pcol[:, j])
            changed += dminus[j]*np.einsum('i,j,k->ijk', mcol[:, j], mcol[:, j], mcol[:, j])
        literal = np.einsum('iij->ij', changed)
        error = float(np.max(np.abs(literal-k.astype(np.float64))))
        bound = (1e-12 if device == 'cpu' else 2e-6)*max(1., float(np.max(np.abs(literal))))
        require(error <= bound, 'Full actual stored-cube plane, including nonzero diagonal and rounding d1')
        star_A = -3*literal.T+2*np.diag(np.diag(literal))
        restored = changed.copy()
        repeated = np.empty((n, n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                for kindex in range(n):
                    repeated[i, j, kindex] = i == j or i == kindex or j == kindex
                    restored[i, j, kindex] += (star_A[i, j]*(j == kindex)
                        +star_A[j, i]*(i == kindex)+star_A[kindex, i]*(i == j))/3
        require(np.max(np.abs(restored[repeated])) <= 1e-12
            and np.array_equal(restored[~repeated], changed[~repeated])
            and np.linalg.norm(changed[~repeated]) > 0,
            'Present repeated entries restored while the actual distinct source survives')
        record = dict(passed=True, null_values_exact=True, null_nonzero_weight_bytes_exact=True,
            zero_Q_gradient_connected=True, full_stored_cube_plane_max_error=error,
            full_stored_cube_plane_limit=bound, repeated_plane_restored=True,
            distinct_increment_nonzero=True, actual_d1_max_abs=float(np.max(np.abs(ds[1]))),
            plus=save('tiny_'+device+'_plus', p), minus=save('tiny_'+device+'_minus', m),
            plane=save('tiny_'+device+'_plane', array(plane)),
            zero_Q_gradient=save('tiny_'+device+'_zero_Q_gradient', zero_gradient),
            gradients={name: save('tiny_'+device+'_'+name+'_gradient', value)
                       for name, value in gradients.items()})
        if device == 'cpu':
            cpu_gradient = gradients['Q'].astype(np.float64)
            cpu_plane, cpu_plus, cpu_minus = array(plane), p, m
        else:
            relative = float(np.linalg.norm(gradients['Q'].astype(np.float64)-cpu_gradient)
                / max(np.linalg.norm(cpu_gradient), 1e-12))
            require(relative <= .05, 'Tiny CPU64/MPS32 Q-gradient agreement<=5%')
            record['CPU64_MPS32_Q_gradient_relative'] = relative
        records[device] = record
        del B, H, a, t, s, births, selected, pc, ac, bc, zero, response, p0, m0, k0, loss0
        del Q, response1, plus, minus, plane, loss, value
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    return dict(passed=True, no_card_calls=True, n=n, seed=9309102,
        null_values_exact_both_devices=True, zero_Q_gradient_connected=True,
        endogenous_H_and_physical_gradients_connected=True, full_stored_pair_plane_checked=True,
        CPU64_MPS32_Q_gradient_relative=relative, Q_gradient_limit=.05,
        native_gradient_qualified=False, records=records)


def renamed(source, before, after):
    items = list(tokenize.generate_tokens(io.StringIO(source).readline))
    require(not any(t.type == tokenize.NAME and t.string == after for t in items),
        'Unused parameter identifier before exact rename')
    require(any(t.type == tokenize.NAME and t.string == before for t in items), 'Original parameter identifier present')
    return tokenize.untokenize([t._replace(string=after) if t.type == tokenize.NAME
        and t.string == before else t for t in items])


def derived_source():
    require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA, 'Frozen qualified admission/lifetime owner')
    original = OWNER.read_text()
    nodes = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == 'tiny']
    require(len(nodes) == 1, 'One original tiny function')
    node = nodes[0]
    old_tiny = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    changes = (
        (old_tiny, 'def tiny(np, torch, model, fixture, save):\n    return common_tiny(np, torch, model, fixture, save)\n'),
        ('PINS = {\n', "PINS = {\n    NEW/'current_comp_receiver_directions_v1/qualification.py': '"+OWNER_SHA+"',\n"),
        ("'current_comp_receiver_directions_qualification_v1'", "'current_comp_common_cubic_qualification_v1'"),
        ("'qualified_receiver_directions_model'", "'qualified_common_cubic_model'"),
        ('180s whole receiver-direction qualification', '180s whole common-cubic qualification'),
        ("require(set(d['node_head_calls']) == set(d['receiver_head_calls']) == {str(l) for l in range(1, 15)},",
         "require(set(d['node_head_calls']) == {str(l) for l in range(1, 15)}\n"
         "                and set(d['common_head_calls']) == {str(l) for l in range(10, 15)},"),
        ('Every actual original NODE and new receiver birth call', 'Every original NODE and prescribed five common-cubic calls'),
        ('receiver_R=card.R', 'common_Q=card.R'),
        ('p, q = np.arange(1, 65, dtype=np.float64)[:, None], np.arange(1, 5, dtype=np.float64)[None, :]',
         'p, q = np.arange(1, 66, dtype=np.float64), .19/.19'),
        ("result['R_fixture'] = save('fixed_nonzero_R', fixture)", "result['Q_fixture'] = save('fixed_nonzero_Q', fixture)"),
        ('trainable_R=False', 'trainable_Q=False'),
        ("'receiver_backward'", "'common_backward'"),
        ("'receiver_backward_repeat'", "'common_backward_repeat'"),
        ('Sole actual256-coordinate receiver leaf, all parent parameters frozen', 'Sole actual65-coordinate common-Q leaf, all parent parameters frozen'),
        ('Owned detached full receiver gradient', 'Owned detached full common-Q gradient'),
        ('gradient.shape == (64, 4)', 'gradient.shape == (65,)'),
        ('Finite nonzero256-coordinate R gradient', 'Finite nonzero65-coordinate Q gradient'),
        ('gradient_coordinates=256', 'gradient_coordinates=65'),
        ("f'call_{index:02d}_R_gradient'", "f'call_{index:02d}_Q_gradient'"),
        ('zero_lr_Adam_R_byte_exact=True', 'zero_lr_Adam_Q_byte_exact=True'),
        ('null_forwards=2, receiver_backwards=3', 'null_forwards=2, common_backwards=3'),
        ("result['source_proof'] = model.source_proof()", "result['source_proof'] = model.source_proof()\n"
         "        result['qualification_adapter_source_proof'] = ADAPTER_PROOF"),
    )
    source = original
    counts = []
    for before, after in changes:
        count = source.count(before)
        require(count > 0, 'Exact qualification source span: '+before[:90])
        counts.append(count); source = source.replace(before, after)
    changed = renamed(source, 'R', 'Q')
    inverse = renamed(changed, 'Q', 'R')
    require(inverse == source, 'Parameter-identifier-only rename roundtrip')
    for (before, after), count in reversed(list(zip(changes, counts))):
        require(inverse.count(after) == count, 'Exact qualification inverse span: '+after[:90])
        inverse = inverse.replace(after, before)
    require(inverse == original, 'Whole qualified RD worker inverse byte-for-byte')
    ast.parse(changed)
    return changed, dict(owner=str(OWNER), owner_sha256=OWNER_SHA,
        full_source_inverse_verified=True, parameter_identifier_roundtrip=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_admission_scalar_reference_domain_retirement_and_resource_limits=True,
        scope='New stored-common-cube tiny function and65-Q leaf; frozen COMP P/A')


_source, _proof = derived_source()
exec(compile(_source, str(OWNER)+':common-cubic', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'common_tiny': common_tiny, 'ADAPTER_PROOF': _proof})
