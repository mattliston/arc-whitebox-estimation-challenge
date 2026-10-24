"""Five target-free P/Q calls; inverse-owned original admission and lifetime guard."""
import ast
import gc
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_receiver_directions_v1/qualification.py'
OWNER_SHA = 'a964c0f7a0ff5f1725da2d68983452d4fbae6e58e3579a31f2bf0179f4a66dd9'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def covariance_tiny(np, torch, model, fixture, save):
    """Only the new small function: exact null, explicit congruence and connectivity."""
    rng = np.random.Generator(np.random.PCG64DXSM(9309106))
    n = 8
    seed_matrix = rng.standard_normal((n, n))
    Cn = seed_matrix@seed_matrix.T/n + np.eye(n)
    Cn[0, 1], Cn[1, 0] = 0., -0.
    Hn = (.2*rng.standard_normal((n, 64))).astype(np.float32)
    cot = rng.standard_normal((n, n))
    qn = fixture[0]
    records = {}
    for device in ('cpu', 'mps'):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        C = torch.tensor(Cn, dtype=dtype, device=device, requires_grad=True)
        H = torch.tensor(Hn, dtype=torch.float32, device=device, requires_grad=True)
        zero = torch.zeros(65, dtype=torch.float32, device=device, requires_grad=True)
        target = torch.tensor(cot, dtype=dtype, device=device)
        array = lambda value: value.detach().cpu().contiguous().numpy().copy()
        null = model.covariance_action(C, H, zero)
        require(array(null).tobytes() == array(C).tobytes(), 'Tiny zero-Q preserves every covariance byte, including signed zero')
        null_loss = (null*target).sum()/(n*n)
        null_loss.backward(); null_gradient = array(zero.grad)
        require(np.isfinite(null_gradient).all() and np.linalg.norm(null_gradient) > 0,
            'Live zero-Q branch has a connected covariance gradient')
        H.grad = None; C.grad = None
        q = torch.tensor(qn, dtype=torch.float32, device=device, requires_grad=True)
        out = model.covariance_action(C, H, q)
        loss = (out*target).sum()/(n*n); loss.backward()
        gradient, h_gradient, c_gradient = array(q.grad), array(H.grad), array(C.grad)
        require(all(np.isfinite(value).all() and np.linalg.norm(value) > 0
            for value in (gradient, h_gradient, c_gradient)), 'Connected finite Q/H/C derivatives without detach')
        with torch.no_grad():
            stored_scale = array(model.covariance_scale(H, q))
        physical = array(C); scale = stored_scale.astype(physical.dtype)
        expected = (physical*scale[:, None])*scale[None, :]
        error = float(np.max(np.abs(array(out).astype(np.float64)-expected.astype(np.float64))))
        limit = (1e-12 if device == 'cpu' else 2e-6)*max(1., float(np.max(np.abs(expected))))
        require(error <= limit and np.isfinite(stored_scale).all() and (stored_scale > 0).all(),
            'Explicit two-sided congruence from actual stored positive scale')
        record = dict(null_byte_exact=True, zero_Q_gradient_connected=True, endogenous_H_gradient_connected=True,
            live_C_gradient_connected=True, congruence_max_abs_error=error, congruence_limit=limit,
            scale=save('tiny_'+device+'_scale', stored_scale), output=save('tiny_'+device+'_output', array(out)),
            Q_gradient=save('tiny_'+device+'_Q_gradient', gradient), H_gradient=save('tiny_'+device+'_H_gradient', h_gradient),
            C_gradient=save('tiny_'+device+'_C_gradient', c_gradient), zero_Q_gradient=save('tiny_'+device+'_zero_Q_gradient', null_gradient))
        if device == 'cpu':
            # Independent smooth derivative of the ideal scalar function;
            # tolerance includes the declared FP32 head and gradient arithmetic.
            z = Hn.astype(np.float64)@qn[:64].astype(np.float64) + float(qn[64])
            tangent = np.tanh(z); s = np.exp(.005*tangent)
            product = Cn*cot
            ds = (product@s+product.T@s)/(n*n)
            dz = ds*s*.005*(1-tangent*tangent)
            explicit = np.concatenate((Hn.astype(np.float64).T@dz, [np.sum(dz)]))
            grad_error = float(np.linalg.norm(gradient.astype(np.float64)-explicit))
            grad_limit = 1e-4*float(np.linalg.norm(explicit))+1e-10
            require(grad_error <= grad_limit, 'Tiny CPU analytic congruence derivative, including FP32 head rounding')
            record.update(analytic_Q_gradient_error=grad_error, analytic_Q_gradient_limit=grad_limit)
        records[device] = record
        del C, H, zero, target, null, null_loss, q, out, loss
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    return dict(passed=True, no_card_calls=True, n=n, seed=9309106,
        null_byte_exact_both_devices=True, null_Q_gradient_connected=True,
        endogenous_H_gradient_connected=True, live_C_gradient_connected=True,
        explicit_congruence_checked=True, CPU_analytic_Q_gradient_checked=True,
        native_gradient_qualified=False, records=records)


BACKWARDS = '''        P, A, initial_zero = model.frozen('mps', trainable_P=True, trainable_Q=False); del initial_zero
        Q = torch.tensor(fixture, dtype=torch.float32, device='mps', requires_grad=True)
        require(Q.is_leaf and Q.requires_grad and not A.requires_grad
            and all(P[key].is_leaf and P[key].requires_grad for key in model.PKEYS),
            'Eleven actual P leaves plus one910-coordinate Q leaf; fixed A')
        initial_P = {key: array(value) for key, value in P.items()}
        initial_A, initial_Q = array(A), array(Q)
        parameters = [P[key] for key in model.PKEYS]+[Q]
        optimizer = torch.optim.Adam(parameters, lr=0., betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
        first = None
        for update, (rank, kind) in enumerate(ROSTER[2:], 1):
            index = result['full_model_calls']; optimizer.zero_grad(set_to_none=True)
            weights = np.load(STORE/f'rows/observer_fit/{rank:04d}/weights.npy', allow_pickle=False)
            card = model.card(rank, P=P, A=A, Q=Q, device='mps', weights=weights); before = frozen_hash(card)
            require(card.Q is Q and card.A is A and all(card.P0[key] is value for key, value in P.items()),
                'Uncopied persistent Q/P/A identities')
            raw, trace = card.run(training=True); served = model._CO.served(raw); rv, sv = array(raw), array(served)
            require(np.isfinite(rv).all() and np.isfinite(sv).all() and (sv >= 0).all()
                and sv[0].tobytes() == references[rank][1][0].tobytes(), 'Finite full output and untouched first Gaussian row')
            forward_d = card.diagnostics()
            loss = scalar(torch, model._CO, raw, cots[rank], card.dtype)
            require(bool(torch.isfinite(loss).detach().cpu()), 'Finite unchanged original synthetic scalar')
            loss.backward(); d = domain(card)
            require(all(d[key] == forward_d[key] for key in ('variance_domain', 'selection_cells')),
                'Every observed checkpoint replay retains the actual current trajectory')
            gradients = {}
            for key, parameter in list(P.items())+[('covariance_Q', Q)]:
                if key in model.PKEYS or key == 'covariance_Q':
                    require(parameter.grad is not None and not parameter.grad.requires_grad and parameter.grad.grad_fn is None,
                        'Owned detached complete P/Q gradient')
                    value = array(parameter.grad)
                    require(value.dtype == np.float32 and np.isfinite(value).all(), 'Finite actual FP32 gradient')
                    if key in model.INACTIVE_KEYS:
                        require((value[0] == 0).all(), 'Original inactive66 gradients remain zero')
                    gradients[key] = value
                else:
                    require(not parameter.requires_grad and parameter.grad is None, 'No fixed-normalizer gradient')
            del parameter
            require(sum(value.size for value in gradients.values()) == 46110
                and gradients['covariance_Q'].shape == (14, 65) and np.linalg.norm(gradients['covariance_Q']) > 0,
                'Complete eleven-P plus910-Q coordinates and connected new action')
            require(A.grad is None and not card.coefficients.requires_grad and card.coefficients.grad is None,
                'Frozen A/cstar receive no gradient')
            require(frozen_hash(card) == before
                and all(array(value).tobytes() == initial_P[key].tobytes() for key, value in P.items())
                and array(A).tobytes() == initial_A.tobytes() and array(Q).tobytes() == initial_Q.tobytes(),
                'No P/Q/A/cstar/weight update during backward')
            if first is None:
                first = dict(raw=rv.copy(), served=sv.copy(), gradients={key: value.copy() for key, value in gradients.items()},
                    variance_domain=d['variance_domain'], selection_cells=d['selection_cells'])
            elif rank == 35:
                require(rv.tobytes() == first['raw'].tobytes() and sv.tobytes() == first['served'].tobytes()
                    and all(value.tobytes() == first['gradients'][key].tobytes() for key, value in gradients.items())
                    and all(d[key] == first[key] for key in ('variance_domain', 'selection_cells')),
                    'Exact repeated nonzero-Q35 full output/domain/P/Q-gradient bytes')
            rec = dict(call=index, rank=rank, kind=kind, raw=save(f'call_{index:02d}_raw', rv),
                served=save(f'call_{index:02d}_served', sv),
                gradient={key: save(f'call_{index:02d}_gradient_{key}', value) for key, value in gradients.items()},
                diagnostics=doc(f'call_{index:02d}_diagnostics', d), gradient_coordinates=46110,
                synthetic_scalar=float(loss.detach().cpu()), frozen_hashes=before)
            optimizer.step()
            require(len(optimizer.state) == len(parameters) == 12
                and all(parameter in optimizer.state and int(optimizer.state[parameter]['step'].item()) == update
                    and all(bool(torch.isfinite(value).all().cpu()) for value in optimizer.state[parameter].values())
                    for parameter in parameters), 'Twelve finite persistent Adam states at steps1/2/3')
            require(frozen_hash(card) == before
                and all(array(value).tobytes() == initial_P[key].tobytes() for key, value in P.items())
                and array(A).tobytes() == initial_A.tobytes() and array(Q).tobytes() == initial_Q.tobytes(),
                'Zero-LR Adam changes no P/Q/A bytes')
            optimizer.zero_grad(set_to_none=True)
            rec.update(passed=True, completed=True, zero_lr_Adam_P_Q_byte_exact=True, Adam_step=update)
            result['calls'].append(rec); doc(f'call_{index:02d}_complete', rec)
            refs = [weakref.ref(card), weakref.ref(raw), weakref.ref(served), weakref.ref(loss)]
            del card, raw, served, trace, loss, weights, rv, sv, d, forward_d, gradients
            retire(index, refs)
        optimizer.state.clear(); optimizer.zero_grad(set_to_none=True); parameters.clear()
        parameter_refs = [weakref.ref(Q), weakref.ref(A)] + [weakref.ref(value) for value in P.values()]
        del optimizer, P, A, Q; gc.collect(); guard()
        require(all(ref() is None for ref in parameter_refs), 'All P/Q/A parameter owners retired at completion')
'''


def derived_source():
    require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA, 'Frozen qualified function/lifecycle owner')
    original = OWNER.read_text()
    nodes = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == 'tiny']
    require(len(nodes) == 1, 'Unique original tiny function')
    node = nodes[0]
    old_tiny = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    begin = "        P, A, initial_zero = model.frozen('mps', trainable_R=False); del initial_zero\n"
    end = "        require(result['full_model_calls'] == result['retired_call_count'] == 5\n"
    require(original.count(begin) == original.count(end) == 1, 'Exact single fixed-parameter backward section')
    old_backwards = original[original.index(begin):original.index(end)]
    changes = (
        (old_tiny, 'def tiny(np, torch, model, fixture, save):\n    return covariance_tiny(np, torch, model, fixture, save)\n', 1),
        (old_backwards, BACKWARDS, 1),
        ("ROSTER = ((35, 'null_forward'), (100, 'null_forward'), (35, 'receiver_backward'),\n"
         "          (100, 'receiver_backward'), (35, 'receiver_backward_repeat'))",
         "ROSTER = ((35, 'null_forward'), (100, 'null_forward'), (35, 'joint_backward'),\n"
         "          (100, 'joint_backward'), (35, 'joint_backward_repeat'))", 1),
        ('PINS = {\n', 'PINS = {\n    NEW/\'current_comp_receiver_directions_v1/qualification.py\': \''+OWNER_SHA+'\',\n', 1),
        ("'current_comp_receiver_directions_qualification_v1'", "'current_comp_joint_p_covariance_qualification_v1'", 1),
        ("'qualified_receiver_directions_model'", "'qualified_joint_covariance_model'", 1),
        ('180s whole receiver-direction qualification', '180s whole joint covariance qualification', 1),
        ("d['receiver_head_calls']", "d['covariance_head_calls']", 1),
        ('Every actual original NODE and new receiver birth call', 'Every original NODE and new covariance head call', 1),
        ('receiver_R=card.R', 'covariance_Q=card.Q', 1),
        ('np.arange(1, 65, dtype=np.float64)[:, None], np.arange(1, 5, dtype=np.float64)[None, :]',
         'np.arange(1, 15, dtype=np.float64)[:, None], np.arange(1, 66, dtype=np.float64)[None, :]', 1),
        ("result['R_fixture'] = save('fixed_nonzero_R', fixture)", "result['Q_fixture'] = save('fixed_nonzero_Q', fixture)", 1),
        ("P, A, R = model.frozen('mps', trainable_R=False)",
         "P, A, Q = model.frozen('mps', trainable_P=False, trainable_Q=False)", 1),
        ("card = model.card(rank, P=P, A=A, R=R, device='mps', weights=weights); before = frozen_hash(card)\n"
         "            require(card.R is R and card.A is A and all(card.P0[k] is value for k, value in P.items()), 'Exact supplied leaf identities')",
         "card = model.card(rank, P=P, A=A, Q=Q, device='mps', weights=weights); before = frozen_hash(card)\n"
         "            require(card.Q is Q and card.A is A and all(card.P0[k] is value for k, value in P.items()), 'Exact supplied leaf identities')", 1),
        ('Null P/A/R/cstar/weights remain frozen', 'Null P/A/Q/cstar/weights remain frozen', 1),
        ('del card, raw, served, trace, P, A, R, weights, rv, sv, d',
         'del card, raw, served, trace, P, A, Q, weights, rv, sv, d', 1),
        ("result['source_proof'] = model.source_proof()", "result['source_proof'] = model.source_proof()\n"
         "        result['qualification_adapter_source_proof'] = ADAPTER_PROOF", 1),
        ('null_forwards=2, receiver_backwards=3', 'null_forwards=2, joint_backwards=3', 1),
    )
    source = original
    for before, after, count in changes:
        require(source.count(before) == count, 'Exact qualified source span: '+before[:90])
        source = source.replace(before, after)
    inverse = source
    for before, after, count in reversed(changes):
        require(inverse.count(after) == count, 'Exact qualification inverse span: '+after[:90])
        inverse = inverse.replace(after, before)
    require(inverse == original, 'Whole frozen receiver qualification source byte inverse')
    ast.parse(source)
    return source, dict(owner=str(OWNER), owner_sha256=OWNER_SHA, full_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        unchanged_admission_reference_scalar_domain_retirement_and_limits=True,
        changed_scope='Q-only tiny function and live eleven-P/one-Q zero-LR qualification')


_source, _proof = derived_source()
exec(compile(_source, str(OWNER)+':joint-P-covariance', 'exec'),
    {'__name__': __name__, '__file__': __file__, 'covariance_tiny': covariance_tiny, 'ADAPTER_PROOF': _proof})
