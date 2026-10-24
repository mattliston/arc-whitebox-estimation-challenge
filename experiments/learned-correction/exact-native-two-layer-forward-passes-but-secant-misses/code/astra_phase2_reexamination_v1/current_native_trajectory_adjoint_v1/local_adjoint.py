"""Inert two-cell fresh-native-state / local real-algebra proposal adjoint.

No numerical imports, files, target, optimizer, parameter updates or execution
at module import. All APIs below run only within the caller's owned scope.

Before card construction: handle=install(model). At runtime the caller supplies
the admitted native-Y model/card, fresh decoded entry8/head7 and end8/head8
Torch14 states, native N selector metadata, served native m9, cotangent and P
direction. evaluate returns only detached/plain diagnostic numbers. It does
not generate directions, admit a secant, authorize fitting or claim native
rounding derivatives. Old failed full-Torch fidelity receipts stay negative.

First evaluate local m9 on FRESH N end8, then obtain cotangents for ALL eleven
continuous boundary coordinates plus direct P9. Retire that graph. Evaluate
the complete original step8 on entry8, pair ALL eleven returned coordinates
with those cotangents, and obtain P8. Sum P8+P9. Native end8 values are never
replaced by Torch step8 values; their discrepancies are reported explicitly.
The retained cotangents are first-order adjoints, not detached carrier state.

Native ordered selector IDs are the current-call discrete branch; inherited
Torch selections are still computed/reported and must have the same membership.
There is no derivative across selection boundaries. A future new P requires a
new native trajectory/branch inventory; this helper is only the fixed entry8
two-cell diagnostic, not a saved-trajectory trainer.

The step9-only cutoff is after exactly `m = host+delta`, preserving every prior
statement and depth16/layer9 behavior. The original served clamp/FP32 cast is
then applied to the scalar objective. Only unused birth/commit output is not
built. Whole-source reconstruction, executed full-step SHA and one surviving
physical clamp are checked. Step8 is the unchanged complete card.step.

work_bound is a conservative source-price certificate, not a flopscope bill:
all matrix forward+two adjoints, QR backward, generous differentiated pointwise
allowance, complete input/gradient/gap copies and cold workspace/feature setup.
The caller owns whole120s/tree8GB/no-swap, actual native-call bills and source
admission. No assertion here substitutes for those measurements.
"""
import gc
import hashlib
import math
import types
import weakref

CONTINUOUS = (0,1,2,3,4,5,7,8,9,10,11)
STATE_NAMES = ('m','C','p_full','lam','star','b','previous_alive','radial',
               'hstate','Q','F','core','stops','previous')
CELLS = ('8:0','8:1','8:2','9:0','9:1')
ENTRY_STOPS = (2048,4096,6144,8128,9984,11712,13376,14912)
LOCAL_WORK_CAP = 1_000_000_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def install(model):
    """Source-only installation before the private model class is compiled."""
    require(model._CLASS is None and hasattr(model,'native_prepared')
            and hasattr(model,'_physical_align_source'), 'Fresh admitted native-Y model, before card class')
    original = model._physical_align_source
    handle = dict(model_identity=id(model),source=None,source_sha256=None)
    def capture(source):
        text,proof = original(source)
        require(handle['source'] is None or handle['source'] == text, 'One identical current step source')
        handle.update(source=text,source_sha256=_digest(text))
        return text,proof
    model._physical_align_source = capture
    return handle


def _m9_function(model, card, handle):
    require(handle['model_identity'] == id(model) and handle['source'] is not None
            and handle['source_sha256'] == model._PROOF['shared_source_sha256']
            and model._PROOF['physical_clamp_count'] == 2
            and card.depth == 16 and card.n == 1024, 'Actual compiled full production step source')
    source = handle['source']; anchor = '    m = host+delta\n'
    require(source.count(anchor) == 1 and source.count('def step(') == 1, 'Unique corrected mean and step owner')
    stop = source.index(anchor)+len(anchor); prefix,tail = source[:stop],source[stop:]
    require(tail.startswith('    if l == self.depth-1:\n'), 'Cut only before original unused return/birth suffix')
    renamed = prefix.replace('def step(','def _nadj_m9(',1)
    cut = renamed+'    return m\n'
    restored = cut[:-len('    return m\n')].replace('def _nadj_m9(','def step(',1)+tail
    require(restored == source, 'Complete positional cutoff inverse; no depth or prefix edit')
    outer = model._CLASS.__dict__['step']
    closure = dict(zip(outer.__code__.co_freevars,outer.__closure__ or ()))
    require('env' in closure and isinstance(closure['env'].cell_contents,dict), 'Original shared-step namespace')
    original_env = closure['env'].cell_contents
    require(isinstance(original_env.get('step'),types.FunctionType), 'Original compiled step callable')
    code,count = model.runtime().rewrite_clamps(cut,'<native-adjoint-exact-m9-prefix>')
    require(count == 1, 'Only prevariance clamp survives exact m9-only cutoff')
    env = dict(original_env); exec(code,env)
    return env['_nadj_m9'],dict(full_step_sha256=_digest(source),cut_step_sha256=_digest(cut),
        whole_source_inverse=True,all_prefix_statements_unchanged=True,depth=16,layer=9,
        removed_suffix_sha256=_digest(tail),physical_clamp_count=count,
        original_served_projection_after_return=True)


def work_bound(entry_stops, end_stops):
    """Stdlib-only worst-mask price before either local graph is constructed."""
    require(tuple(entry_stops) == ENTRY_STOPS and len(end_stops) == 9
            and tuple(end_stops[:8]) == ENTRY_STOPS and type(end_stops[-1]) is int
            and 0 < end_stops[-1]-end_stops[-2] <= 2048
            and (end_stops[-1]-end_stops[-2]) % 2 == 0, 'Fixed entry, actual fresh ninth cohort')
    n,r = 1024,384; layers = []
    for l,stops in ((8,entry_stops),(9,end_stops)):
        old = stops[l-6]; after = stops[l-5]; join = after-old; total = stops[-1]
        young = total-after
        shapes = [(join,n,r),(n,join,r),(r,n,r),(r,r,r),(n,r,r),
            (r,n,join),(r,join,r),(r,n,r),(r,r,old),(r,r,r),(r,r,r),
            (n,n,r),(n,r,after),(n,n,young),(n,after,r),(n,r,n),(n,young,n)]
        # Five covariance/four star n-cubes dominate actual compact/half blocks;
        # six messages and four256-wide dense bounds cover the unchanged GNN.
        shapes += [(n,n,n)]*9+[(n,n,16)]*6+[(n,256,256)]*4+[(n,64,4)]
        # 96-element overhead dominates native Y transforms/copies. Using this
        # on every Torch product as well is conservative. Two direct adjoints
        # have the same leading product count; transposition/copy traffic is
        # included. No matrix operation is hidden in the pointwise allowance.
        matrix = sum(3*(2*m*k*q+96*(m*k+k*q+m*q)) for m,k,q in shapes)
        qr = 32*n*r*r  # forward QR + its real-algebra reverse/triangular work
        # Frozen forward source bound is8192*n²+64*nR incl FP64 rate2, full bank
        # diagonal/omitted radial, birth/BD/P/NODE/guards. Factor8 covers primal
        # plus elementary reverse arithmetic/casts/reductions; no iterations.
        pointwise = 8*(8192*n*n+64*n*total)
        layers.append(dict(layer=l,full_columns=total,old_before=old,old_after=after,
            matrix_forward_and_VJP_upper=matrix,QR_forward_and_VJP_upper=qr,
            pointwise_forward_and_VJP_upper=pointwise,upper_flops=matrix+qr+pointwise))
    # Cold original arena/bindings3BF + fixed weights/features/P materialization
    # 1BF. 32 weighted traversals of two <=17k-column banks cover leaf/adjoint
    # copies, all-state gap reductions, dot products and complete finite scans.
    setup = 4_000_000_000
    ancillary = 32*n*(entry_stops[-1]+end_stops[-1])+256*n*n
    total = setup+ancillary+sum(v['upper_flops'] for v in layers)
    require(total < LOCAL_WORK_CAP, 'Local forward/VJP source envelope below1TF before graph creation')
    return dict(layers=layers,complete_setup_upper=setup,copy_gap_dot_guard_upper=ancillary,
        complete_local_work_upper=total,limit=LOCAL_WORK_CAP,actual_deployment_bill_claimed=False,
        native_N_plus_minus_calls_excluded_and_separately_required=True)


def _track(refs, *values):
    refs.extend(weakref.ref(v) for v in values if v is not None)


def _copy_state(torch, state, refs):
    require(type(state) is tuple and len(state) == 14, 'Decoded common Torch14 boundary')
    answer = list(state)
    for index in CONTINUOUS:
        value = state[index]
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
                and not value.requires_grad and bool(torch.isfinite(value).all()),
                'Fresh native boundary value, not a retained prior graph: '+STATE_NAMES[index])
        answer[index] = value.clone().requires_grad_(True)
        _track(refs,answer[index])
    require(not state[6].requires_grad and not state[13].requires_grad and type(state[12]) is tuple,
            'Masks/ordered IDs/cohort stops are explicit nondifferentiable branch metadata')
    return tuple(answer)


def _gap(torch, value, reference):
    require(value.shape == reference.shape, 'Comparable native/local coordinate shape')
    # Reporting is outside the differentiated scalar, never an intermediate
    # carrier substitution and never a fresh diagnostic graph after backward.
    with torch.no_grad():
        difference = value.to(torch.float64)-reference.to(torch.float64)
        require(bool(torch.isfinite(difference).all()), 'Finite reported local-output discrepancy')
        return dict(shape=list(value.shape),max_abs=float(difference.abs().max()),
            mean_squared=float((difference*difference).mean()),
            coordinate_comparison_not_gauge_invariant=True)


def _norm(torch, value):
    with torch.no_grad():
        wide = value.to(torch.float64)
        answer = float(torch.sqrt((wide*wide).sum()))
    require(math.isfinite(answer), 'Finite reported gradient norm')
    return answer


def evaluate(torch, model, card, handle, entry14, end14, *, cotangent,
             parameter_direction, native_m9, native_selectors, tick=None):
    """Return plain first-order diagnostics; never a fit or secant acceptance.

    parameter_direction: exact PKEYS mapping of same-shape CPU FP32 tensors.
    cotangent: caller-fixed CPU FP64 vector1024. native_m9: newly recorded N
    served FP32 vector1024. native_selectors maps the five CELLS to plain
    ordered_ids/membership lists from that same N record.
    """
    certificate = work_bound(entry14[12],end14[12])
    require(card.depth == 16 and card.n == 1024 and hasattr(card,'_native_rangefinder'), 'Original production card')
    require(cotangent.shape == native_m9.shape == (1024,) and cotangent.dtype == torch.float64
            and native_m9.dtype == torch.float32 and cotangent.device.type == native_m9.device.type == 'cpu'
            and not cotangent.requires_grad and not native_m9.requires_grad
            and bool(torch.isfinite(cotangent).all()) and bool(torch.isfinite(native_m9).all())
            and bool((native_m9 >= 0).all()), 'Fixed target-free cotangent and native served reference')
    require(set(native_selectors) == set(CELLS), 'Exactly five fresh native local selector cells')
    keys = tuple(model._JOINT.PKEYS); P = card.P0; params = tuple(P[k] for k in keys)
    require(set(parameter_direction) == set(keys), 'Complete caller-provided P direction, no selected subset')
    for key,value in zip(keys,params):
        direction = parameter_direction[key]
        require(value.is_leaf and value.requires_grad and value.grad is None and value.device.type == 'cpu'
                and value.dtype == torch.float32 and direction.shape == value.shape
                and direction.device.type == 'cpu' and direction.dtype == value.dtype
                and not direction.requires_grad and bool(torch.isfinite(direction).all()),
                'Actual supplied P leaf/direction: '+key)
    require(not card.A.requires_grad and not card.coefficients.requires_grad, 'Original NODE A/cstar remain frozen')
    m9_function,cut_proof = _m9_function(model,card,handle)
    source_select = card._select; had_select = '_select' in card.__dict__
    previous_select = card.__dict__.get('_select'); selection_records = {}

    def select(score):
        cell = str(card._layer)+':'+str(card._selection_slot)
        require(cell in native_selectors and cell not in selection_records, 'One actual local selection in fixed roster')
        ids,mask = source_select(score)
        row = native_selectors[cell]; order = row['ordered_ids']; members = row['membership']
        require(type(order) is list and all(type(v) is int and 0 <= v < 1024 for v in order)
                and sorted(order) == members and len(set(order)) == len(order), 'Native ordered IDs/membership metadata')
        recomputed = ids.detach().cpu().tolist()
        require(sorted(recomputed) == members, 'Recomputed Torch selection must preserve native membership: '+cell)
        exact = torch.tensor(order,dtype=torch.int64,device='cpu')
        native_mask = torch.zeros_like(score).scatter(0,exact,torch.ones(len(order),dtype=score.dtype,device='cpu'))
        selection_records[cell] = dict(native_ordered_ids=order,recomputed_ordered_ids=recomputed,
            membership_equal=True,order_equal=recomputed == order,used_fresh_native_order=True)
        return exact,native_mask

    def copied_gradients(loss, inputs, graph_refs):
        gradients = torch.autograd.grad(loss,inputs,allow_unused=True,create_graph=False,retain_graph=False)
        _track(graph_refs,*gradients)
        connected = [v is not None for v in gradients]; answer = []
        for value,source in zip(gradients,inputs):
            result = torch.zeros_like(source) if value is None else value.clone()
            require(not result.requires_grad and bool(torch.isfinite(result).all()), 'Finite explicit first-order cotangent')
            answer.append(result)
        return tuple(answer),connected

    def phase9(refs):
        state = _copy_state(torch,end14,refs); control = torch.zeros(1024,dtype=torch.float64,device='cpu')
        card._layer,card._selection_slot = 9,0
        mean = m9_function(card,9,state,control,P)
        served = model._JOINT._CO.served(mean)
        loss = (served.to(torch.float64)*cotangent).sum()
        _track(refs,control,mean,served,loss)
        derivatives,connected = copied_gradients(loss,tuple(state[i] for i in CONTINUOUS)+params,refs)
        require(bool(torch.isfinite(loss)), 'Finite local m9 scalar')
        details = dict(local_scalar=loss.item(),native_scalar=float((native_m9.to(torch.float64)*cotangent).sum()),
            served_m9_gap=_gap(torch,served,native_m9),state_connected=connected[:len(CONTINUOUS)],
            P_connected=dict(zip(keys,connected[len(CONTINUOUS):])),
            state_cotangent_norms={STATE_NAMES[i]:_norm(torch,g) for i,g in zip(CONTINUOUS,derivatives)})
        return derivatives[:len(CONTINUOUS)],derivatives[len(CONTINUOUS):],details

    def phase8(state_cotangents, refs):
        state = _copy_state(torch,entry14,refs); control = torch.zeros(1024,dtype=torch.float64,device='cpu')
        out = card.step(8,state,control,P)
        require(type(out) is tuple and len(out) == 14 and out[12] == end14[12]
                and torch.equal(out[6],end14[6]) and torch.equal(out[13],end14[13]),
                'Whole step8 returns identical native discrete boundary coordinates')
        require(all(out[i].shape == g.shape == end14[i].shape
                    for i,g in zip(CONTINUOUS,state_cotangents)),
                'All eleven coordinate/cotangent shapes agree before the scalar; no broadcast')
        loss = sum((out[i].to(torch.float64)*g.to(torch.float64)).sum()
                   for i,g in zip(CONTINUOUS,state_cotangents))
        require(bool(torch.isfinite(loss)), 'Finite complete eleven-coordinate scalar')
        _track(refs,control,loss,*(out[i] for i in CONTINUOUS))
        derivatives,connected = copied_gradients(loss,params,refs)
        details = dict(P_connected=dict(zip(keys,connected)),
            continuous_output_gaps={STATE_NAMES[i]:_gap(torch,out[i],end14[i]) for i in CONTINUOUS},
            all_eleven_output_cotangents_used=True,native_end8_not_replaced_by_local_output=True)
        return derivatives,details

    card._select = select
    try:
        require(not card.selection_cells and not card.shared_cells and not card.variance_domain,
                'Fresh diagnostics card, no stale local branch records')
        refs9 = []; cot_state,P9,details9 = phase9(refs9)
        gc.collect(); require(all(r() is None for r in refs9), 'Step9 graph/input owners retired before step8')
        if tick is not None: tick()
        refs8 = []; P8,details8 = phase8(cot_state,refs8)
        gc.collect(); require(all(r() is None for r in refs8), 'Step8 graph/input owners retired')
        if tick is not None: tick()
        require(set(selection_records) == set(CELLS) and all(v.grad is None for v in params),
                'Complete five-cell reverse order with no P.grad accumulation')
        direct = sum(float((g.to(torch.float64)*parameter_direction[k].to(torch.float64)).sum())
                     for k,g in zip(keys,P9))
        earlier = sum(float((g.to(torch.float64)*parameter_direction[k].to(torch.float64)).sum())
                      for k,g in zip(keys,P8))
        require(math.isfinite(direct) and math.isfinite(earlier)
                and math.isfinite(direct+earlier), 'Finite direct/carried/total directionals')
        denominator = abs(earlier)+abs(direct)
        result = dict(completed=True,phase_order=[9,8],direct_P9_directional=direct,
            causal_P8_directional=earlier,total_P_directional=direct+earlier,
            directional_P9=direct,directional_P8=earlier,directional_total=direct+earlier,
            carried_fraction=abs(earlier)/denominator if denominator else 0.0,
            carried_fraction_denominator=denominator,directional_decomposition_defined=denominator > 0,
            state_cotangent_norms=details9['state_cotangent_norms'],
            P9_norms={k:_norm(torch,g) for k,g in zip(keys,P9)},
            P8_norms={k:_norm(torch,g) for k,g in zip(keys,P8)},
            phase9=details9,phase8=details8,selectors=selection_records,cutoff_source_proof=cut_proof,
            work_certificate=certificate,all_local_graph_owners_retired=True,
            declared_local_VJP_not_native_rounding_derivative=True,full_forward_fidelity_qualified=False,
            parameter_updates=0,truth_accessed=False,fit_authorized=False,secant_qualified=False)
        owned_refs = [weakref.ref(g) for g in (*cot_state,*P9,*P8)]
        del cot_state,P9,P8
        gc.collect(); require(all(r() is None for r in owned_refs), 'All retained first-order adjoint arrays retired')
        return result
    finally:
        if had_select:
            card._select = previous_select
        else:
            del card._select
