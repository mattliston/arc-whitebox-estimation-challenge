"""DRAFT inert, two-function polynomial fusion against the retained W5 source.

Returns source TEXT only. No candidate import, execution, materialization,
parameter change, target access, or numerical qualification is performed.
The identity is real arithmetic on a symmetric physical covariance. Native
FP32 accumulation, separately rounded T31, and any covariance asymmetry make
this a drift-qualified proposal, never an inherited byte-equivalent repair.
"""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
W5 = NEW / 'current_comp_weight_transform_reuse_v1'
PARENT = W5 / 'rpc37_v1/estimator.py'
PARENT_SHA = 'c0d4d37f75d534168b42551f7f9190c46727d92f70e8cf65668caf948d6ef9fb'
NOTE = NEW / 'PAIR_POLYNOMIAL_FUSION_SOURCE_PRICE_20260913_UTC.md'
RESULT = W5 / 'rpc37_v1/result.json'
COMPLETION = W5 / 'rpc37_v1/completion.json'
FINALIZATION = W5 / 'rpc37_v1/finalization.json'
FUNCTIONS = ('_nonlinear_pair_batch', '_nonlinear_terminal_live')
PINS = {
    PARENT: PARENT_SHA,
    NOTE: 'd83787d190f94b187460bdc6008cc57f14eb69282945c93e4d7d6d0faa1b5517',
    W5 / 'FULL_RPC_PROTOCOL.md': '4e02e2403a55aee3ab4301706194f395ddaf99f2167f08754479c118897087d1',
    RESULT: '8502585779442117582852162dc6c2d9526710d5853381dccd9a591696b421a6',
    COMPLETION: 'fa276f746c6096b3de452cee580812abfc4c1b7f5dcfdd1589ccbc6fca972c9c',
    FINALIZATION: 'b76cc3ebad637b06b75027f5ebfdc3f78aba831238b11382b23135f941c7e1d2',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read(path, expected, tick=None):
    if tick:
        tick()
    require(path.is_file() and not path.is_symlink()
        and path.stat().st_size <= 4 * 1024**2, 'Bounded regular source/receipt owner')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Exact owner: ' + str(path))
    if tick:
        tick()
    return raw.decode('utf-8')


def _admission(tick=None):
    texts = {path: _read(path, expected, tick) for path, expected in PINS.items()}
    r, c, f = (json.loads(texts[path]) for path in (RESULT, COMPLETION, FINALIZATION))
    require(all(x['passed'] is True and x['completed'] is True for x in (r, c, f))
        and c['result_sha256'] == f['result_sha256'] == PINS[RESULT]
        and f['completion_sha256'] == PINS[COMPLETION], 'Direct retained W5 three-seal chain')
    require(r['candidate_sha256'] == PARENT_SHA
        and r['artifacts'][str(PARENT)] == PARENT_SHA
        and r['component_retained'] is True
        and r['whole_owned_tree_retired'] is True
        and r['observational_groups_retired'] is True
        and r['release_ready'] is False
        and r['truth_accessed'] is False, 'Actual W5 component, never release qualification')
    return texts


def authenticate(tick=None):
    """Direct bounded source/scalar receipt checks only, no ancestral replay."""
    _admission(tick)
    return {str(path): expected for path, expected in PINS.items()}


def _function_span(source, name):
    # Line offsets avoid ast.get_source_segment on the giant learned literal.
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith('def ' + name + '(')]
    require(len(starts) == 1, 'One effective complete function: ' + name)
    start = starts[0]
    stop = start + 1
    while stop < len(lines) and (not lines[stop].strip() or lines[stop][0].isspace()):
        stop += 1
    value = ''.join(lines[start:stop]).rstrip('\n')
    tree = ast.parse(value)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        and tree.body[0].name == name, 'Complete selected function, not a prefix')
    offset = sum(map(len, lines[:start]))
    return offset, offset + len(value), value


def _replace_once(text, before, after):
    require(text.count(before) == 1, 'One exact source splice: ' + before[:80])
    return text.replace(before, after, 1)


OLD_T31 = '    T31 = t4[:, None] * rho if k31 else None\n'
OLD_CAST = '    T31=T31.astype(fnp.float32)\n'
OLD_LOOP = '''    for r in range(7):
        if r:
            ans=ans+rr[None,:,:]*(F[:,r,:,None]*G[:,r,None,:])
        if r<=2:
            ec=(T21[None,:,:]*(F[:,r+2,:,None]*G[:,r+1,None,:])+T21.T[None,:,:]*(F[:,r+1,:,None]*G[:,r+2,None,:]))/2
            ec=ec+T22[None,:,:]*(F[:,r+2,:,None]*G[:,r+2,None,:])/4
            ec=ec+(T31[None,:,:]*(F[:,r+3,:,None]*G[:,r+1,None,:])+T31.T[None,:,:]*(F[:,r+1,:,None]*G[:,r+3,None,:]))/6
            if r:
                ec=ec+((t3[None,:]*F[:,r+3,:])[:,:,None]*G[:,r,None,:]+F[:,r,:,None]*(t3[None,:]*G[:,r+3,:])[:,None,:])/6
                ec=ec+((t4[None,:]*F[:,r+4,:])[:,:,None]*G[:,r,None,:]+F[:,r,:,None]*(t4[None,:]*G[:,r+4,:])[:,None,:])/24
            ans=ans+rr[None,:,:]*ec
        rr=rr*rho/(r+1)
'''
NEW_LOOP = '''    for r in range(7):
        if 1<=r<=3:
            if r<=2:
                df=t3[None,:]*F[:,r+3,:]/6+t4[None,:]*(4*r*F[:,r+2,:]+F[:,r+4,:])/24
                dg=t3[None,:]*G[:,r+3,:]/6+t4[None,:]*(4*r*G[:,r+2,:]+G[:,r+4,:])/24
            else:
                df=t4[None,:]*F[:,5,:]/2
                dg=t4[None,:]*G[:,5,:]/2
            ans=ans+rr[None,:,:]*((F[:,r,:]+df)[:,:,None]*G[:,r,None,:]+F[:,r,:,None]*dg[:,None,:])
            del df,dg
        elif r:
            ans=ans+rr[None,:,:]*(F[:,r,:,None]*G[:,r,None,:])
        if r<=2:
            ec=(T21[None,:,:]*(F[:,r+2,:,None]*G[:,r+1,None,:])+T21.T[None,:,:]*(F[:,r+1,:,None]*G[:,r+2,None,:]))/2
            ec=ec+T22[None,:,:]*(F[:,r+2,:,None]*G[:,r+2,None,:])/4
            ans=ans+rr[None,:,:]*ec
        rr=rr*rho/(r+1)
'''


def price(n=1024, channels=3):
    """Exact marginal plain-array tariff; unaffected work remains payable.

    No total bill is fabricated. These are operation-count/shape formulas,
    conditional on actual original dtypes and full-width channel geometry.
    The external qualifier must reconcile complete measured original/new F.
    """
    require(type(n) is int and n > 0 and channels in (1, 3), 'Declared physical channel geometry')
    e, u = channels * n * n, channels * n
    before = 59 * e + 8 * u + 4 * n * n
    after = 24 * e + 35 * u
    return dict(n=n, channels=channels,
        original_affected_flops=before, fused_affected_flops=after,
        saved_flops=before-after,
        removed_dense_fp32_multiply_elements=17*e,
        removed_dense_fp32_add_elements=11*e,
        removed_dense_fp32_divide_elements=7*e,
        additional_fp32_vector_flops=27*u,
        removed_fp64_T31_multiply_flops=2*n*n,
        removed_T31_cast_flops=2*n*n,
        removed_dense_arithmetic_calls=35,
        additional_vector_arithmetic_calls=27,
        removed_T31_construction_cast_calls=2,
        net_arithmetic_cast_calls_saved=10,
        extra_coefficient_live_bytes_upper=6*u*4,
        removed_live_fp32_T31_bytes=4*n*n,
        additional_initialization_flops=0, additional_persistent_bytes=0,
        complete_bill_measured=False, plain_array_marginal_only=True)


def source_bundle(tick=None):
    """Return (original TEXT, fused TEXT, proof), without compiling/executing."""
    parent = _admission(tick)[PARENT]
    edits = []
    for name in FUNCTIONS:
        start, stop, original = _function_span(parent, name)
        require('    gauss_order=6;edge_order=2;local=True;k31=True\n' in original,
            'Exact frozen physical polynomial orders/flags')
        changed = _replace_once(original, OLD_T31, '')
        changed = _replace_once(changed, OLD_CAST, '')
        changed = _replace_once(changed, OLD_LOOP, NEW_LOOP)
        # Each individual small-function inverse and its parse are checked.
        restored = _replace_once(changed, NEW_LOOP, OLD_LOOP)
        # Restore deleted lines at their unchanged, unique following anchors.
        restored = _replace_once(restored,
            '    maxorder = max(gauss_order, edge_order + 4)\n',
            OLD_T31+'    maxorder = max(gauss_order, edge_order + 4)\n')
        restored = _replace_once(restored, '    t3=t3.astype(fnp.float32)\n',
            OLD_CAST+'    t3=t3.astype(fnp.float32)\n')
        require(restored == original and 'T31' not in changed,
            'Complete individual fusion inverse, no remaining T31 owner')
        ast.parse(changed)
        edits.append((start, stop, original, changed, name))
    fused, applied = parent, []
    for start, stop, original, changed, name in sorted(edits, reverse=True):
        require(fused[start:stop] == original, 'Exact disjoint whole-function position')
        fused = fused[:start]+changed+fused[stop:]
        applied.append((start, original, changed, name))
    inverse = fused
    for start, original, changed, name in reversed(applied):
        require(inverse[start:start+len(changed)] == changed, 'Exact positional inverse: '+name)
        inverse = inverse[:start]+original+inverse[start+len(changed):]
    require(inverse == parent, 'Entire pinned W5 source restored byte-for-byte')
    pair, terminal = price(1024, 3), price(1024, 1)
    saving = 14*pair['saved_flops']+terminal['saved_flops']
    require(saving == 1_639_832_576, 'Exact selected whole-call marginal source price')
    own = Path(__file__).resolve()
    proof = dict(schema='current_pair_polynomial_fusion_source_v1',
        parent_sha256=PARENT_SHA, candidate_sha256=sha(fused), source_bytes=len(fused.encode()),
        source_bindings={**{str(path): expected for path, expected in PINS.items()},
            str(own):hashlib.sha256(own.read_bytes()).hexdigest()},
        functions=list(FUNCTIONS), whole_function_splices=2,
        individual_function_inverses_verified=True, full_source_inverse_verified=True,
        only_T31_build_cast_and_polynomial_loop_changed=True,
        symmetric_physical_covariance_real_identity_required=True,
        no_symmetrization_or_input_repair=True, no_DF_DG_quadratic_term=True,
        separate_original_T31_rounding_removed=True,
        original_all_orders_recurrence_and_T21_T22_retained=True,
        original_final_unused_rho_update_retained=True,
        original_centralization_diagonals_radial_and_aux_retained=True,
        original_heads_parameters_guards_selectors_arena_and_W5_retained=True,
        original_function_bindings_and_call_sites_retained=True,
        additional_initialization_flops=0, additional_persistent_bytes=0,
        pair_price=pair, terminal_price=terminal, conditional_pair_calls=14,
        conditional_terminal_calls=1, conditional_saved_flops=saving,
        conditional_full_arithmetic_cast_calls_saved=150,
        conditional_rank37_cold_flops=410_097_374_959-saving,
        conditional_rank37_warm_flops=403_321_418_965-saving,
        conditional_rank37_initialization_flops=6_775_955_994,
        actual_block_drift_bill_and_runtime_qualified=False,
        FP32_byte_equivalence_claimed=False, full_card_qualified=False,
        runtime_gain_measured=False, raw_value_measured=False,
        full_card_or_submission_authorized=False,
        module_scope_numerical_work=False, model_calls=0, parameter_changes=0,
        truth_accessed=False, materialized_candidate=False)
    return parent, fused, proof


def derive_source(tick=None):
    """Builder-compatible TEXT/proof API; this does not write a candidate."""
    _, fused, proof = source_bundle(tick)
    return fused, proof


def pair_functions(tick=None):
    """Original/fused complete small-function sources for an owned unit.

    Returns (originals, fused, proof), where each source dictionary is keyed
    by the two unchanged original function names. Caller owns any later
    dependency extraction, runtime installation, fresh contexts and bills.
    """
    original, changed, proof = source_bundle(tick)
    return ({name:_function_span(original,name)[2] for name in FUNCTIONS},
        {name:_function_span(changed,name)[2] for name in FUNCTIONS}, proof)
