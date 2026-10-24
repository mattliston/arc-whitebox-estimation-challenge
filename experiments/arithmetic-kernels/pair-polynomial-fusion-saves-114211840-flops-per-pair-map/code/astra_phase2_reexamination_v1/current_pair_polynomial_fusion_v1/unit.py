"""DRAFT inert fixed pre8 pair/terminal integrity and billing unit.

Only evaluate() does numerical work, in its caller's fresh paid context.
It calls four extracted functions, never an Estimator, prediction, head,
initializer, arena or training routine. Returns JSON-safe scalar metadata;
snapshot bytes are hashed and released, and no numerical array escapes.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref


HERE = Path(__file__).resolve().parent
PREPARE_OWNER = HERE.parent / 'current_native_trajectory_adjoint_v1/native_cell.py'
PREPARE_SHA = '814d8bf5f4161ef388ef8c811421fc8c90ca8929ab440dbba4d4de3dffc99e9c'
PORT_SHA = 'b064846625289cbd75352201f31e90868a9084609d650bc651797660b70f58e9'
PINS = {PREPARE_OWNER: PREPARE_SHA, HERE / 'port.py': PORT_SHA}
FLOP_CAP = 10_000_000_000
FUNCTION_FLOP_UPPER = 1_000_000_000
DIAGNOSTIC_FLOP_UPPER = 2_000_000_000
WIDTH = 1024
INPUT_NAMES = ('mu', 'Cpre', 'K21', 'k4', 'k22')
ROSTER = ('original_pair', 'fused_pair', 'original_terminal', 'fused_terminal')
MAX_ABS = 1e-6
RMS = 1e-7
RADIAL_ABS = 1e-7
PAIR_SAVING = 114_211_840
TERMINAL_SAVING = 40_866_816


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def cost_certificate():
    """Prospective conservative source envelope, never a claimed actual bill.

    The unfused FP32 pair loop is 95*c*n^2+14*n^2+8*c*n F,
    at most313,548,800F here. All pre/post FP64 matrix construction,
    casts and centralization are below128*n^2 F; the deliberately loose
    65,536*n+1,000,000 vector/scalar allowance covers Gaussian/CDF,
    derivatives, marginal moments, diagonals, radial and auxiliary work.
    Their sum is below1BF per function. Fusion reduces this tariff.
    Input/snapshot/finite/symmetry/error work has a separate2BF allowance,
    substantially above its fixed <128 full1024-square FP64-equivalent
    passes. Both bounds are also enforced against actual section deltas.
    """
    n = WIDTH
    loop = 95*3*n*n + 14*n*n + 8*3*n
    function_bound = loop + 128*n*n + 65_536*n + 1_000_000
    require(function_bound < FUNCTION_FLOP_UPPER, 'Conservative complete function envelope')
    return dict(schema='current_pair_polynomial_fusion_unit_price_v1',
        width=n, function_calls=4, input_arrays=5, input_dtype='float64',
        original_loop_upper=loop, complete_function_source_upper=function_bound,
        per_function_enforced_upper=FUNCTION_FLOP_UPPER,
        diagnostic_enforced_upper=DIAGNOSTIC_FLOP_UPPER,
        aggregate_enforced_section_upper=4*FUNCTION_FLOP_UPPER+DIAGNOSTIC_FLOP_UPPER,
        complete_context_hard_cap=FLOP_CAP, extra_candidate_initialization=0,
        original_pair_minus_fused_pair=PAIR_SAVING,
        original_terminal_minus_fused_terminal=TERMINAL_SAVING,
        all_actual_section_deltas_required=True, measured_bill=False,
        dense_reference_or_full_model_calls=0, no_unmetered_numerical_comparison=True)


def _track(refs, value):
    if isinstance(value, dict):
        for item in value.values():
            _track(refs, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _track(refs, item)
    elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
        try:
            refs.append(weakref.ref(value))
        except TypeError:
            require(tuple(value.shape) == () and int(value.nbytes) <= 8,
                'Only immediate numerical scalars may lack a weak reference')


def evaluate(context, port_api, np, fnp, ctx, refs, tick):
    """Fresh caller-owned native context; only bytes/scalar metadata can leave."""
    import flopscope as flops
    require('torch' not in sys.modules, 'Native-only unit, no Torch or model bridge')
    require(context['source_bindings'][str(PREPARE_OWNER)] == PREPARE_SHA
        and digest((HERE/'port.py').read_bytes()) == PORT_SHA,
        'Actual qualified packet admission and exact fusion owner')
    manifest = context['manifest']
    require(manifest['rank'] == 35 and manifest['layer'] == 8 and manifest['consumer'] == 9
        and manifest['truth_accessed'] is False and manifest['saved_after_context_release'] is True
        and manifest['candidate_sha256'] == context['source_proof']['candidate_sha256'],
        'Qualified target-free actual pre8 physical capture')
    packet_path = Path(manifest['packet_file'])
    require(context['input_bindings'][str(packet_path)] == manifest['packet_sha256']
        and packet_path.is_file() and not packet_path.is_symlink()
        and packet_path.stat().st_size == manifest['bytes'] <= 512*1024**2,
        'Bounded admitted immutable packet')
    before_unit = int(ctx.flops_used)
    audit = dict(schema='current_pair_polynomial_fusion_unit_v1', passed=False,
        function_calls=0, full_model_calls=0, truth_accessed=False, parameter_changes=0,
        source_proof=None, sections=[], products={}, outputs={}, checks={},
        native_timing_descriptive_only=True, RPC_runtime_qualified=False,
        full_card_qualified=False, release_ready=False, score_measured=False,
        packet_sha256=manifest['packet_sha256'], metric='covariance',
        input_arrays=INPUT_NAMES, offdiagonal_rms_denominator=WIDTH*(WIDTH-1))
    inputs = {}; live = {}; environments = []; raw = None

    @contextlib.contextmanager
    def section(name, kind='diagnostic'):
        tick()
        begin = int(ctx.flops_used); started = time.monotonic()
        failed = True
        try:
            yield
            failed = False
        finally:
            used = int(ctx.flops_used)-begin
            audit['sections'].append(dict(name=name, kind=kind, flops=used,
                wall_s=time.monotonic()-started, failed=failed))
        require(int(ctx.flops_used) <= FLOP_CAP, 'Actual complete10BF cap')
        tick()

    def snapshot(value, label):
        require(hasattr(value, 'shape') and hasattr(value, 'dtype'), 'Typed array/scalar: '+label)
        copy = finite = reduced = None
        try:
            copy = fnp.copy(value); _track(refs, copy)
            require(str(copy.dtype) == 'float64' and tuple(copy.shape) in
                ((), (WIDTH,), (WIDTH, WIDTH)), 'Original full physical FP64 output: '+label)
            finite = fnp.isfinite(copy); reduced = fnp.all(finite)
            _track(refs, (finite, reduced))
            require(bool(reduced), 'Finite copied physical value: '+label)
            data = copy.tobytes()
            require(len(data) == int(copy.nbytes), 'Complete detached snapshot bytes: '+label)
            return dict(shape=list(copy.shape), dtype=str(copy.dtype), bytes=len(data), sha256=digest(data))
        finally:
            copy = finite = reduced = None

    def matches(row, expected):
        if 'alias' in expected:
            require(expected == {'alias':'pre8.Cpre'}, 'Only original explicit covariance alias')
            expected = manifest['packets']['pre8']['Cpre']
        expected = expected['array']
        return all(row[k] == expected[k] for k in ('shape', 'dtype', 'bytes', 'sha256'))

    def take_snapshot(output, role):
        require(type(output) is tuple and len(output) == 5, 'Exact original five-return interface')
        m, covariance, target, radial, aux = output
        _track(refs, output)
        record = dict(m=snapshot(m,role+'.m'), C=snapshot(covariance,role+'.C'), aux={})
        record['C_diagonal'] = snapshot(fnp.diag(covariance),role+'.C_diagonal')
        if target is not None:
            record['target'] = snapshot(target,role+'.target')
            record['target_diagonal'] = snapshot(fnp.diag(target),role+'.target_diagonal')
        else:
            require(role.endswith('terminal'), 'Only original terminal has no target output')
            record['target'] = record['target_diagonal'] = None
        if hasattr(radial, 'dtype'):
            record['radial_new'] = snapshot(radial,role+'.radial')
        else:
            require(role.endswith('terminal') and type(radial) is float and radial == 0.0,
                'Unchanged terminal scalar zero, not an alternative metric')
            record['radial_new'] = dict(scalar=0.0)
        expected_aux = manifest['packets']['pair8']['aux']['mapping']
        require(type(aux) is dict and set(aux) == set(expected_aux), 'Every original auxiliary member')
        for key, value in aux.items():
            if key == 'pre_cov':
                require(value is inputs['Cpre'], 'Original borrowed input covariance alias')
                record['aux'][key] = dict(audit['input_snapshots']['Cpre'])
            else:
                record['aux'][key] = snapshot(value,role+'.aux.'+key)
        return record

    def errors(left, right, label):
        difference = absolute = square = maximum = total = rms = None
        try:
            difference = left-right; absolute = fnp.abs(difference)
            maximum = fnp.max(absolute); square = difference*difference
            total = fnp.sum(square)
            rms = fnp.sqrt(total/(WIDTH*(WIDTH-1)))
            _track(refs,(difference,absolute,maximum,square,total,rms))
            max_abs, rms_value = float(maximum), float(rms)
            return dict(label=label, max_abs=max_abs, rms=rms_value,
                rms_denominator=WIDTH*(WIDTH-1),
                diagonal_checked_separately=True,
                passed=max_abs<=MAX_ABS and rms_value<=RMS)
        finally:
            difference = absolute = square = maximum = total = rms = None

    try:
        original, fused, proof = port_api.source_bundle(tick)
        audit['source_proof'] = proof
        require(proof['full_source_inverse_verified'] is True
            and proof['conditional_saved_flops'] == 1_639_832_576,
            'Complete selected two-function source inverse and tariff')
        # Only four complete numerical function definitions, with their exact
        # original public numerical dependencies. No giant literal or class runs.
        dependencies = '\n\n'.join(port_api._function_span(original,name)[2]
            for name in ('gaussian_raw','derivatives'))
        for text in (original, fused):
            require(all(port_api._function_span(text,name)[2] == port_api._function_span(original,name)[2]
                for name in ('gaussian_raw','derivatives')), 'Byte-exact common Gaussian dependencies')
            source = dependencies+'\n\n'+'\n\n'.join(port_api._function_span(text,name)[2]
                for name in port_api.FUNCTIONS)
            env = {'fnp':fnp,'flops':flops}
            exec(compile(source,'<fixed-paid-fusion-functions>','exec'),env)
            environments.append(env)
        del original, fused, source, dependencies, env
        with section('fixture_ingress_and_symmetry'):
            raw = packet_path.read_bytes()
            require(len(raw) == manifest['bytes'] and digest(raw) == manifest['packet_sha256'],
                'Actual immutable full packet before selected input decoding')
            input_snapshots = {}
            for name in INPUT_NAMES:
                row = manifest['packets']['pre8'][name]['array']
                shape = (WIDTH,) if name in ('mu','k4') else (WIDTH,WIDTH)
                require(row['dtype']=='float64' and tuple(row['shape'])==shape
                    and row['byteorder']==sys.byteorder and row['original_c_contiguous'] is True
                    and row['offset']>=0 and row['bytes']>0
                    and row['offset']+row['bytes']<=len(raw), 'Actual complete pre8 input layout: '+name)
                data = raw[row['offset']:row['offset']+row['bytes']]
                require(digest(data)==row['sha256'], 'Exact admitted pre8 member: '+name)
                view = np.frombuffer(data,dtype=np.float64).reshape(shape)
                _track(refs,view)
                value = fnp.array(view,dtype=fnp.float64,copy=True); _track(refs,value)
                inputs[name] = value
                input_snapshots[name] = snapshot(value,'input.'+name)
                require(matches(input_snapshots[name],{'array':row}), 'Lossless paid original input copy')
                del view, value, data
            audit['input_snapshots'] = input_snapshots
            raw = None
            variance = fnp.diag(inputs['Cpre'])
            positive = fnp.all(variance>1e-10); _track(refs,(variance,positive))
            require(bool(positive), 'Original strict positive physical variance, no repair')
            difference = inputs['Cpre']-inputs['Cpre'].T
            maximum = fnp.max(fnp.abs(difference))
            square = difference*difference
            rms = fnp.sqrt(fnp.sum(square)/(WIDTH*WIDTH))
            _track(refs,(difference,maximum,square,rms))
            audit['captured_covariance_asymmetry'] = dict(max_abs=float(maximum),rms=float(rms),repaired=False)
            del variance,positive,difference,maximum,square,rms
        for index,role in enumerate(ROSTER):
            env = environments[index%2]
            name = port_api.FUNCTIONS[0 if index<2 else 1]
            with section(role,'function'):
                started = time.monotonic()
                output = env[name](inputs['mu'],inputs['Cpre'],inputs['K21'],inputs['k4'],
                    inputs['k22'],metric='covariance')
                elapsed = time.monotonic()-started
                live[role] = output
                audit['function_calls'] += 1
            used = audit['sections'][-1]['flops']
            audit['products'][role] = dict(flops=used,native_function_wall_s=elapsed,
                timing_excludes_separate_input_snapshot_and_diagnostics=True)
            require(0<used<=FUNCTION_FLOP_UPPER, 'Actual complete function section envelope: '+role)
            with section(role+'_snapshots'):
                audit['outputs'][role] = take_snapshot(output,role)
            del output
            if role == 'original_pair':
                expected = manifest['packets']['pair8']; actual = audit['outputs'][role]
                checks = {key:matches(actual[key],expected[key]) for key in ('m','C','target','radial_new')}
                checks.update({'aux.'+key:matches(row,expected['aux']['mapping'][key])
                    for key,row in actual['aux'].items()})
                audit['original_pair_capture_replay'] = checks
                require(all(checks.values()), 'Actual original pair byte replay BEFORE fused work: '+
                    json.dumps(checks,sort_keys=True))
        with section('comparisons_and_input_immutability'):
            reference = audit['outputs']['original_pair']
            for role, output in audit['outputs'].items():
                audit['checks'][role+'.mean_exact'] = output['m']==reference['m']
                audit['checks'][role+'.covariance_diagonal_exact'] = output['C_diagonal']==reference['C_diagonal']
                audit['checks'][role+'.all_auxiliary_exact'] = output['aux']==reference['aux']
                if output['target_diagonal'] is not None:
                    audit['checks'][role+'.target_diagonal_exact'] = output['target_diagonal']==reference['target_diagonal']
            audit['differences'] = {
                'pair_covariance':errors(live['original_pair'][1],live['fused_pair'][1],'pair covariance'),
                'pair_target':errors(live['original_pair'][2],live['fused_pair'][2],'pair target'),
                'terminal_covariance':errors(live['original_terminal'][1],live['fused_terminal'][1],'terminal covariance')}
            radial_error = fnp.abs(live['original_pair'][3]-live['fused_pair'][3]); _track(refs,radial_error)
            audit['radial_abs_error'] = float(radial_error)
            audit['checks']['radial'] = audit['radial_abs_error']<=RADIAL_ABS
            audit['checks']['terminal_radial_zero'] = all(audit['outputs'][role]['radial_new']=={'scalar':0.0}
                for role in ('original_terminal','fused_terminal'))
            del radial_error
            for name,value in inputs.items():
                after = snapshot(value,'input_after.'+name)
                audit['checks']['input_unchanged.'+name] = after==audit['input_snapshots'][name]
            del value
        products = audit['products']
        audit['actual_function_savings'] = dict(
            pair=products['original_pair']['flops']-products['fused_pair']['flops'],
            terminal=products['original_terminal']['flops']-products['fused_terminal']['flops'])
        audit['checks']['pair_tariff_exact'] = audit['actual_function_savings']['pair']==PAIR_SAVING
        audit['checks']['terminal_tariff_exact'] = audit['actual_function_savings']['terminal']==TERMINAL_SAVING
        function_flops = sum(row['flops'] for row in audit['sections'] if row['kind']=='function')
        diagnostic_flops = sum(row['flops'] for row in audit['sections'] if row['kind']=='diagnostic')
        actual = int(ctx.flops_used)-before_unit
        require(function_flops+diagnostic_flops==actual and diagnostic_flops<=DIAGNOSTIC_FLOP_UPPER
            and 0<actual<=FLOP_CAP, 'Every actual numerical operation has one complete paid section')
        audit.update(function_flops=function_flops,diagnostic_flops=diagnostic_flops,
            complete_evaluation_flops=actual,section_reconciliation_exact=True,
            passed=all(audit['checks'].values()) and all(row['passed'] for row in audit['differences'].values()),
            local_array_owners_cleared_before_return=True,
            caller_must_verify_closed_context_and_all_weakrefs=True)
        # Serialization validates that the result contains no arrays, NaN or Infinity.
        json.dumps(audit,allow_nan=False)
        return audit
    finally:
        raw = None
        inputs.clear(); live.clear()
        for env in environments:
            env.clear()
        environments.clear()
