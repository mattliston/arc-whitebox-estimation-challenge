"""Inert, fixed .01 -> .1 response-only F32 read pricing; stdlib only.

No files are read at import. No estimator/module, array, stage, target, or CLI is
executed here. price_record(record) accepts ONE authenticated capture call:

  {source_sha256, rank:35, call:0|1, includes_initialization:bool,
   actual_flops:int, square_depths_bound_before:[...],
   square_depths_bound_after:[...], layers:[
     {layer:0..15, q0:int, q1:int, a:int, R:int,
      read_actual_flops:int, read_square_depths_bound_before:[...],
      read_square_depths_bound_after:[...], read_tiles:[
        {shape:[m,k,n], depth:0|3|4|5|6, dtype:'float32', flops:int}]}]}

read_actual_flops is the OUTER read _mm bill delta, after argument evaluation.
Each tile is one leaf product (including direct depth0 leaves); its shape is
the actual possibly padded product shape, and its bill excludes lazy bindings.
Layer0 has R=0, no _mm call, zero read bill and an empty tile list.

The caller owns source/artifact/byte, threshold-count and resource admission.
This module validates geometry and reconciles every old read _mm bill exactly.
It preserves the entire measured nonresponse remainder, including every old
setup FLOP, all-n diagonal work and full-width omitted_radial_source. It adds
any newly needed square binding not already paid in that conservative model.
Thus the result is a frozen-trajectory upper bound, NOT a new candidate bill,
runtime estimate, raw-error forecast, numerical qualification or permission.

price_pair([cold, warm]) additionally owns the fixed repeat's binding history.
Standalone warm price_record deliberately charges a conservative extra binding
again if it cannot prove the hypothetical cold call already paid that depth.
"""

from fractions import Fraction
import ast
import hashlib
import json
from pathlib import Path


SCHEMA = 'current_d50_response10_frozen_price_v1'
SOURCE_SHA256 = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
INITIALIZATION_FLOPS = 1383913786
DEPTHS = (3, 4, 5, 6)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label + ': integer domain')
    return value


def _depths(value, label):
    _require(isinstance(value, list), label + ': list required')
    _require(all(type(d) is int and d in DEPTHS for d in value), label + ': depth')
    _require(len(set(value)) == len(value), label + ': duplicate depth')
    return set(value)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _model():
    """Read literals from the final authenticated source; never exec/import it."""
    new = Path(__file__).resolve().parent.parent
    bank = new / 'birth_direction_fit_v1/build_terminal_v1/estimator.py'
    package = new.parent / 'data_gate/.venv/lib/python3.11/site-packages/flopscope'
    pins = {
        bank: SOURCE_SHA256,
        package / '_ndarray.py': 'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
        package / '_array_ops.py': 'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
        package / '_dtype_billing.py': 'd6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
        package / 'data/default_weights.json': 'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
        new / 'current_d50_d6_cost_census_v1/run.py': '7e278ed04becda90602773ab607538b0a2330b1e597766a52a391cc6794704cc',
    }
    for path, digest in pins.items():
        _require(_sha(path) == digest, 'Frozen pricing owner changed: ' + str(path))
    source = bank.read_bytes()
    _require(hashlib.sha256(source).hexdigest() == SOURCE_SHA256, 'Source read race')
    nodes = ast.parse(source).body
    names = {'_PACKED_ELEMENTS', '_PACKED_TILE', '_SQUARE_COSTS',
             '_SQUARE_EQUAL_DEPTHS', '_SQUARE_CROSS_DEPTHS', '_PACKED_SQUARE',
             '_PACKED_SQUARE4', '_PACKED_SQUARE5', '_PACKED_SQUARE6'}
    constants = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    constants[target.id] = ast.literal_eval(node.value)
    _require(set(constants) == names, 'Required final source constants')
    _require(constants['_PACKED_TILE'] == 8192 and
             constants['_PACKED_ELEMENTS'] == 292225024, 'Current D4 arena owner')
    table = constants['_SQUARE_EQUAL_DEPTHS']
    cross = constants['_SQUARE_CROSS_DEPTHS']
    _require(len(table) == 32 and all(len(row) == 32 for row in table) and
             len(cross) == 32, 'Current square lookup shape')
    _require(all(d in (0,) + DEPTHS for row in table for d in row) and
             all(d in (0,) + DEPTHS for d in cross), 'Current square lookup depths')
    weights = json.loads((package / 'data/default_weights.json').read_text())
    for name in ('add', 'subtract', 'multiply', 'matmul', 'getitem', 'astype',
                 'put', 'copyto', 'stack', 'reshape'):
        _require(weights['weights'][name] == 1, 'Pricing weight: ' + name)
    _require(weights['weights']['zeros'] == 0 and
             weights['dtype_rates']['float32'] == 1 and
             weights['dtype_rates']['float64'] == 2 and
             weights['dtype_rates']['int64'] == 2 and
             weights['dtype_rates']['bool'] == 1, 'Official dtype/zero tariff')
    bindings = {}
    for depth, name in ((3, '_PACKED_SQUARE'), (4, '_PACKED_SQUARE4'),
                        (5, '_PACKED_SQUARE5'), (6, '_PACKED_SQUARE6')):
        total = 0
        for start, stop, shape in constants[name]:
            _require(0 <= start < stop <= constants['_PACKED_ELEMENTS'],
                     'Square views must remain inside the existing arena')
            elements = 1
            for dim in shape:
                elements *= _integer(dim, 'Square view dimension', 1)
            _require(elements == stop - start, 'Square reshape extent')
            total += stop - start  # each public FP32 reshape is billed once
        bindings[depth] = total
    owners = {node.name: node.lineno for node in nodes
              if isinstance(node, ast.FunctionDef) and node.name in
              ('_mm', '_square_product', '_square_schedule',
               '_square_schedule_reference', '_cp3_bind_arena_views',
               '_packed_product_cached', '_packed_frame', 'select',
               'omitted_radial_source', '_predict_binary')}
    return dict(constants=constants, bindings=bindings, owners=owners,
                source_bindings={str(path): digest for path, digest in pins.items()})


def _packed_price(m, k, n, depth):
    """Same tariff as the current, measured census; includes operand copies."""
    unit = 2 ** depth
    _require(depth in DEPTHS and all(v > 0 and v % unit == 0 for v in (m, k, n)),
             'Packed aligned dimensions')
    ratio = Fraction(7, 4)
    value = 7 ** depth * (m // unit) * (n // unit) * (2 * k // unit - 1)
    value += ratio * (m * k + k * n + m * n) * sum(ratio ** j for j in range(depth))
    _require(value.denominator == 1, 'Packed tariff must be integral')
    return int(value)


def _square_schedule(m, k, n, constants):
    """Final lookup + source-identical fallback selection, NOT re-optimization."""
    if m == n:
        depth = constants['_SQUARE_EQUAL_DEPTHS'][m // 32 - 1][k // 32 - 1]
    elif (m == k and n == 1024) or (k == n and m == 1024):
        depth = constants['_SQUARE_CROSS_DEPTHS'][k // 32 - 1]
    else:
        depth, best, dims = 0, Fraction(2 * m * k * n - m * n), (m, k, n)
        for candidate, coefficients in constants['_SQUARE_COSTS']:
            unit = 2 ** candidate
            mp, kp, npad = (((v + unit - 1) // unit) * unit for v in (m, k, n))
            # Source coefficients are dyadic; at these <=1024, depth-aligned
            # dimensions their products/sums are exact in source binary64.
            # Fraction(float_literal) therefore preserves the same ordering.
            c0, c1, c2, c3 = (Fraction(v) for v in coefficients)
            cost = c0 * mp * kp * npad + c1 * mp * kp + c2 * kp * npad + c3 * mp * npad
            if (mp, kp, npad) != (m, k, n):
                # Preserve the old selection heuristic literally. Its padding
                # estimate is NOT the actual copy tariff used below.
                cost += 2 * (mp * kp + kp * npad)
            if cost < best:
                depth, best, dims = candidate, cost, (mp, kp, npad)
        return depth, dims
    unit = 2 ** depth
    return depth, tuple(((v + unit - 1) // unit) * unit for v in (m, k, n))


def _read_plan(q, R, a, model):
    """Only the existing read _mm; no all-n diagonal, gathers or scatter here."""
    result = dict(tiles=[], kernel_flops=0, padding_copy_flops=0,
                  inner_chunk_add_flops=0, square_depths=[], mm_without_bindings=0)
    if q == 0 or R == 0 or a == 0:
        return result  # current _mm zero-dimension zeros has zero FLOPs
    depths = set()
    for start in range(0, R, 8192):
        k = min(8192, R - start)
        shape, depth, kind = (q, k, a), 0, 'direct'
        if all(v % 32 == 0 for v in shape):
            if max(shape) <= 1024:
                depth, padded = _square_schedule(*shape, model['constants'])
                kind = 'square' if depth else 'direct'
                if depth:
                    depths.add(depth)
            else:
                depth, padded, kind = 4, shape, 'rectangle'
        if depth == 0:
            padded = shape
            kernel = q * a * (2 * k - 1)
        else:
            kernel = _packed_price(*padded, depth)
        mp, kp, npad = padded
        copies = (q * k if (mp, kp) != (q, k) else 0)
        copies += k * a if (kp, npad) != (k, a) else 0
        result['tiles'].append(dict(shape=list(padded), input_shape=list(shape),
                                    depth=depth, kind=kind, dtype='float32',
                                    flops=kernel, padding_copy_flops=copies))
        result['kernel_flops'] += kernel
        result['padding_copy_flops'] += copies
    result['inner_chunk_add_flops'] = (len(result['tiles']) - 1) * q * a
    result['square_depths'] = sorted(depths)
    result['mm_without_bindings'] = (result['kernel_flops'] +
        result['padding_copy_flops'] + result['inner_chunk_add_flops'])
    return result


def _outside_mm(q, R, a):
    # select() always runs, even at R=0. Its scan/sort/full-mask allocation
    # shapes are unchanged; only q bool scatter writes vary.
    result = dict(response_mask_put=q, gather_left=0, gather_right=0,
                  square_left=0, weight_right=0, response_cast=0,
                  scatter_indices=0, response_scatter=0)
    if R:
        result.update(gather_left=4*q*R, gather_right=4*a*R,
                      square_left=q*R, weight_right=a*R,
                      response_cast=2*q*a, scatter_indices=2*(q+q*a),
                      response_scatter=2*q*a)
    result['total'] = sum(result.values())
    return result


def _price(record, model, previously_paid=()):
    _require(isinstance(record, dict) and record.get('source_sha256') == SOURCE_SHA256,
             'Only the fixed d50 source may be priced')
    _require(type(record.get('rank')) is int and record['rank'] == 35, 'Fixed rank35')
    call = _integer(record.get('call'), 'Call')
    _require(call in (0, 1) and type(record.get('includes_initialization')) is bool and
             record['includes_initialization'] == (call == 0), 'Cold/warm ownership')
    actual = _integer(record.get('actual_flops'), 'Complete actual bill', 1)
    _require(actual <= 2 ** 41, 'Actual bill ceiling')
    before = _depths(record.get('square_depths_bound_before'), 'Call bound-before')
    after = _depths(record.get('square_depths_bound_after'), 'Call bound-after')
    _require(before <= after and (call != 0 or not before), 'Original binding lifecycle')
    layers = record.get('layers')
    _require(isinstance(layers, list) and len(layers) == 16, 'All16 layers required')
    diagnostics, required_new, current_read, proposed_read = [], set(), 0, 0
    last_bound, last_R, charged_read_depths = before, 0, set()
    for index, layer in enumerate(layers):
        _require(type(layer.get('layer')) is int and layer['layer'] == index, 'Layer order')
        q0, q1, a, R = (_integer(layer.get(key), 'Layer ' + str(index) + ' ' + key)
                        for key in ('q0', 'q1', 'a', 'R'))
        _require(0 <= q1 <= q0 <= a <= 1024 and
                 all(v % 32 == 0 for v in (q0, q1, a)), 'Rounded response/active counts')
        _require(R % 64 == 0 and last_R <= R <= last_R + 2048 and
                 R <= 2048 * index, 'Intact current birthpair carrier extent')
        old = _read_plan(q0, R, a, model)
        new = _read_plan(q1, R, a, model)
        required_new.update(new['square_depths'])
        rb = _depths(layer.get('read_square_depths_bound_before'), 'Read bound-before')
        ra = _depths(layer.get('read_square_depths_bound_after'), 'Read bound-after')
        _require(last_bound <= rb <= ra <= after, 'Read/global binding chronology')
        _require(ra == rb | set(old['square_depths']), 'Exact lazy read binding set')
        bound_here = ra - rb
        _require(not (bound_here & charged_read_depths), 'Read binding charged twice')
        charged_read_depths.update(bound_here)
        setup = sum(model['bindings'][depth] for depth in bound_here)
        read_actual = _integer(layer.get('read_actual_flops'), 'Measured outer read _mm')
        _require(read_actual == old['mm_without_bindings'] + setup,
                 'Measured read _mm tariff mismatch at layer ' + str(index))
        observed = layer.get('read_tiles')
        _require(isinstance(observed, list) and len(observed) == len(old['tiles']),
                 'Complete read leaf roster at layer ' + str(index))
        for tile, expected in zip(observed, old['tiles']):
            _require(tile.get('shape') == expected['shape'] and
                     type(tile.get('depth')) is int and tile['depth'] == expected['depth'] and
                     tile.get('dtype') == 'float32' and
                     type(tile.get('flops')) is int and tile['flops'] == expected['flops'],
                     'Measured leaf tariff/shape/depth mismatch at layer ' + str(index))
        old_out, new_out = _outside_mm(q0, R, a), _outside_mm(q1, R, a)
        old_price = old['mm_without_bindings'] + old_out['total']
        new_price = new['mm_without_bindings'] + new_out['total']
        current_read += old_price
        proposed_read += new_price
        diagnostics.append(dict(layer=index, q0=q0, q1=q1, a=a, R=R,
            current_mm=old, proposed_mm=new, current_outside_mm=old_out,
            proposed_outside_mm=new_out, measured_read_actual_flops=read_actual,
            observed_old_read_binding_flops=setup, old_binding_flops_retained=True,
            current_priced_flops=old_price, proposed_priced_flops=new_price,
            read_saving_flops=old_price-new_price,
            zero_response_rows=(q1 == 0), source_omission_product_saving_flops=0))
        last_bound, last_R = ra, R
    _require(current_read <= actual, 'Read subset exceeds complete actual bill')
    paid = set(previously_paid)
    _require(paid <= set(DEPTHS), 'Conservative preceding price binding set')
    extra_depths = required_new - after - paid
    extra = sum(model['bindings'][depth] for depth in extra_depths)
    unchanged = actual - current_read  # ALL original setup is deliberately kept
    counterfactual = unchanged + proposed_read + extra
    saving = actual - counterfactual
    return dict(schema=SCHEMA, rank=35, call=call,
        includes_initialization=record['includes_initialization'],
        source_sha256=SOURCE_SHA256, original_response_threshold='.01',
        proposed_response_threshold='.1', active_threshold='.01', birth_threshold='.01',
        actual_complete_flops=actual, current_priced_read_flops=current_read,
        proposed_priced_read_flops=proposed_read, unchanged_complete_remainder_flops=unchanged,
        extra_square_binding_flops=extra, extra_square_binding_depths=sorted(extra_depths),
        square_binding_flops_by_depth={str(d): f for d, f in model['bindings'].items()},
        conservative_bound_depths_after=sorted(after | required_new | paid),
        counterfactual_complete_flops_upper_bound=counterfactual,
        complete_saving_flops_lower_bound=saving,
        saving_fraction_lower_bound=dict(numerator=saving, denominator=actual),
        passes_eight_percent_information_gate=(100 * saving >= 8 * actual),
        all_observed_read_tiles_and_outer_bills_reconciled=True,
        all_original_initialization_and_nonresponse_work_retained=True,
        full_width_omitted_source_savings_flops=0, all_n_exact_diagonal_savings_flops=0,
        fixed_arena_bytes=4 * model['constants']['_PACKED_ELEMENTS'],
        binding_policy='retain all old setup; add newly required depths conservatively',
        standalone_warm_extra_binding_may_be_overcharged=(call == 1 and not paid),
        frozen_trajectory_only=True, candidate_complete_bill_claim=False,
        raw_error_or_wall_claim=False, source_bindings=model['source_bindings'],
        active_source_owner_lines=model['owners'], layers=diagnostics)


def _unchanged_sources(model):
    for path, digest in model['source_bindings'].items():
        _require(_sha(Path(path)) == digest, 'Pricing source changed during calculation')


def price_record(record):
    """Price one cold/warm call; conservatively charge unproven warm setup."""
    model = _model()
    result = _price(record, model)
    _unchanged_sources(model)
    return result


def price_pair(records):
    """Optional fixed cold35/repeat35 wrapper; avoids charging new setup twice."""
    _require(isinstance(records, list) and len(records) == 2, 'Exactly cold + warm')
    cold, warm = records
    _require(cold.get('call') == 0 and warm.get('call') == 1, 'Cold then warm')
    _require(set(warm['square_depths_bound_before']) == set(cold['square_depths_bound_after'])
             and set(warm['square_depths_bound_after']) == set(warm['square_depths_bound_before']),
             'Original warm repeat binding ownership')
    _require(cold['actual_flops'] - warm['actual_flops'] == INITIALIZATION_FLOPS,
             'Own cold/warm complete initialization difference')
    _require(len(cold['layers']) == len(warm['layers']) == 16, 'Two full layer rosters')
    for left, right in zip(cold['layers'], warm['layers']):
        _require(all(left.get(key) == right.get(key) for key in ('layer', 'q0', 'q1', 'a', 'R')),
                 'Exact repeated score-derived geometry')
    model = _model()
    first = _price(cold, model)
    second = _price(warm, model, first['conservative_bound_depths_after'])
    _unchanged_sources(model)
    return dict(schema=SCHEMA, records=[first, second],
                both_pass_eight_percent_information_gate=all(
                    item['passes_eight_percent_information_gate'] for item in (first, second)),
                cold_initialization_difference_flops=INITIALIZATION_FLOPS,
                repeated_counterfactual_binding_not_double_charged=True,
                frozen_trajectory_only=True, candidate_complete_bill_claim=False,
                source_bindings=model['source_bindings'])
