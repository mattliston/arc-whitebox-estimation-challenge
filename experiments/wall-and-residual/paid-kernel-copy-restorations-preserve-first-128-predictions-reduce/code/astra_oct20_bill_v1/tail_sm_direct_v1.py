# Exact Smirnov input-circuit output ownership.
# L/R outputs write directly to the existing separate LA/RB owner. A node may
# also feed later groups; those consumers read its new destination. Nodes in
# a group have the same depth/operator and cannot depend on each other.
# Each scalar arithmetic operation and operand order remains unchanged.
_O20_SMDIRECT_ON = True
_O20_SMDIRECT_PARENT = _astra_sm40_linear_grouped
_O20_SMDIRECT_KEY_PARENT = _sm40_program_key
_O20_SMDIRECT_PLANS = {
    id(record['prepared'][kind]): record['prepared'][kind]
    for record in _SMX_RECORDS.values() for kind in ('L', 'R')
}
_O20_SMDIRECT_STATS = dict(calls=0, compiled=0, avoided_stack_elements=0,
                          old_arithmetic_calls=0, new_arithmetic_calls=0)


def _o20_smdirect_eligible(plan, out, prepared):
    return (_O20_SMDIRECT_ON and out is not None
            and isinstance(prepared, _Sm40Prepared)
            and _O20_SMDIRECT_PLANS.get(id(plan)) is plan)


def _sm40_program_key(plan, inputs, out, prepared):
    if _o20_smdirect_eligible(plan, out, prepared):
        return ('o20_direct_v1', id(plan), id(inputs), id(out))
    return _O20_SMDIRECT_KEY_PARENT(plan, inputs, out, prepared)


def _o20_smdirect_runs(positions):
    """Partition sorted plane indices into exact basic strided slices."""
    positions = sorted(positions)
    at = 0
    while at < len(positions):
        if at + 1 == len(positions):
            yield tuple(positions[at:at + 1])
            return
        end = at + 2
        step = positions[at + 1] - positions[at]
        while end < len(positions) and positions[end] - positions[end - 1] == step:
            end += 1
        yield tuple(positions[at:end])
        at = end


def _astra_sm40_linear_grouped(fnp, plan, inputs, shape, stats,
                                out=None, prepared=None):
    if not _o20_smdirect_eligible(plan, out, prepared):
        return _O20_SMDIRECT_PARENT(fnp, plan, inputs, shape, stats,
                                    out=out, prepared=prepared)
    key = _sm40_program_key(plan, inputs, out, prepared)
    program = prepared.programs.get(key)
    if program is None:
        input_refs, groups, output_refs, uses = plan
        batches, node_views = prepared
        output_at = {ref: i for i, ref in enumerate(output_refs)}
        assert len(output_at) == len(output_refs) == int(out.shape[0])
        produced = {ref for _, nodes in groups for ref, args in nodes}
        assert set(output_at) <= produced
        output_views = tuple(fnp.unstack(out, axis=0))
        values = {ref: inputs[i] for ref, i in input_refs.items()}
        operations = {'add': fnp.add, 'sub': fnp.subtract,
                      'scale': fnp.multiply, 'neg': fnp.multiply}
        ops = []
        node_at = 0
        peak = 0
        scalar_bytes = _rc_math.prod(shape) * 4
        for group_index, ((depth, op, lit), nodes) in enumerate(groups):
            node_map = dict(nodes)
            output_nodes = {output_at[ref]: ref for ref, args in nodes
                            if ref in output_at}
            internal_nodes = {i: ref for i, (ref, args) in enumerate(nodes)
                              if ref not in output_at}
            # No reference created by this group is available until all its
            # operations have been compiled. This asserts same-group DAG
            # independence rather than relying only on its depth label.
            assert all(arg in values for ref, args in nodes for arg in args)
            for is_output, mapping in ((False, internal_nodes), (True, output_nodes)):
                owner = out if is_output else batches[group_index]
                for positions in _o20_smdirect_runs(mapping):
                    refs = tuple(mapping[i] for i in positions)
                    left = [values[node_map[ref][0]] for ref in refs]
                    right = ([values[node_map[ref][1]] for ref in refs]
                             if op in ('add', 'sub') else None)
                    if len(positions) == 1:
                        i = positions[0]
                        destination = output_views[i] if is_output else node_views[node_at + i]
                        lhs = left[0]
                        rhs = right[0] if right else -1 if op == 'neg' else lit
                    else:
                        step = positions[1] - positions[0]
                        destination = owner[positions[0]:positions[-1] + 1:step]
                        lhs = left
                        rhs = right if right else -1 if op == 'neg' else lit
                    ops.append((operations[op], lhs, rhs, destination))
            for i, (ref, args) in enumerate(nodes):
                values[ref] = (output_views[output_at[ref]] if ref in output_at
                               else node_views[node_at + i])
            node_at += len(nodes)
            peak = max(peak, 2 * len(nodes) * scalar_bytes)
        program = (inputs, out, tuple(ops), output_views, peak)
        if len(prepared.programs) >= 16:
            prepared.programs.clear()
        # Existing setup precompiler intercepts this exact insertion before
        # any arithmetic. Its key helper above follows the same runtime flag.
        prepared.programs[key] = program
        _O20_SMDIRECT_STATS['compiled'] += 1
    _, _, ops, output_views, peak = program
    held = []
    for operation, lhs, rhs, destination in ops:
        held.append(operation(lhs, rhs, out=destination))
    stats['temp_peak'] = max(stats['temp_peak'], peak)
    stats['temp_live'] = 0
    _O20_SMDIRECT_STATS['calls'] += 1
    _O20_SMDIRECT_STATS['avoided_stack_elements'] += _rc_math.prod(out.shape)
    _O20_SMDIRECT_STATS['old_arithmetic_calls'] += len(plan[1])
    _O20_SMDIRECT_STATS['new_arithmetic_calls'] += len(ops)
    return out
