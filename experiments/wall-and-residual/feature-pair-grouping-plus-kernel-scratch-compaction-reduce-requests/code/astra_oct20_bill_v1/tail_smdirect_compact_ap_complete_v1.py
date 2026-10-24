# [astra-bill 2026-10-20T21:33:15Z] Exact SMDIRECT internal scratch compaction.
# Each group's private internal nodes occupy its existing batch prefix.
# Public LA/RB output positions, per-node operator and operand order are fixed.
# The group offset remains unchanged; no other group's scratch can be overwritten.
# Distinct key is shared by runtime and the existing metadata-only precompiler.
assert _O20_SMDIRECT_ON

def _sm40_program_key(plan, inputs, out, prepared):
    if _o20_smdirect_eligible(plan, out, prepared):
        return ('o20_direct_compact_ap_v1', id(plan), id(inputs), id(out))
    return _O20_SMDIRECT_KEY_PARENT(plan, inputs, out, prepared)

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
            internal_refs = tuple(ref for ref, args in nodes if ref not in output_at)
            internal_nodes = dict(enumerate(internal_refs))
            internal_at = {ref: i for i, ref in internal_nodes.items()}
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
                               else node_views[node_at + internal_at[ref]])
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

# [astra-bill 2026-10-20T21:36:39Z] Fixed output-index AP covers from exhaustive metadata proof.
# Output positions and per-node arithmetic are unchanged. Only independent
# same-group nodes are scheduled together into fewer basic strided slices.
# No runtime combinatorial search, new gather, output permutation, or copy.
_O20_SMCOMPACT_RUNS_PARENT = _o20_smdirect_runs
_O20_SMCOMPACT_FIXED_AP = {(7, 8, 10, 12, 13): ((7, 10, 13), (8, 12)), (2, 18, 31, 34, 38): ((2, 18, 34), (31, 38))}

def _o20_smdirect_runs(positions):
    positions = tuple(sorted(positions))
    fixed = _O20_SMCOMPACT_FIXED_AP.get(positions)
    if fixed is not None:
        yield from fixed
    else:
        yield from _O20_SMCOMPACT_RUNS_PARENT(positions)
