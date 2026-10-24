# [astra-bill 2026-10-20T21:33:15Z] Exact SMDIRECT internal scratch compaction.
# Each group's private internal nodes occupy its existing batch prefix.
# Public LA/RB output positions, per-node operator and operand order are fixed.
# The group offset remains unchanged; no other group's scratch can be overwritten.
# Distinct key is shared by runtime and the existing metadata-only precompiler.
assert _O20_SMDIRECT_ON

def _sm40_program_key(plan, inputs, out, prepared):
    if _o20_smdirect_eligible(plan, out, prepared):
        return ('o20_direct_smdag_v1', id(plan), id(inputs), id(out))
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

# Static exact DAG schedule: every node keeps its original operator and ordered operands.
_SMDAG_FIXED_GROUPS = {('signed_executable_1', 'L'): (((0, 'scale', 8.0), ((18, (0,)), (20, (6,)), (22, (10,)), (24, (4,)), (31, (16,)), (33, (12,)))), ((1, 'sub', None), ((43, (8, 14)), (50, (9, 13)), (21, (15, 20)), (23, (17, 22)), (34, (9, 33)), (36, (7, 22)), (38, (3, 18)))), ((2, 'add', None), ((28, (9, 13)), (30, (3, 8)), (39, (7, 17)), (19, (14, 18)), (25, (5, 24)), (29, (8, 20)), (32, (1, 31)), (47, (2, 31)), (54, (5, 43)), (52, (2, 38)))), ((3, 'sub', None), ((26, (1, 23)), (27, (11, 19)), (41, (23, 34)), (45, (21, 30)), (46, (2, 38)), (48, (11, 32)), (60, (29, 36)), (63, (24, 54)))), ((4, 'add', None), ((53, (20, 39)), (42, (14, 25)), (49, (29, 36)), (51, (28, 47)), (64, (21, 30)), (35, (21, 26)), (55, (19, 48)), (61, (1, 41)))), ((5, 'sub', None), ((56, (5, 53)), (37, (24, 27)), (62, (26, 34)), (66, (27, 32)), (40, (25, 35)))), ((6, 'add', None), ((70, (55, 64)), (72, (45, 66)), (76, (61, 63)), (79, (40, 52)))), ((7, 'sub', None), ((69, (45, 66)), (71, (55, 64)), (74, (61, 63)), (84, (40, 52)))), ((8, 'add', None), ((59, (18, 51)), (65, (8, 42)), (44, (28, 37)), (57, (25, 35)), (89, (71, 74)), (85, (69, 79)))), ((9, 'add', None), ((67, (56, 59)), (75, (62, 65)), (81, (44, 49)), (83, (46, 57)))), ((10, 'sub', None), ((58, (37, 50)), (117, (69, 74)), (121, (70, 76)), (93, (75, 83)), (95, (84, 89)), (105, (72, 89)), (114, (71, 84)))), ((11, 'sub', None), ((68, (56, 59)), (73, (62, 65)), (78, (58, 60)), (80, (46, 57)), (82, (44, 49)))), ((12, 'add', None), ((77, (58, 60)), (92, (67, 76)), (108, (72, 73)), (86, (75, 80)), (87, (73, 85)), (96, (69, 93)), (99, (73, 95)), (127, (67, 93)))), ((13, 'sub', None), ((94, (68, 75)), (112, (71, 80)), (90, (78, 87)))), ((14, 'sub', None), ((116, (69, 92)), (118, (82, 117)), (123, (72, 94)), (88, (77, 86)), (104, (78, 85)), (113, (68, 112)), (102, (87, 92)), (91, (70, 90)))), ((15, 'add', None), ((110, (71, 92)), (97, (72, 96)), (100, (81, 91)))), ((16, 'add', None), ((109, (81, 108)), (106, (86, 105)), (122, (87, 121)), (124, (70, 99)), (126, (88, 89)), (101, (68, 100)))), ((17, 'sub', None), ((98, (82, 88)), (119, (90, 94)))), ((18, 'sub', None), ((107, (88, 99)), (103, (68, 97)), (111, (98, 110)), (125, (67, 98)), (128, (76, 97)), (115, (91, 114)), (120, (81, 119))))), ('signed_executable_1', 'R'): (((0, 'scale', 8.0), ((18, (5,)), (22, (8,)), (29, (11,)), (37, (6,)), (41, (7,)), (43, (9,)), (46, (4,)), (56, (15,)), (67, (17,)), (73, (3,)))), ((1, 'add', None), ((20, (3, 16)), (32, (4, 15)), (45, (3, 9)), (19, (13, 18)), (23, (1, 22)), (38, (0, 37)), (42, (2, 41)), (47, (14, 46)))), ((2, 'sub', None), ((21, (6, 10)), (28, (5, 11)), (31, (6, 15)), (33, (8, 17)), (34, (16, 17)), (24, (13, 22)), (30, (12, 29)), (25, (12, 19)), (44, (42, 43)))), ((3, 'add', None), ((27, (20, 21)), (26, (12, 19)), (52, (23, 43)), (74, (38, 73)))), ((4, 'scale', 8.0), ((35, (31,)), (39, (32,)), (50, (45,)), (53, (21,)), (59, (20,)), (62, (28,)), (70, (33,)), (75, (34,)), (48, (27,)))), ((5, 'sub', None), ((36, (0, 35)), (40, (2, 39)), (51, (23, 50)), (54, (30, 53)), (72, (2, 70)), (49, (14, 48)), (57, (52, 56)), (58, (47, 52)), (60, (44, 59)), (68, (26, 67)), (77, (25, 67)), (94, (25, 74)), (95, (26, 74)))), ((6, 'add', None), ((55, (30, 53)), (71, (2, 70)), (76, (38, 75)), (61, (44, 59)), (66, (24, 40)), (63, (49, 62)))), ((7, 'sub', None), ((65, (24, 40)), (69, (26, 51)), (64, (49, 62)))), ((8, 'sub', None), ((78, (58, 76)), (80, (55, 66)), (83, (54, 65)), (85, (61, 77)), (87, (60, 68)), (92, (36, 69)), (97, (57, 95)), (88, (64, 72)), (90, (63, 71)))), ((9, 'add', None), ((79, (58, 76)), (81, (55, 66)), (82, (54, 65)), (84, (61, 77)), (86, (60, 68)), (93, (36, 69)), (96, (57, 94)), (89, (64, 72)), (91, (63, 71)), (99, (80, 85)), (117, (92, 97)))), ((10, 'sub', None), ((98, (78, 80)), (100, (83, 85)), (101, (83, 84)), (102, (78, 84)), (104, (78, 87)), (105, (81, 87)), (107, (87, 89)), (109, (86, 91)), (110, (86, 88)), (113, (82, 90)))), ((11, 'add', None), ((103, (80, 84)), (106, (82, 86)), (115, (79, 96)), (111, (81, 88)), (122, (104, 117)), (125, (87, 101)), (119, (82, 107)))), ((12, 'sub', None), ((108, (89, 98)), (112, (90, 102)), (116, (96, 100)), (121, (82, 104)), (124, (81, 101)), (129, (85, 122)), (136, (81, 110)), (138, (99, 119)))), ((13, 'add', None), ((114, (97, 106)), (118, (92, 105)), (123, (79, 101)), (120, (93, 107)), (128, (83, 122)), (130, (86, 122)), (131, (81, 122)), (142, (113, 125)), (139, (99, 136)))), ((14, 'add', None), ((134, (109, 123)), (141, (113, 123)), (126, (115, 120)), (127, (111, 120)), (132, (109, 128)))), ((15, 'sub', None), ((133, (109, 124)), (135, (110, 129)), (137, (99, 120)), (140, (103, 126))))), ('signed_executable_1_transpose', 'L'): (((0, 'scale', 8.0), ((18, (13,)), (22, (14,)), (29, (15,)), (37, (2,)), (41, (8,)), (43, (3,)), (46, (7,)), (56, (5,)), (67, (17,)), (73, (1,)))), ((1, 'add', None), ((20, (1, 11)), (32, (5, 7)), (45, (1, 3)), (19, (10, 18)), (23, (6, 22)), (38, (0, 37)), (42, (12, 41)), (47, (16, 46)))), ((2, 'sub', None), ((21, (2, 9)), (28, (13, 15)), (31, (2, 5)), (33, (14, 17)), (34, (11, 17)), (24, (10, 22)), (30, (4, 29)), (25, (4, 19)), (44, (42, 43)))), ((3, 'add', None), ((27, (20, 21)), (26, (4, 19)), (52, (23, 43)), (74, (38, 73)))), ((4, 'scale', 8.0), ((35, (31,)), (39, (32,)), (50, (45,)), (53, (21,)), (59, (20,)), (62, (28,)), (70, (33,)), (75, (34,)), (48, (27,)))), ((5, 'sub', None), ((36, (0, 35)), (40, (12, 39)), (51, (23, 50)), (54, (30, 53)), (72, (12, 70)), (49, (16, 48)), (57, (52, 56)), (58, (47, 52)), (60, (44, 59)), (68, (26, 67)), (77, (25, 67)), (94, (25, 74)), (95, (26, 74)))), ((6, 'add', None), ((55, (30, 53)), (71, (12, 70)), (76, (38, 75)), (61, (44, 59)), (66, (24, 40)), (63, (49, 62)))), ((7, 'sub', None), ((65, (24, 40)), (69, (26, 51)), (64, (49, 62)))), ((8, 'sub', None), ((78, (58, 76)), (80, (55, 66)), (83, (54, 65)), (85, (61, 77)), (87, (60, 68)), (92, (36, 69)), (97, (57, 95)), (88, (64, 72)), (90, (63, 71)))), ((9, 'add', None), ((79, (58, 76)), (81, (55, 66)), (82, (54, 65)), (84, (61, 77)), (86, (60, 68)), (93, (36, 69)), (96, (57, 94)), (89, (64, 72)), (91, (63, 71)), (99, (80, 85)), (117, (92, 97)))), ((10, 'sub', None), ((98, (78, 80)), (100, (83, 85)), (101, (83, 84)), (102, (78, 84)), (104, (78, 87)), (105, (81, 87)), (107, (87, 89)), (109, (86, 91)), (110, (86, 88)), (113, (82, 90)))), ((11, 'add', None), ((103, (80, 84)), (106, (82, 86)), (115, (79, 96)), (111, (81, 88)), (122, (104, 117)), (125, (87, 101)), (119, (82, 107)))), ((12, 'sub', None), ((108, (89, 98)), (112, (90, 102)), (116, (96, 100)), (121, (82, 104)), (124, (81, 101)), (129, (85, 122)), (136, (81, 110)), (138, (99, 119)))), ((13, 'add', None), ((114, (97, 106)), (118, (92, 105)), (123, (79, 101)), (120, (93, 107)), (128, (83, 122)), (130, (86, 122)), (131, (81, 122)), (142, (113, 125)), (139, (99, 136)))), ((14, 'add', None), ((134, (109, 123)), (141, (113, 123)), (126, (115, 120)), (127, (111, 120)), (132, (109, 128)))), ((15, 'sub', None), ((133, (109, 124)), (135, (110, 129)), (137, (99, 120)), (140, (103, 126))))), ('signed_executable_1_transpose', 'R'): (((0, 'scale', 8.0), ((18, (0,)), (20, (1,)), (22, (13,)), (24, (12,)), (31, (14,)), (33, (2,)))), ((1, 'add', None), ((28, (5, 10)), (30, (7, 9)), (39, (4, 17)), (19, (8, 18)), (25, (15, 24)), (29, (7, 20)), (32, (3, 31)), (47, (6, 31)))), ((2, 'sub', None), ((43, (7, 8)), (50, (5, 10)), (21, (11, 20)), (23, (17, 22)), (34, (10, 33)), (36, (4, 22)), (38, (9, 18)), (27, (16, 19)), (48, (16, 32)))), ((3, 'sub', None), ((26, (3, 23)), (41, (23, 34)), (45, (21, 30)), (46, (6, 38)), (60, (29, 36)), (37, (24, 27)), (66, (27, 32)))), ((4, 'add', None), ((53, (20, 39)), (54, (15, 43)), (42, (8, 25)), (49, (29, 36)), (51, (28, 47)), (52, (6, 38)), (64, (21, 30)), (35, (21, 26)), (55, (19, 48)), (61, (3, 41)), (44, (28, 37)), (58, (37, 50)))), ((5, 'add', None), ((59, (18, 51)), (65, (7, 42)), (57, (25, 35)))), ((6, 'sub', None), ((56, (15, 53)), (63, (24, 54)), (62, (26, 34)), (40, (25, 35)), (69, (45, 66)), (71, (55, 64)), (78, (58, 60)), (80, (46, 57)), (82, (44, 49)))), ((7, 'sub', None), ((68, (56, 59)), (73, (62, 65)), (74, (61, 63)), (84, (40, 52)))), ((8, 'add', None), ((67, (56, 59)), (70, (55, 64)), (72, (45, 66)), (75, (62, 65)), (76, (61, 63)), (77, (58, 60)), (79, (40, 52)), (81, (44, 49)), (83, (46, 57)))), ((9, 'add', None), ((89, (71, 74)), (92, (67, 76)), (108, (72, 73)), (85, (69, 79)), (86, (75, 80)))), ((10, 'sub', None), ((94, (68, 75)), (117, (69, 74)), (121, (70, 76)), (93, (75, 83)), (95, (84, 89)), (105, (72, 89)), (112, (71, 80)), (114, (71, 84)))), ((11, 'add', None), ((109, (81, 108)), (110, (71, 92)), (87, (73, 85)), (96, (69, 93)), (99, (73, 95)), (106, (86, 105)), (127, (67, 93)))), ((12, 'sub', None), ((90, (78, 87)),)), ((13, 'sub', None), ((116, (69, 92)), (118, (82, 117)), (123, (72, 94)), (88, (77, 86)), (104, (78, 85)), (113, (68, 112)), (102, (87, 92)), (91, (70, 90)))), ((14, 'sub', None), ((98, (82, 88)), (119, (90, 94)))), ((15, 'add', None), ((97, (72, 96)), (100, (81, 91)))), ((16, 'add', None), ((122, (87, 121)), (124, (70, 99)), (126, (88, 89)), (101, (68, 100)))), ((17, 'sub', None), ((107, (88, 99)), (103, (68, 97)), (111, (98, 110)), (125, (67, 98)), (128, (76, 97)), (115, (91, 114)), (120, (81, 119))))), ('signed_executable_0', 'L'): (((0, 'add', None), ((9, (6, 7)), (10, (0, 2)), (11, (4, 5)), (14, (0, 1)), (26, (7, 8)))), ((1, 'sub', None), ((12, (6, 7)), (22, (0, 1)), (13, (4, 9)), (15, (10, 11)))), ((2, 'sub', None), ((21, (9, 14)), (16, (5, 13)), (19, (12, 15)))), ((3, 'add', None), ((23, (12, 22)), (24, (9, 14)), (27, (11, 26)), (18, (12, 15)), (25, (3, 13)), (20, (10, 16)))), ((4, 'sub', None), ((17, (10, 16)), (29, (18, 23)), (34, (20, 24)))), ((5, 'add', None), ((57, (21, 24)), (28, (17, 20)))), ((6, 'add', None), ((33, (18, 27)), (58, (25, 57)), (31, (17, 21)), (35, (19, 29)), (37, (21, 29)), (60, (20, 25)), (30, (19, 28)))), ((7, 'sub', None), ((36, (17, 25)), (39, (24, 29)), (49, (20, 33)), (42, (23, 34)), (47, (21, 34)), (50, (23, 31)), (53, (28, 33)), (55, (18, 28)), (32, (23, 30)), (46, (30, 33)))), ((8, 'sub', None), ((44, (27, 39)), (45, (18, 36)), (40, (21, 32)))), ((9, 'add', None), ((41, (24, 31)), (43, (31, 35)), (48, (19, 36)), (56, (25, 37)), (59, (34, 35)), (38, (24, 32)))), ((10, 'sub', None), ((51, (33, 41)), (52, (27, 40)))), ((11, 'add', None), ((54, (25, 38)),))), ('signed_executable_0', 'R'): (((0, 'scale', 8.0), ((18, (0,)), (20, (6,)), (22, (10,)), (24, (4,)), (31, (16,)), (33, (12,)))), ((1, 'sub', None), ((43, (8, 14)), (50, (9, 13)), (21, (15, 20)), (23, (17, 22)), (34, (9, 33)), (36, (7, 22)), (38, (3, 18)))), ((2, 'add', None), ((28, (9, 13)), (30, (3, 8)), (39, (7, 17)), (19, (14, 18)), (25, (5, 24)), (29, (8, 20)), (32, (1, 31)), (47, (2, 31)), (54, (5, 43)), (52, (2, 38)))), ((3, 'sub', None), ((26, (1, 23)), (27, (11, 19)), (41, (23, 34)), (45, (21, 30)), (46, (2, 38)), (48, (11, 32)), (60, (29, 36)), (63, (24, 54)))), ((4, 'add', None), ((53, (20, 39)), (42, (14, 25)), (49, (29, 36)), (51, (28, 47)), (64, (21, 30)), (35, (21, 26)), (55, (19, 48)), (61, (1, 41)))), ((5, 'sub', None), ((56, (5, 53)), (37, (24, 27)), (62, (26, 34)), (66, (27, 32)), (40, (25, 35)))), ((6, 'add', None), ((70, (55, 64)), (72, (45, 66)), (76, (61, 63)), (79, (40, 52)))), ((7, 'sub', None), ((69, (45, 66)), (71, (55, 64)), (74, (61, 63)), (84, (40, 52)))), ((8, 'add', None), ((59, (18, 51)), (65, (8, 42)), (44, (28, 37)), (57, (25, 35)), (89, (71, 74)), (85, (69, 79)))), ((9, 'add', None), ((67, (56, 59)), (75, (62, 65)), (81, (44, 49)), (83, (46, 57)))), ((10, 'sub', None), ((58, (37, 50)), (117, (69, 74)), (121, (70, 76)), (93, (75, 83)), (95, (84, 89)), (105, (72, 89)), (114, (71, 84)))), ((11, 'sub', None), ((68, (56, 59)), (73, (62, 65)), (78, (58, 60)), (80, (46, 57)), (82, (44, 49)))), ((12, 'add', None), ((77, (58, 60)), (92, (67, 76)), (108, (72, 73)), (86, (75, 80)), (87, (73, 85)), (96, (69, 93)), (99, (73, 95)), (127, (67, 93)))), ((13, 'sub', None), ((94, (68, 75)), (112, (71, 80)), (90, (78, 87)))), ((14, 'sub', None), ((116, (69, 92)), (118, (82, 117)), (123, (72, 94)), (88, (77, 86)), (104, (78, 85)), (113, (68, 112)), (102, (87, 92)), (91, (70, 90)))), ((15, 'add', None), ((110, (71, 92)), (97, (72, 96)), (100, (81, 91)))), ((16, 'add', None), ((109, (81, 108)), (106, (86, 105)), (122, (87, 121)), (124, (70, 99)), (126, (88, 89)), (101, (68, 100)))), ((17, 'sub', None), ((98, (82, 88)), (119, (90, 94)))), ((18, 'sub', None), ((107, (88, 99)), (103, (68, 97)), (111, (98, 110)), (125, (67, 98)), (128, (76, 97)), (115, (91, 114)), (120, (81, 119)))))}

assert not getattr(_SMX_ENV, '_sm40_group_cache', {})
assert not getattr(_ASTRA_SM40_ENV, '_sm40_group_cache', {})
_SMDAG_ORIGINAL_PLANS = {}
for (_smdag_name, _smdag_kind), _smdag_groups in _SMDAG_FIXED_GROUPS.items():
    _smdag_record = _SMX_RECORDS[_smdag_name]
    _smdag_old = _smdag_record['prepared'][_smdag_kind]
    _SMDAG_ORIGINAL_PLANS[_smdag_name, _smdag_kind] = _smdag_old
    _smdag_old_nodes = {r: (tuple(a), tag[1:]) for tag, items in _smdag_old[1] for r, a in items}
    _smdag_new_nodes = {r: (tuple(a), tag[1:]) for tag, items in _smdag_groups for r, a in items}
    assert _smdag_old_nodes == _smdag_new_nodes
    assert sum(len(items) for tag, items in _smdag_old[1]) == sum(len(items) for tag, items in _smdag_groups)
    _smdag_have = set(_smdag_old[0])
    for _smdag_tag, _smdag_items in _smdag_groups:
        assert all(a in _smdag_have for r, args in _smdag_items for a in args)
        _smdag_have.update(r for r, args in _smdag_items)
    assert set(_smdag_old[2]) <= _smdag_have
    # The existing record's prepared DICTIONARY is shared with every existing
    # kernel.circuits. Replacing its L/R tuple is visible to both group-view
    # allocation and evaluation. Inputs, output positions and use counts stay.
    _smdag_record['prepared'][_smdag_kind] = (_smdag_old[0], _smdag_groups, _smdag_old[2], _smdag_old[3])
_O20_SMDIRECT_PLANS = {
    id(record['prepared'][kind]): record['prepared'][kind]
    for record in _SMX_RECORDS.values() for kind in ('L', 'R')
}
