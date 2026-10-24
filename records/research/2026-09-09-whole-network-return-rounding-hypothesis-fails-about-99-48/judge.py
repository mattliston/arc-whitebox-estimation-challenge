"""Judge the fixed20 full-return captures; no model, derivative, target or fit.

An externally SHA-pinned source/roster manifest is mandatory. Completion and
integrity are gates; return-rounding dominance is a separate classification.
The old served-output derivative gates remain unchanged, including failures.
"""
import json
import math
from pathlib import Path
import sys
import traceback

import common as c

HERE = Path(__file__).resolve().parent
BUDGET = 2 ** 41
BR_JUDGE_SHA = 'a8ff76e5d020a59da6626d52b79993c9e28da118abdc458ff9bbcc1c9372f4f5'
NATIVE_OWNER_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
WATCH_HELPER_SHA = 'f79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b'
TIMINGS = ('wall_s', 'context_wall_s', 'flopscope_backend_s',
           'flopscope_overhead_s', 'native_residual_s')
LAYER_KEYS = {str(layer) for layer in range(16)}
SELECTOR_KEYS = {f'{layer}:{slot}' for layer in range(16)
                 for slot in range(2 if layer == 15 else 3)}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def bill_guard(bill):
    require(type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET,
            'Positive complete native bill within original budget')
    require(all(isinstance(bill[key], (int, float)) and math.isfinite(bill[key])
                and bill[key] >= 0 for key in TIMINGS), 'Finite nonnegative native timings')
    require(bill['wall_s'] < 120 and bill['peak_rss_bytes'] < 8_000_000_000
            and type(bill['operation_count']) is int and bill['operation_count'] >= 0,
            'Original per-call wall/RSS/operation-count gates')


def read_owned(stage, item, folder):
    path = Path(item['path']).resolve()
    require(path.parent == folder, 'Receipt must belong to its singleton')
    stage.pin(path, item['sha256'])
    return json.loads(path.read_text())


def lifecycle_guard(life, bill, binding, rank):
    require(life['saved_bill'] == binding['path'] and life['saved_bill_sha256'] == binding['sha256']
            and life['reset_api'] == 'flopscope.budget_reset'
            and life['after_context_exit'] is True and life['candidate_state_reset'] is False
            and life['closed_context_bill_unchanged_by_reset'] is True
            and life['context_released'] is True and life['complete'] is True
            and life['call'] == 0 and life['rank'] == rank, 'Immutable closed bill/reset ownership')
    closed = life['closed_metrics']
    for left, right in (('flops', 'flops'), ('operation_count', 'operation_count'),
                        ('context_wall_s', 'context_wall_s'), ('backend_s', 'flopscope_backend_s'),
                        ('overhead_s', 'flopscope_overhead_s'), ('residual_s', 'native_residual_s')):
        require(closed[left] == bill[right], 'Reset preserved actual closed metrics')


def acquisition_guard(stage, record, role, rank, case, old_native, proof):
    folder = HERE/f'{role}_{case}_r{rank:04d}'
    require(record['schema'] == 'current_d50_full_return_capture_v1'
            and record['role'] == role and record['case'] == case and record['rank'] == rank
            and record['hashes_unchanged'] is True and record['sources_unchanged'] is True,
            'Exact completed source/cell identity')
    require(record['truth_accessed'] is False and record['MPS_used'] is False
            and record['network_calls'] == 1 and record['gradient_calls'] == 0
            and record['parameter_updates'] == 0 and record['training_admitted'] is False
            and record['score_measured'] is False and record['new_derivative_qualified'] is False
            and record['original_derivative_gates_unchanged'] is True,
            'Capture-only acquisition scope')
    require(record['resources']['peak_rss_bytes'] < 8_000_000_000 and record['elapsed_s'] < 120,
            'Singleton full-process resource limits')
    old_cpu = stage.old_cpu['outputs'][str(rank)][case]
    require(record['candidate_sha256'] == old_native['candidate_sha256']
            and record['candidate'] == old_native['source']
            and record['payload'] == stage.old_cpu['payloads'][case]
            and record['cotangent'] == stage.old_cpu['cotangents'][str(rank)], 'Original candidate/P/cotangent')
    require(record['prediction'] == record['served'] and record['raw_capture_is_existing_return'] is True
            and record['served_reconstruction_byte_exact'] is True, 'Existing raw-return capture ownership')
    expected_files = set()
    for key, dtype in (('raw_unclamped', 'float64'), ('pre_return', 'float64'), ('served', 'float32')):
        item = record[key]
        require(item['shape'] == [16, 1024] and item['dtype'] == dtype
                and Path(item['path']).resolve() == folder/(key+'.npy'), 'Canonical capture array contract')
        require(record['artifacts'][item['path']] == item['sha256'], 'Capture artifact seal')
        expected_files.add(item['path'])
    d = record['diagnostics']
    require(record['all16_domains_present'] is True and record['all47_selectors_present'] is True
            and set(d['variance_domain']) == LAYER_KEYS and set(d['selection_cells']) == SELECTOR_KEYS,
            'Complete all16/47 diagnostics')
    require(all(v['finite'] is True and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10
                and v['below_floor_count'] == 0 for v in d['variance_domain'].values()), 'No variance floor events')
    if role == 'cpu':
        require(record['environment']['versions'] == {'numpy': '1.26.4', 'torch': '2.7.0'}
                and record['environment']['device'] == 'cpu'
                and record['environment']['moment_dtype'] == 'torch.float64', 'Original CPU frame')
        require(record['old_cpu_served_byte_exact'] is True and record['old_cpu_scalar_exact'] is True
                and record['old_cpu_diagnostics_exact'] is True and d == old_cpu['diagnostics']
                and record['synthetic_loss'] == old_cpu['synthetic_loss'], 'Exact original CPU replay')
        require(record['training'] is False and record['parent_reruns'] == 0
                and record['native_G0_injected'] is False and record['all_payload_bytes_unchanged'] is True
                and record['frozen_cstar_bytes_unchanged'] is True
                and record['payload_arrays_before'] == record['payload_arrays_after']
                == stage.old_cpu['payloads'][case]['arrays'], 'No CPU parameter/initialization action')
    else:
        require(record['environment']['versions'] == {'numpy': '2.4.6', 'flopscope': '0.12.0',
                'whestbench': '0.16.0', 'threadpoolctl': '3.6.0'}, 'Original native runtime frame')
        require(record['ownership_proof'] == proof and proof['original_native_sha256'] == NATIVE_OWNER_SHA,
                'Source-replayed paid-block/reset ownership proof')
        require(record['old_native_served_byte_exact'] is True and record['observed_predict_calls'] == 1
                and record['raw_capture_numeric_bytes'] == 16*1024*8
                and record['candidate_source_edited'] is False, 'One passive original native capture')
        require(record['cold'] is True and record['all_primary_cold'] is True
                and record['includes_initialization'] is True and record['initialization_subtracted'] is False
                and record['input_preparation_in_participant_bill'] is False, 'Own full cold participant bill')
        pools = record['threadpools']
        require(isinstance(pools, list) and all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas'),
                'Every reported native BLAS pool is one-threaded; empty is unknown inventory')
        require(d['variance_count'] == 16 and d['selector_count'] == 47
                and d['added_participant_array_operations'] == 0
                and d['gaussian_observation'] == 'Unfloored v at call entry', 'Native observation boundary')
        for key, selector in d['selection_cells'].items():
            ids, members = selector['ordered_ids'], selector['membership']
            require(selector['count'] == len(ids) and len(ids) % 32 == 0
                    and len(set(ids)) == len(ids) and sorted(ids) == members
                    and all(type(i) is int and 0 <= i < 1024 for i in ids)
                    and selector['floor'] == .01, 'Native selector internal validity')
        require(all(d['selection_cells'][f'0:{slot}']['ordered_ids'] == list(range(1024)) for slot in range(3)),
                'Original full Gaussian first-layer selection')
        for key in ('bill', 'input_bill', 'lifecycle_receipt', 'input_lifecycle_receipt'):
            expected_files.add(record[key]['path'])
            require(record['artifacts'][record[key]['path']] == record[key]['sha256'], 'Owned accounting artifact')
        bill = read_owned(stage, record['bill'], folder)
        ibill = read_owned(stage, record['input_bill'], folder)
        life = read_owned(stage, record['lifecycle_receipt'], folder)
        ilife = read_owned(stage, record['input_lifecycle_receipt'], folder)
        require(bill == record['complete_bill'] and ibill == record['input_preparation']
                and life == record['lifecycle'] and ilife == record['input_lifecycle'], 'Actual accounting JSON identity')
        bill_guard(bill); bill_guard(ibill)
        require(bill['call'] == ibill['call'] == 0 and bill['rank'] == ibill['rank'] == rank,
                'Singleton billed rank')
        require(bill['cold'] is True and bill['includes_initialization'] is True
                and bill['observed_predict_calls'] == 1 and bill['capture_serialization_inside_context'] is True
                and bill['multiplier'] == max(.1, bill['flops']/BUDGET), 'Complete actual prediction bill')
        require(ibill['charged_to_participant'] is False and ibill['flops'] == 16*1024*1024
                and ibill['layers'] == 16 and ibill['context_closed'] is True
                and ibill['official_fnp_type'] is True and ibill['exact_bytes'] is True
                and ibill['independent_weight_buffers'] is True, 'Separate official-input ownership')
        lifecycle_guard(life, bill, record['bill'], rank)
        lifecycle_guard(ilife, ibill, record['input_bill'], rank)
        old = next(x for x in old_native['trials'] if x['rank'] == rank)
        require(record['original_reference_includes_initialization'] == old['includes_initialization']
                and record['original_reference_flops'] == old['flops'], 'Original native cost reference identified')
        if old['includes_initialization']:
            require(record['cold_bill_matches_original_if_cold'] is True and bill['flops'] == old['flops'],
                    'Exact old comparable cold bill')
        else:
            require(record['cold_bill_matches_original_if_cold'] is None,
                    'Do not manufacture cold100 from the old warm reference')
    require(set(record['artifacts']) == expected_files, 'Exactly declared capture/accounting artifacts')


def main():
    args = c.arguments('judge')
    if not args.run:
        return 0
    require(args.manifest and args.manifest_sha256, 'Reviewed source/roster manifest and SHA are mandatory')
    stage = c.Stage('judge', args, Path(__file__))
    result, h, model = stage.result, stage.h, stage.model
    try:
        manifest_path = Path(args.manifest).resolve()
        require(manifest_path == HERE/'launch_manifest_v1.json', 'Canonical independently pinned launch manifest')
        manifest_pin = stage.pin(manifest_path, args.manifest_sha256)
        manifest = json.loads(manifest_path.read_text())
        source_paths = {str(HERE/name) for name in ('native.py', 'cpu.py', 'judge.py', 'common.py', 'watch.py', 'PROTOCOL.md')}
        source_paths.add(str(c.Q/'watch.py'))
        require(manifest['schema'] == 'full_return_launch_manifest_v1'
                and set(manifest['source_bindings']) == source_paths
                and manifest['source_bindings'][str(c.Q/'watch.py')] == WATCH_HELPER_SHA,
                'Exact externally authorized seven-source/helper roster')
        for path, digest in manifest['source_bindings'].items():
            stage.pin(path, digest)
        require(manifest['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256
                and manifest['source_bindings'][str(HERE/'common.py')] == args.common_sha256
                and manifest['source_bindings'][str(HERE/'PROTOCOL.md')] == args.protocol_sha256,
                'Manifest agrees with caller-reviewed source pins')
        expected = [dict(role=role, case=case, rank=rank,
                         folder=str(HERE/f'{role}_{case}_r{rank:04d}'))
                    for role in ('native', 'cpu') for case in c.CASES for rank in c.RANKS]
        roster = manifest['roster']
        require(len(roster) == 20 and roster == expected,
                'Exactly the original20 singleton cells')
        for row in roster:
            require(Path(row['folder']).resolve() == HERE/f"{row['role']}_{row['case']}_r{row['rank']:04d}",
                    'Canonical singleton folder')
        result.update(manifest=manifest_pin, manifest_roster=roster, stage_seals=[], endpoint_reconstruction=[],
                      measurements=[], original_native_scalar_receipt_sha256=BR_JUDGE_SHA,
                      new_derivative_qualified=False, dominance_is_advancement=False,
                      native_CPU_selector_equality_is_required=False)
        native_owner = c.module(HERE/'native.py', 'full_return_source_ownership_only')
        proof = native_owner.ownership_proof()  # AST/source only, not a native/estimator call.
        result['native_ownership_proof_replayed'] = proof
        old_natives = {case: stage.old_native_case(case) for case in c.CASES}
        stages = {}
        for row in roster:
            role, rank, case = row['role'], row['rank'], row['case']
            folder = Path(row['folder']).resolve()
            rp, cp = folder/'result.json', folder/'completion.json'
            rb, cb = stage.pin(rp), stage.pin(cp)
            seal = json.loads(cp.read_text())
            require(seal['result_sha256'] == rb['sha256'], 'Pinned result/completion pair')
            record = stage.sealed(folder, seal['result_sha256'], True)
            for path in (HERE/(role+'.py'), HERE/'common.py', HERE/'PROTOCOL.md'):
                require(record['bindings'][str(path)]['sha256'] == manifest['source_bindings'][str(path)],
                        'Acquisition used the externally authorized source')
            acquisition_guard(stage, record, role, rank, case, old_natives[case], proof)
            stages[role, rank, case] = record
            result['stage_seals'].append(dict(row, result_sha256=rb['sha256'], completion_sha256=cb['sha256']))
            stage.guard()
        # This historical judge stopped at the first failing secant. Its five
        # rank35 scalars are references, never a claim of completed qualification.
        brp, bcp = c.BR/'judgment_v1/result.json', c.BR/'judgment_v1/completion.json'
        stage.pin(brp, BR_JUDGE_SHA); stage.pin(bcp)
        bridge, bridge_seal = json.loads(brp.read_text()), json.loads(bcp.read_text())
        require(bridge['passed'] is False and bridge['completed'] is False
                and bridge_seal['passed'] is False and bridge_seal['completed'] is False
                and bridge_seal['result_sha256'] == BR_JUDGE_SHA, 'Preserve original failed bridge judgment')
        stage.inventory(bridge)
        available = {(x['rank'], x['case']): x['synthetic_loss'] for x in bridge['comparisons']}
        require(set(available) == {(35, case) for case in c.CASES}, 'Exact recorded old native scalar subset')
        result['all20_sources_inputs_artifacts_sealed_before_array_reads'] = True
        require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'All acquisitions authenticated before numerical imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Original scalar library frame')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        result['environment'] = dict(versions={'numpy': np.__version__, 'torch': torch.__version__}, device='cpu')

        def linear(value, cot):
            y = torch.from_numpy(value).to(torch.float64)
            q = torch.as_tensor(cot, dtype=torch.float64)
            products = y*q
            total = products[-1].sum()/math.sqrt(1024) + .05*products[:-1].sum()/math.sqrt(15*1024)
            layers = [float(.05*products[l].sum()/math.sqrt(15*1024)) for l in range(15)]
            layers.append(float(products[-1].sum()/math.sqrt(1024)))
            return float(total), layers

        def norm(value):
            return float(np.linalg.norm(value))

        def comparison(dy, dz, dq, sy, sz, sq):
            nv, nz = norm(dy), norm(dz)
            vr, sr = norm(dy-dz-dq), abs(sy-sz-sq)
            require(vr <= 1e-12 and sr <= 1e-12, 'Original signed additive decomposition')
            nontrivial = nv > 1e-10 and abs(sy) > 1e-10
            ratios = (nz/nv, abs(sz)/abs(sy)) if nontrivial else (None, None)
            return dict(served_discrepancy_l2=nv, pre_return_discrepancy_l2=nz,
                rounding_discrepancy_l2=norm(dq), served_scalar=sy, pre_return_scalar=sz,
                rounding_scalar=sq, vector_identity_residual=vr, scalar_identity_residual=sr,
                pre_return_norm_ratio=ratios[0], pre_return_absolute_scalar_ratio=ratios[1],
                rounding_signed_scalar_fraction=sq/sy if nontrivial else None,
                rounding_projection_fraction=float(np.sum(dy*dq)/np.sum(dy*dy)) if nontrivial else None,
                nontrivial=nontrivial, local_dominance=bool(nontrivial and ratios[0] <= .25 and ratios[1] <= .25))

        for rank in c.RANKS:
            cot = stage.load(np, stage.old_cpu['cotangents'][str(rank)])
            require(cot.shape == (16, 1024) and cot.dtype == np.float32, 'Original cotangent bytes')
            endpoints = {role: {} for role in ('native', 'cpu')}
            for role in endpoints:
                for case in c.CASES:
                    record = stages[role, rank, case]
                    raw, z, y = [stage.load(np, record[key]) for key in ('raw_unclamped', 'pre_return', 'served')]
                    require(raw.shape == z.shape == y.shape == (16, 1024)
                            and raw.dtype == z.dtype == np.float64 and y.dtype == np.float32
                            and (z >= 0).all() and (y >= 0).all(), 'Finite whole-return frames')
                    reconstructed = np.maximum(raw, 0.) if role == 'native' else np.where(raw < 0, np.zeros_like(raw), raw)
                    require(reconstructed.tobytes() == z.tobytes() and z.astype(np.float32).tobytes() == y.tobytes(),
                            'Own raw/clamp/return-byte reconstruction')
                    require(record['clipping_count_by_layer'] == [int((v < 0).sum()) for v in raw]
                            and record['clipping_count'] == int((raw < 0).sum()), 'Clipping observations replay')
                    if role == 'cpu':
                        meta = stage.old_cpu['outputs'][str(rank)][case]['prediction']
                    else:
                        old = next(x for x in old_natives[case]['trials'] if x['rank'] == rank)
                        meta = dict(path=old['output'], sha256=old['output_sha256'], shape=[16, 1024],
                                    dtype='float32', raw_sha256=old['output_bytes_sha256'])
                    old_y = stage.load(np, meta)
                    require(y.tobytes() == old_y.tobytes(), 'Every original endpoint reproduced exactly')
                    sy = float(h.scalar(torch, model, torch.from_numpy(y), cot, torch.float64))
                    direct_y, layers_y = linear(y, cot)
                    sz, layers_z = linear(z, cot)
                    old_sy = float(h.scalar(torch, model, torch.from_numpy(old_y), cot, torch.float64))
                    require(sy == direct_y == old_sy, 'Exact original scalar grouping/endpoint')
                    if role == 'cpu':
                        require(sy == stage.old_cpu['outputs'][str(rank)][case]['synthetic_loss']
                                == record['synthetic_loss'], 'Exact old CPU scalar')
                    elif (rank, case) in available:
                        require(sy == available[rank, case], 'Exact available old native bridge scalar')
                    require(all(math.isfinite(v) for v in (sy, sz, *layers_y, *layers_z)), 'Finite scalar decomposition')
                    result['endpoint_reconstruction'].append(dict(role=role, rank=rank, case=case,
                        old_served_byte_exact=True, raw_reconstruction_byte_exact=True, original_scalar_exact=True,
                        old_native_stored_scalar_available=role == 'native' and (rank, case) in available,
                        served_scalar=sy, pre_return_scalar=sz, layer_served_scalars=layers_y,
                        layer_pre_return_scalars=layers_z,
                        layer_sum_grouping_residual_served=sy-math.fsum(layers_y),
                        layer_sum_grouping_residual_pre_return=sz-math.fsum(layers_z)))
                    endpoints[role][case] = dict(y=y.astype(np.float64), z=z, sy=sy, sz=sz,
                                                ly=np.asarray(layers_y), lz=np.asarray(layers_z))
            for suffix, amplitude in (('025', .025), ('050', .05)):
                responses = {}
                for role in endpoints:
                    plus, minus = endpoints[role]['plus'+suffix], endpoints[role]['minus'+suffix]
                    ry, rz = (plus['y']-minus['y'])/(2*amplitude), (plus['z']-minus['z'])/(2*amplitude)
                    sy, sz = (plus['sy']-minus['sy'])/(2*amplitude), (plus['sz']-minus['sz'])/(2*amplitude)
                    ly, lz = (plus['ly']-minus['ly'])/(2*amplitude), (plus['lz']-minus['lz'])/(2*amplitude)
                    responses[role] = dict(y=ry, z=rz, q=ry-rz, sy=sy, sz=sz, sq=sy-sz, ly=ly, lz=lz, lq=ly-lz)
                n, a = responses['native'], responses['cpu']
                dy, dz, dq = n['y']-a['y'], n['z']-a['z'], n['q']-a['q']
                sy, sz, sq = n['sy']-a['sy'], n['sz']-a['sz'], n['sq']-a['sq']
                measured = comparison(dy, dz, dq, sy, sz, sq)
                direct_dy, _ = linear(dy, cot)  # Linear only: never serve or clip this signed response.
                require(abs(direct_dy-sy) <= 1e-12, 'Endpoint-derived scalar agrees with S(Dy)')
                measured.update(direct_signed_response_scalar=direct_dy,
                                scalar_grouping_residual=direct_dy-sy)
                old_secant = next(x['symmetric_secant'] for x in stage.old_cpu['secants']
                                  if x['rank'] == rank and x['amplitude'] == amplitude)
                require(a['sy'] == old_secant, 'Original CPU served secant exact')
                old_bridge = [x for x in bridge['secants'] if x['rank'] == rank and x['amplitude'] == amplitude]
                require(not old_bridge or (len(old_bridge) == 1 and n['sy'] == old_bridge[0]['native_secant']
                        and a['sy'] == old_bridge[0]['cpu_secant']), 'Preserve available failed native secant')
                layers = []
                for layer in range(16):
                    layers.append(dict(layer=layer, **comparison(dy[layer], dz[layer], dq[layer],
                        float(n['ly'][layer]-a['ly'][layer]), float(n['lz'][layer]-a['lz'][layer]),
                        float(n['lq'][layer]-a['lq'][layer]))))
                arrays = {key: stage.save(np, f'r{rank:04d}_h{suffix}_{key}.npy', value)
                          for key, value in (('Dy', dy), ('Dz', dz), ('Q', dq))}
                result['measurements'].append(dict(rank=rank, amplitude=amplitude, primary=measured,
                    arrays=arrays, layers=layers,
                    arm_scalar_responses={role: {key: responses[role][key] for key in ('sy', 'sz', 'sq')}
                                          for role in responses},
                    layer_sum_grouping_residuals={key: total-math.fsum(x[key] for x in layers)
                        for key, total in (('served_scalar', sy), ('pre_return_scalar', sz), ('rounding_scalar', sq))},
                    original_served_relative_error=abs(sy)/max(abs(n['sy']), abs(a['sy']), 1e-10),
                    original_five_percent_gate_reinterpreted=False))
                stage.guard()
        require(len(result['endpoint_reconstruction']) == 20 and len(result['measurements']) == 4,
                'All fixed endpoints and rank/amplitude comparisons completed')
        result.update(return_rounding_locally_dominant_all_four=all(x['primary']['local_dominance'] for x in result['measurements']),
                      native_total_diagnostic_flops=sum(x['complete_bill']['flops'] for key, x in stages.items() if key[0] == 'native'),
                      cost_is_diagnostic_not_estimator_saving=True, full_card_capture_completed=True,
                      smooth_derivative_qualified=False, old_failed_gates_preserved=True)
        require(not any(x in sys.modules for x in ('flopscope', 'threadpoolctl')), 'No native numerical runtime in judgment')
        stage.guard()
        stage.finish()  # Diagnostic completion, independent of dominance.
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                      error=traceback.format_exc())
    finally:
        stage.close()
    return 0 if result.get('passed') and result.get('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
