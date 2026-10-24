def guard_normalized(q, record, command, seconds):
    # Match actual nice child argv semantics; raw receipt remains unchanged.
    g = json.loads(q.checked(record))
    actual = g['command'][1:] if g['command'][:1] == ['--'] else g['command']
    assert actual == command
    assert g['completed'] and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows']
    assert g['rss_trigger_bytes'] == 7500000000 and 0 < g['max_rss_bytes'] < 7500000000
    assert g['max_swap_growth_bytes'] == 0 and 0 < g['elapsed_s'] <= seconds
    return g


def verify_cheap_evidence(q, c):
    d = Path(c['candidate']['path']).parent
    cheap = json.loads(q.checked(c['cheap_gate']))
    root = json.loads(q.checked(c['root_cheap_gate']))
    assert c['root_cheap_gate']['sha256'] == '7b20f949b6f34d759985975e77b553395247afdb808e063c45309a6f6e14ad71'
    assert cheap['candidate'] == root['candidate'] == c['candidate']
    assert cheap['base'] == c['native_comparator'] and cheap['passed'] and root['passed']
    assert cheap['all16_bytes_equal'] and cheap['flops_equal'] and cheap['requests_lower_each_rank']
    source_protocol = json.loads(q.checked(c['source_protocol']))
    q.checked(source_protocol['degraded_probe'])
    q.checked(root['source_review'])
    assert root['protocol'] == c['source_protocol']
    for record in cheap['evidence']:
        q.checked(record)
    normal_driver = H2/'hill_climb_essential_fusion_v1/probe_driver.py'
    assert pin(normal_driver)['sha256'] == '2fb0035189788e763ecec94a00c25dc0d71cfdbdd480b847e11c595d55e93cbe'
    q.checked(pin(normal_driver))
    for i, block in enumerate(root['normal_setup3'], 1):
        tag = d.name+f'_normal_setup{i}_v1'
        assert block['receipt'] == pin(H2/(tag+'_setup.json'))
        s = json.loads(q.checked(block['receipt']))
        assert s['candidate'] == c['candidate']['path'] and s['candidate_sha256'] == c['candidate']['sha256']
        assert s['mode'] == 'normal' and s['phase1_deadline'] == 4.0
        assert 0 <= s['setup_wall_s'] <= 4.1 and block['setup_s'] == s['setup_wall_s'] == cheap['normal_setup_seconds'][i-1]
        assert s['fused_entries'] == 649 and all(x['fused'] for x in s['essential'])
        for key, count in c['setup_counts'].items():
            assert [x[1] for x in s['setup_log'] if x[0] == key] == [count]
        guard_normalized(q, pin(d/f'normal_setup{i}_guard.json'),
            [str(q.NATIVE), '-B', str(normal_driver), c['candidate']['path'], '-', tag, 'normal'], 60)
    assert len(root['normal_setup3']) == 3
    a = H2/'hill_climb_all_eager_fusion_v1'
    driver = a/'probe_driver_degraded_v1.py'
    assert pin(driver)['sha256'] == 'fde8e8ab03a48b6c135992ed3b88d6843d2c89f144bf6b65e7f2e24aa742a700'
    q.checked(pin(driver))
    datasets = []
    for tag, source, gp, fused in (
            (a.name+'_phase1_3s_base_v1', c['native_comparator'], a/'phase1_3s_base_guard_v1.json', 0),
            (d.name+'_phase1_3s_candidate_v1', c['candidate'], d/'degraded_candidate_guard.json', 128)):
        s = json.loads(q.checked(pin(H2/(tag+'_setup.json'))))
        assert s['candidate_sha256'] == source['sha256'] and s['mode'] == 'phase1_3s'
        assert s['phase1_deadline'] == 4.0 and 0 <= s['setup_wall_s'] <= 4.1
        sim = s['phase1_simulation']
        assert sim['requested_phase1_budget_s'] == 3.0 and sim['restored_deadline_s'] == 4.0
        assert len(sim['phase1_bound_roster']) == 649 and len(sim['phase1_fused_roster']) == fused
        assert s['bound_entries'] == 649 and s['fused_entries'] == fused
        if fused:
            assert sim['phase1_fused_roster'] == source_protocol['prefix_keys']
        launch = json.loads(q.checked(pin(H2/(tag+'_launcher.json'))))
        assert launch['candidate'] == source['path'] and launch['candidate_sha256'] == source['sha256']
        assert launch['client_exit'] == 0 and launch['server_retired']
        guard_normalized(q, pin(gp), [str(q.NATIVE), '-B', str(driver), source['path'], '35,100', tag, 'phase1_3s'], 240)
        rows = json.loads(q.checked(pin(H2/(tag+'_census.json'))))
        assert [r['rank'] for r in rows] == [35, 100]
        for row in rows:
            p = pin(H2/f"{tag}_pred_{row['rank']}.bin")
            q.checked(p)
            assert p['sha256'] == row['prediction_sha256'] and p['bytes'] == 65536
            assert row['prediction_shape'] == [16, 1024] and row['prediction_dtype'] == 'float32' and row['nonfinite'] == 0
        datasets.append(rows)
    for b, x, check in zip(*datasets, root['rows']):
        assert b['prediction_sha256'] == x['prediction_sha256'] and b['flops'] == x['flops'] == x['flops_ctx']
        assert x['requests'] < b['requests'] and check['requests_saved'] == b['requests']-x['requests']
        assert check['rank'] == x['rank'] and check['all16_bytes_equal'] and check['flops_equal']
    failure = json.loads(q.checked(c['official_parent_failure']))
    # Preserved official failure is provenance, never a successful bank or timing anchor.
    assert failure['submission_id'] == 333099
