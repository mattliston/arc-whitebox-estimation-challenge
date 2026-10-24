"""Validate immutable records and retirement separately from paid calls."""
from common import *

def verify_records(folder,source,role,stage,ranks,watch_path):
    folder=Path(folder);watch=read(watch_path)
    assert watch['completed'] and watch['exit_code']==0 and watch['whole_group_retired'] and not watch['remaining_rows']
    assert watch['max_rss_bytes']<7_500_000_000 and watch['max_swap_growth_bytes']==0
    result=read(folder/'result.json')
    assert result['completed'] and result['source']==source
    assert result['role']==role
    if stage is not None:assert result['stage']==stage
    rows=result['records'];assert [r['rank'] for r in rows]==ranks
    pins=[pin(folder/'result.json'),pin(watch_path),source]
    check(source)
    for index,row in enumerate(rows):
        assert row['index']==index and row['includes_initialization']==(index==0)
        for k in ('prediction','bill','weights','input_bill','input_reset','lifecycle'):
            check(row[k]);pins.append(row[k])
        bill=read(row['bill']['path']);life=read(row['lifecycle']['path'])
        assert bill['complete_counted_predict'] and not bill['initialization_subtracted']
        assert bill['flops']==row['flops']==life['closed_metrics']['flops']
        assert row['multiplier']==max(.1,row['flops']/2**41)
        assert bill['multiplier']==row['multiplier'] and 0<row['flops']<=2**41
        assert life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset']
        ip=read(row['input_bill']['path']);ir=read(row['input_reset']['path'])
        assert ip['exact_bytes'] and ip['owned_fnp_float32'] and ip['independent_buffers'] and not ip['charged_to_participant']
        assert ir['complete'] and ir['context_released'] and ir['closed_context_bill_unchanged_by_reset']
        assert len(row['domain'])==16 and len(row['selectors'])==45
        assert all(v['finite'] and v['minimum']>=1e-10 for v in row['domain'].values())
        assert row['numeric_cache_retired'] and not row['truth_accessed']
        if folder.resolve().is_relative_to(HERE.resolve()):
            assert row['gnn_layers']==list(range(16)) and row['recurrent_state_ownership_verified']
            states=row['gnn_states'];assert [s['layer'] for s in states]==list(range(16))
            assert all(states[i]['incoming_state_sha']==states[i-1]['outgoing_state_sha'] for i in range(1,16))
        retirement=folder/f'call_{index:02d}_retirement.json';rr=read(retirement)
        assert rr['complete'] and rr['all_tracked_call_objects_released'] and rr['tracked_objects']==18 and not rr['live_objects']
        assert rr['after_closed_saved_bill'] and rr['saved_bill_unchanged']
        check(rr['saved_bill']);assert rr['saved_bill']==row['bill']
        pins.append(pin(retirement))
    if len(rows)>1:
        assert rows[-1]['rank']==rows[0]['rank']
        assert rows[-1]['prediction']['sha256']==rows[0]['prediction']['sha256']
        assert rows[-1]['domain']==rows[0]['domain'] and rows[-1]['selectors']==rows[0]['selectors']
        if folder.resolve().is_relative_to(HERE.resolve()):assert rows[-1]['gnn_states']==rows[0]['gnn_states']
    return rows,pins
