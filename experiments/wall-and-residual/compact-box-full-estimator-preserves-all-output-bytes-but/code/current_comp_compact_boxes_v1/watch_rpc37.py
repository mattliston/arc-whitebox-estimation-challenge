"""Whole qualified BOX RPC37 supervisor; canonical shared-layout candidate."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_push_frontier_boxes_v1/watch_rpc37.py'
OWNER_SHA='b00d1ab822d6e9ba9071741d5bbc4adc02c76bb175a34d323a02490abc928645'
ORIGINAL_PROTOCOL=OWNER.parent/'RPC37_PROTOCOL.md'
ORIGINAL_PROTOCOL_SHA='2eb20392b14e516fbd49a6c25a75e3038add17ec33758fcaa4bfa653e030b6c5'


def derive_source():
    if ORIGINAL_PROTOCOL.is_symlink() or hashlib.sha256(ORIGINAL_PROTOCOL.read_bytes()).hexdigest()!=ORIGINAL_PROTOCOL_SHA:
        raise RuntimeError('Whole original RPC resource/scientific contract')
    original=OWNER.read_text()
    if OWNER.is_symlink() or hashlib.sha256(original.encode()).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Complete qualified BOX supervisor owner')
    source,positions=original,[]
    def replace(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact canonical-layout supervisor splice: '+before[:100])
        for unused in range(count):
            at=source.index(before);positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    replace("CLIENT = NEW/'current_comp_padded_push_d6_v1/client.py'","CLIENT = HERE/'client.py'")
    replace("CLIENT: '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3'",
            "CLIENT: '51051da837a11cd9b769eaa43a993fa05b611cccbca4291ebf75acbea9b38e02'")
    replace("TAGS = ('unused_frozen_negative_baseline','current_comp_push_frontier_boxes37_v1')",
            "TAGS = ('unused_frozen_negative_baseline','current_comp_compact_boxes37_v1')")
    replace("schema='current_comp_push_frontier_boxes_rpc37_v1'",
            "schema='current_comp_compact_boxes_rpc37_v1'")
    replace("prefix='astra_frontier_boxes_rpc37_'","prefix='astra_compact_boxes_rpc37_'")
    replace("proof['full_source_inverse_verified'] and proof['parent_sha256'] == BASE_SHA",
            "proof['full_source_inverse_verified'] and proof['parent_sha256'] == "
            "'378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'"
            " and proof['c2_ancestor_sha256'] == BASE_SHA")
    replace("'Exact current-c2 plus fixed two-box schedule source inverse'",
            "'Exact BOX ancestor and original c2-head source inverse'")
    old=("        init = [None,new[0]['flops']-new[1]['flops']]\n"
         "        require(init == [None,1944970298+EXTRA_BINDING_FLOPS],'Actual complete initialization, no subtraction from served F')\n"
         "        require(tuple(n['flops'] for n in new) == (BASE_F[0]+EXTRA_BINDING_FLOPS,BASE_F[1]),\n"
         "            'Exact complete cold addition and unchanged warm F, no borrowed binding credit')")
    new=("        extra = [n['compact_binding_flops'] for n in new]\n"
         "        events = [n['compact_binding_events'] for n in new]\n"
         "        require(all(type(v) is int and v>=0 for v in (*extra,*events)),\n"
         "            'Actual scalar binding dimensions/events, not numerical array values')\n"
         "        require(extra == [3488967424,0] and events == [27,0],\n"
         "            'Fixed37 full canonical-box census; no omitted rebindings')\n"
         "        init = [None,new[0]['flops']-new[1]['flops']]\n"
         "        require(init == [None,1944970298+EXTRA_BINDING_FLOPS+extra[0]-extra[1]],\n"
         "            'Own complete startup including both old and all new bindings')\n"
         "        require(tuple(n['flops'] for n in new) ==\n"
         "            (BASE_F[0]+EXTRA_BINDING_FLOPS+extra[0],BASE_F[1]+extra[1]),\n"
         "            'Exact own cold/warm bills reconcile every executed reshape')\n"
         "        result.update(compact_binding_flops=extra,compact_binding_events=events,\n"
         "            cold_complete_cost_ratio_to_c2=new[0]['flops']/BASE_F[0])")
    replace(old,new)
    replace("ratio = walls[1][0]/walls[0][0]; advances = max(walls[1]) <= 45 and ratio <= .80",
            "ratio = walls[1][0]/walls[0][0]\n"
            "        cost_gate = new[0]['flops']/BASE_F[0] <= 1.015\n"
            "        advances = max(walls[1]) <= 45 and ratio <= .80 and cost_gate\n"
            "        result['complete_cold_cost_gate_passed'] = cost_gate")
    replace("source_bindings={str(k):v for k,v in {**own,**h.MON_PINS}.items()},",
            "source_bindings={str(k):v for k,v in {**own,**h.MON_PINS,OWNER:OWNER_SHA,\n"
            "            ORIGINAL_PROTOCOL:ORIGINAL_PROTOCOL_SHA}.items()},")
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional whole supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire original sampling/cleanup/publication supervisor restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        same_resource_observations_cleanup_publication=True,
        own_complete_actual_binding_deltas_required=True,
        original_45s_and_historical_0p80_gates_retained=True,
        prospective_maximum_cold_cost_ratio_to_c2=1.015)


_source,ADAPTER_PROOF=derive_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_boxes_watch_inert',
          OWNER=OWNER,OWNER_SHA=OWNER_SHA,ORIGINAL_PROTOCOL=ORIGINAL_PROTOCOL,
          ORIGINAL_PROTOCOL_SHA=ORIGINAL_PROTOCOL_SHA)
exec(compile(_source,str(Path(__file__).resolve())+':whole-qualified-supervisor','exec'),_api)
main=_api['main']
del _source


if __name__=='__main__':
    raise SystemExit(main())

