"""Independent standard-library anchor invoice reconciliation; no prediction work.

Run only on complete six-source anchors. Every charged operation cost must sum
exactly to its own reported total, before comparing plain and capture invoices.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def invoice(row):
    assert row['phase']=='scored' and row['nonfinite']==0 and row['n_out']==16384
    total=row['flops'];assert isinstance(total,int) and not isinstance(total,bool) and total>0
    operations=row['operations'];assert isinstance(operations,dict) and operations
    positive={};zero=[];summed=0
    for key,value in operations.items():
        assert isinstance(key,str) and isinstance(value,list) and len(value)==2
        count,cost=value
        assert isinstance(count,int) and not isinstance(count,bool) and count>=0
        assert isinstance(cost,int) and not isinstance(cost,bool) and cost>=0
        summed+=cost
        if cost:positive[key]=cost
        else:zero.append(dict(operation=key,calls=count))
    assert summed==total,(row['rank'],summed,total)
    return dict(total_flops=total,sum_operation_flops=summed,charged_map=positive,zero_flop_operations=zero,
                requests=row['requests'],prediction_sha256=row['prediction_sha256'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ready-sha256',required=True);ap.add_argument('--anchors-sha256',required=True);args=ap.parse_args()
    ready_path=HERE/'capture_plans_v1/CAPTURE_READY_v1.json';anchors_path=HERE/'ANCHORS_FINAL_v1.json'
    assert sha(ready_path)==args.ready_sha256 and sha(anchors_path)==args.anchors_sha256
    ready,anchors=read(ready_path),read(anchors_path)
    assert ready['successful'] and anchors['successful'] and anchors['ready_sha256']==sha(ready_path)
    assert anchors['all_six_prediction_and_charged_invoice_anchors_exact'] and anchors['actual_scored_predictions']==36
    rows=[]
    for family in ready['outputs']:
        key=family['key'];source=family['teacher']['sha256'];pair=[]
        for stage in ('control','anchor'):
            item=family['outputs'][stage];plan=ROOT/item['plan']['path'];assert sha(plan)==item['plan']['sha256']
            seeds=read(plan)['groups'];assert len(seeds)==1 and len(seeds[0])==3;seeds=seeds[0]
            folder=Path(item['out']);part=folder/'part01';manifest=read(folder/'RUN_MANIFEST.json')
            assert manifest['candidate_sha256']==source and not manifest['warmup_each'] and manifest['blas_threads']==4
            guard_path=folder/'part01_GUARD.json';guard=read(guard_path)
            assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
            assert guard['max_owned_swap_bytes']==0 and all(v==0 for v in guard['cgroup_events_final'].values())
            path=part/('rows.jsonl' if stage=='control' else 'diagnostic_rows.jsonl')
            raw=[json.loads(line) for line in path.read_text().splitlines()]
            assert [r['rank'] for r in raw]==seeds
            checked={r['rank']:invoice(r) for r in raw}
            for seed in seeds:
                pred=part/(('pred_%d.bin' if stage=='control' else 'diagnostic_pred_%d.bin')%seed)
                assert pred.stat().st_size==65536 and sha(pred)==checked[seed]['prediction_sha256']
            pair.append(dict(stage=stage,seeds=seeds,rows=checked,invoice_path=str(path),invoice_sha256=sha(path),guard_sha256=sha(guard_path)))
        plain,capture=pair;assert plain['seeds']==capture['seeds']
        for seed in plain['seeds']:
            a,b=plain['rows'][seed],capture['rows'][seed]
            assert a['total_flops']==b['total_flops'] and a['charged_map']==b['charged_map']
            assert a['prediction_sha256']==b['prediction_sha256']
            rows.append(dict(family=key,seed=seed,source_sha256=source,total_flops=a['total_flops'],plain_operation_sum=a['sum_operation_flops'],
                             capture_operation_sum=b['sum_operation_flops'],charged_maps_exact=True,prediction_hashes_exact=True,
                             plain_requests=a['requests'],capture_requests=b['requests'],diagnostic_request_difference=b['requests']-a['requests'],
                             plain_invoice_sha256=plain['invoice_sha256'],capture_invoice_sha256=capture['invoice_sha256'],
                             plain_guard_sha256=plain['guard_sha256'],capture_guard_sha256=capture['guard_sha256']))
    assert len(rows)==18 and len({(r['family'],r['seed']) for r in rows})==18
    result=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,
                ready_sha256=sha(ready_path),anchors_sha256=sha(anchors_path),script_sha256=sha(__file__),pairs=rows,
                all36_operation_sums_equal_own_total=True,all18_plain_capture_charged_maps_and_prediction_hashes_exact=True,
                diagnostic_zero_flop_requests_disclosed=True,no_target_arrays_or_predictions_executed=True,no_runtime_or_accuracy_claim=True)
    path=HERE/'INDEPENDENT_ANCHOR_INVOICE_SUM_v1.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(successful=True,pairs=len(rows),receipt_sha256=sha(path))),flush=True)
if __name__=='__main__':main()
