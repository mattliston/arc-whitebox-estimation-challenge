def prediction(path,seed):
    import numpy as np
    p=Path(path).with_name(Path(path).stem+'_pred.npz')
    with np.load(p,allow_pickle=False) as z:
        assert z.files==[str(seed)];y=z[str(seed)].copy()
    assert y.shape==(16,1024) and y.dtype==np.dtype('float32') and np.isfinite(y).all()
    return y

def setup_equal(a,b,parent=False):
    keys=('pb_single_sha256','pb_single_count','pb_wall_s','cache_counts')
    assert all(a['cert']['setup'][k]==b['cert']['setup'][k] for k in keys)

def admit(run,role,seed):
    # A transient lack of prospective headroom is not a failed model case.
    # Hold this root reservation; launch no child until admitted, or stop once.
    started=time.monotonic();attempt=0;last_status=None
    while True:
        state=guard.resources();elapsed=time.monotonic()-started;attempt+=1
        okay=state['free_percent']>=60 and state['counted']<8 and elapsed<=1800
        record=dict(role=role,seed=seed,controller_pid=os.getpid(),attempt=attempt,
                    elapsed_seconds=elapsed,admitted=okay,**state)
        guard.event(run/'admission_wait.jsonl',record)
        if okay:
            guard.event(run/'admission.jsonl',record)
            return record
        if last_status is None or elapsed-last_status>=30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child',
                  role=role,seed=seed,free_percent=state['free_percent'],counted=state['counted'],
                  elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
            last_status=elapsed
        assert elapsed<1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15,1800-elapsed))

def admissions(run):
    records=[json.loads(x)for x in (run/'admission.jsonl').read_text().splitlines()]
    assert [(x['role'],x['seed'])for x in records]==ORDER
    assert all(x['controller_pid']==os.getpid() and x['free_percent']>=60 and x['counted']<8 for x in records)
    attempts=[json.loads(x)for x in (run/'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']] == records
    assert all(x['admitted'] is True and 0<=x['elapsed_seconds']<=1800 for x in records)
    return dict(path=str(run/'admission.jsonl'),sha256=sha(run/'admission.jsonl'),records=records,wait_path=str(run/'admission_wait.jsonl'),wait_sha256=sha(run/'admission_wait.jsonl'))
