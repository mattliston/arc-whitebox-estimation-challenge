"""Private guard runner: original bankv2 with owned spawn/OWNER cleanup hardened."""
def run_native(home, protocol, kind, seed, env):
    global OWNED_RETIREMENT_CONFIRMED
    pins(protocol)
    # Pinned guard applies /usr/bin/nice -n10 once to its own native child.
    # Enforce the controller's base priority, avoiding accidental nice20.
    require(os.getpriority(os.PRIO_PROCESS,0)==0,'Controller must start at nice0; guard applies native nice10')
    before=resources()
    event(home/'resources.jsonl',dict(stage='before',kind=kind,seed=seed,**before))
    require(before['free_percent']>=50 and before['counted']<10, 'No free native slot or memory below50')
    stem=home/('native_%d_%s'%(seed,kind))
    result,log,guard=(Path(str(stem)+suffix) for suffix in ('.json','.log','_guard.json'))
    require(all(not p.exists() for p in (result,log,guard,Path(str(stem)+'_pred.npz'))), 'Fresh output required')
    candidate=protocol['candidates'][kind]
    env=dict(env,CANDIDATE_SHA256=candidate['sha256'])
    command=[str(PY),'-B',str(GUARD),'--record',str(guard),'--seconds','900',
             '--footprint-limit','7500000000','--rss-backstop','7500000000',
             str(RT),'-B',str(JUDGE),candidate['path'],str(seed),str(result)]
    started=time.monotonic()
    with log.open('x') as stream:
        child=None
        try:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=P2)
            OWNED_RETIREMENT_CONFIRMED = False
            owner=dict(controller_nice=0, guard_native_nice=10, guard_record=str(guard), utc=stamp(),kind=kind,seed=seed,controller_pid=os.getpid(),owned_guard_pid=child.pid,
                       command=command,candidate_sha256=candidate['sha256'],environment={k:env[k] for k in THREADS+('PB_ALL','CANDIDATE_SHA256')})
            event(home/'ownership.jsonl',owner)
            print(json.dumps(owner),flush=True)
            last_print=time.monotonic()
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current=resources()
                    event(home/'resources.jsonl',dict(stage='running',kind=kind,seed=seed,**current))
                    require(current['free_percent']>=50 and current['counted']<=10,'Live resource bound exceeded; retire only owned guard')
                    if time.monotonic()-last_print>=30:
                        print(json.dumps(dict(kind=kind,seed=seed,**current)),flush=True)
                        last_print=time.monotonic()
        except BaseException as original_error:
            if child is None:
                OWNED_RETIREMENT_CONFIRMED=True
                raise
            wait_error=None
            if child.poll() is None:
                try:child.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    wait_error=repr(error)
            try:
                stopped_guard=json.loads(guard.read_text())
                OWNED_RETIREMENT_CONFIRMED=(child.poll() is not None and stopped_guard.get('whole_group_retired') is True
                                            and stopped_guard.get('remaining_rows')==[])
            except (OSError,ValueError):
                OWNED_RETIREMENT_CONFIRMED=False
            exclusive(home/('interruption_%s_%d.json'%(kind,seed)),dict(
                utc=stamp(),owned_guard_pid=child.pid,returncode=child.poll(),wait_error=wait_error,
                original_error=repr(original_error),whole_group_retirement_verified=OWNED_RETIREMENT_CONFIRMED,
                slot_released=OWNED_RETIREMENT_CONFIRMED))
            if not OWNED_RETIREMENT_CONFIRMED:
                raise RuntimeError('Owned group retirement unconfirmed; retain slot reservation for root') from original_error
            raise
    elapsed=time.monotonic()-started
    g=json.loads(guard.read_text())
    OWNED_RETIREMENT_CONFIRMED=(child.poll() is not None and g.get('whole_group_retired') is True and g.get('remaining_rows')==[])
    require(child.returncode==0 and g.get('completed') is True and g.get('exit_code')==0
            and not g.get('error') and g.get('whole_group_retired') is True and g.get('remaining_rows')==[], 'Native guard incomplete or failed')
    require(g['max_footprint_bytes']<7500000000 and g['max_rss_bytes']<7500000000,'Guard footprint/RSS limit')
    pins(protocol)
    return dict(certificate=str(result),guard=str(guard),elapsed_s=elapsed,
                certificate_sha256=sha(result),prediction_npz_sha256=sha(result.with_name(result.stem+'_pred.npz')),
                guard_sha256=sha(guard),guard_max_rss_bytes=g['max_rss_bytes'],guard_max_footprint_bytes=g['max_footprint_bytes'])
