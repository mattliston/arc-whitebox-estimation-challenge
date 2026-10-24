"""Read-only, stdlib-only20s metadata diagnosis; never a qualification PASS."""
import faulthandler
import json
import resource
import signal
import sys
import time
from types import SimpleNamespace
import panel_common as c
import panel_admit as a


def main():
    began = time.monotonic()
    c.environment()
    def guard(*unused):
        if time.monotonic()-began >= 20 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= 1_000_000_000:
            raise RuntimeError('Read-only metadata diagnosis20s/1GB cap')
    signal.signal(signal.SIGALRM,guard)
    signal.setitimer(signal.ITIMER_REAL,.1,.1)
    faulthandler.dump_traceback_later(4,repeat=True)
    values = dict(watcher_sha256='56793bf236fb43ef363abbff7cf8d323a828715947a625ec01b4ecd1c469335f',
        worker_sha256='9f6dd0de56e3af8fc2c8c3be400963148b9fde1a414b7f888a603bb988a501b3',
        admitter_sha256='ac38f0c27d61289c53b1278309ef99b5e9d126fb7c03f06188354fdad015b83f',
        helper_sha256='7d9c8b2d97ac5bbb85c077da694472a52fbede2946ec2643df44e1018a6185b7',
        protocol_sha256=c.PROTOCOL_SHA,
        watch_spec_sha256='c9c3df03121c7b559784a2d3114c48712babcf2805f5922a928be6dbbc144266',
        score40_sha256=c.SCORE_SHA,score40_watch_sha256=c.SCORE_WATCH_SHA)
    for name in ('qualified_native40','qualified_comp'):
        original = getattr(a,name)
        def wrapper(*args,_original=original,_name=name,**kwargs):
            print('ENTER',_name,time.monotonic()-began,flush=True)
            out = _original(*args,**kwargs)
            print('EXIT',_name,time.monotonic()-began,flush=True)
            return out
        setattr(a,name,wrapper)
    try:
        a.full_admission(SimpleNamespace(**values),guard)
        print('METADATA_RETURNED_NOT_QUALIFIED',flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        signal.setitimer(signal.ITIMER_REAL,0)
        print(json.dumps(dict(diagnostic_only=True,elapsed_s=time.monotonic()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            numerical_imports=sorted(set(sys.modules)&{'numpy','torch','flopscope','threadpoolctl'}),
            no_stage_writes=True,no_candidate_execution=True)),flush=True)


if __name__ == '__main__':
    main()
