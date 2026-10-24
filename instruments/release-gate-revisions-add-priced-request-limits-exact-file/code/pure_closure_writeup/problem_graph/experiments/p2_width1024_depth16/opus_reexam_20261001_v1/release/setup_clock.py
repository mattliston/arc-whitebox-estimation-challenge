"""TAG opus-release: the estimator's setup under the PLATFORM's clock instead of this machine's (standard library only).

The estimator's setup works against its own wall guard (_PB_WALL_S = 4.0 s): it binds tree views most-used first until the guard
(phase 1), then fuses the bound keys until the same guard (phase 2), then runs an unguarded tail. How far it gets therefore
depends on the speed of the worker, and whatever it did not bind is bound inside predictions (billed, more requests, never fused).
On this shared machine the same setup gets through anything from 146 to all 649 keys depending on the load of the moment, which
made boundary sessions incomparable. On the platform it is reproducible: on four of S35's five workers the setup booked exactly the
flops of a complete binding phase (34,922,268,623) and on the fifth 1.6% less, all five with the same kernel and wire time to 2%
(#333111 setup_executions). So a normal platform worker gets through S35's binding phase and hardly more before the guard.

Setup work is proportional to requests (the client executes 337-357 thousand instructions per setup request whatever the key),
so platform time is modelled as requests x a fixed pace: install(mod, count, slowdown) replaces the candidate module's setup clock
(_pb_time) by  requests so far x GUARD_S / PLATFORM_REQUESTS x slowdown.
PLATFORM_REQUESTS = the requests a normal platform worker gets through in the 4.0 s of the guard = 19,000 (range 18,600-20,100).
Calibration (2026-10-01, RESULT.md section 6; requests and flops per key are deterministic, setup_keys.py):
(a) S35's binding phase takes 18,555 requests and its deferred fusion 9,481 more. Four of the five workers of #333111 booked
    the flops of a complete binding phase, so they reached at least 18,555. (The earlier banks S31-S34 and #333099 completed
    theirs on all 25 first setups, but their binding phases are shorter: 16,241 requests for the #333099 source.)
(b) The fifth (worker w-2 of #333111, 34,378,719,666 flops) had bound exactly 623 of 649 keys = 17,954 requests.
(c) The worker of #333099 that was set up again while slow booked 33,113,512,758 flops = exactly 580 of 649 keys of that source
    = 14,589 requests; its predictions then ran 1.30x (residual) to 1.34x (overhead, which includes its extra requests) slower
    than normal, so a normal worker gets through 14,589 x 1.22..1.34 = 17,800-19,550.
(d) The platform's own wire time (decode + encode per request) is 0.2923 s per S35 setup against 0.8587 s per S35 prediction;
    in instructions of that same server code a local prediction costs 7.17 G, a complete binding phase with the tail 2.28 G
    and the whole deferred fusion 0.96 G more, which puts the platform's setup at the binding phase plus 16% of the deferred
    fusion = 20,100 requests (this rests on wire time being proportional to wire instructions across request types, and the
    platform runs Python 3.10 against 3.11 here: +-7%).
(e) 2026-10-02, #333607 (S42: the first fired file whose setup builds the once-per-process constants, the engineer's CS, 493
    requests at the head of the binding phase): its five first setups booked the flops of exactly 623, 622, 632, 621 and 630
    of 649 keys = guard after 17,740 / 17,715 / 17,933 / 17,694 / 17,891 requests, in an hour of normal kernel speed (S40's),
    where S41's three complete setups of the same afternoon had reached at least 18,026 without CS. The constants are not
    view requests: they decode and upload the learned literals. Locally the phase takes 0.30-0.56 s for its 493 requests
    where binding requests take 0.29 ms each at the same load, i.e. the time of 1,030-1,930 binding requests. HEAVY charges
    that: from the log entry 'constants' on, the clock runs 1,200 requests ahead of the count, which puts a normal worker's
    guard at 17,800 requests of an S42-type setup (the platform's median: 17,740). Range of the charge by the two readings:
    500 to 1,450.
slowdown 1.0 = a normal platform worker; 1.06 = worker w-2 of #333111; 1.30 = the re-set-up worker of #333099; slowdown 0 = no
guard at all (everything is bound and fused: the calibration run).
Every entry the estimator appends to its setup log (_PB_LOG) is stamped with real seconds and requests so far.
The candidate file itself is not changed."""
import time

GUARD_S = 4.0
PLATFORM_REQUESTS = 19000
HEAVY = {'constants': 1200}      # setup log entry -> requests' worth of guard time its phase costs beyond its own request count


class Clock:
    def __init__(self, mod, count, slowdown, log):
        self.keep = getattr(mod, '_LVF_KEEP', None)      # [True] exactly while phase 1 (binding) runs
        self.count, self.pace, self.log = count, (GUARD_S/PLATFORM_REQUESTS*slowdown if slowdown else 0.0), log
        self.base, self.was_binding, self.binding_requests, self.trace = None, False, None, []
        self.extra = 0                                   # requests' worth of time charged by HEAVY phases (not requests)

    def monotonic(self):
        if self.base is None:
            self.base = self.count()                     # the setup's own first read (its t0)
        n = self.count() - self.base
        binding = bool(self.keep and self.keep[0])
        if self.was_binding and not binding and self.binding_requests is None:
            self.binding_requests = n                    # first clock read after phase 1: requests of the binding phase
            if self.log is not None:
                self.log.stamp('(binding phase ends)', None)
        if binding:
            self.trace.append(n)                         # one read per key of the list: requests before that key is bound
        self.was_binding = binding
        return (n + self.extra)*self.pace

    def __getattr__(self, name):                         # anything else a candidate asks of its clock module is the real one
        return getattr(time, name)


class Log(list):
    def __init__(self, items, count, mark):
        super().__init__(items)
        self.count, self.mark, self.stamps = count, mark, []

    def stamp(self, name, value):
        self.stamps.append([name, value, time.monotonic(), self.count(), self.mark() if self.mark else None])

    def append(self, item):
        self.stamp(item[0], item[1] if len(item) > 1 else None)
        clock = getattr(self, 'clock', None)
        if clock is not None and item[0] in HEAVY:
            clock.extra += HEAVY[item[0]]
        super().append(item)


def install(mod, count, slowdown=None, mark=None):
    """count() -> requests sent so far. slowdown None: the candidate keeps its own (wall) clock and only the log is stamped.
    mark() -> anything to store with each log entry (the setup probe passes the server's counters)."""
    log = clock = None
    if hasattr(mod, '_PB_LOG'):
        log = mod._PB_LOG = Log(mod._PB_LOG, count, mark)
    if slowdown is not None and hasattr(mod, '_pb_time'):
        clock = mod._pb_time = Clock(mod, count, slowdown, log)
        if log is not None:
            log.clock = clock
    return log, clock


def record(mod, log, clock, start, start_requests):
    """Setup facts for the manifest: phases [name, value, real seconds since start, requests since start], keys bound of the
    list, deferred keys fused, requests of the binding phase (paced runs), platform seconds at the end (paced runs)."""
    if log is None:
        return {}
    entries = {x[0]: x[1] for x in log}
    total = len(getattr(mod, '_PB_SINGLE', ()))
    out = dict(phases=[[s[0], s[1], s[2] - start, s[3] - start_requests] for s in log.stamps], keys_total=total,
               marks=[s[4] for s in log.stamps] if log.mark else None,
               failed=any(x[0] == 'restored after failure' for x in log),
               keys_bound=entries.get('wall guard', total) if 'lvfuse' in entries else None,
               deferred_fused=entries.get('lvfuse'))
    if clock is not None:
        out.update(binding_requests=clock.binding_requests, pace_s_per_request=clock.pace,
                   platform_requests=PLATFORM_REQUESTS, binding_trace=clock.trace)
    return out
