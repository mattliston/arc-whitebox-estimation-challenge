"""TAG opus-release: paths, limits and helpers shared by the release stages (see RUNBOOK.md). No numerical imports.

Nothing here uploads anything. Every stage writes only below O/release/runs/<tag>/."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

R = Path(__file__).resolve().parent            # O/release
O = R.parent
P2 = O.parent
ROOT = P2.parents[3]
H = P2/'codex_resume_20260926_v2'
SERVER_PY = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'   # OpenBLAS numpy + flopscope (server, native)
CLIENT_PY = P2/'data_gate/.client-venv/bin/python'                                   # numpy-free flopscope client
LIGHT_PY = ROOT/'.venv/bin/python'
WHEST = ROOT/'.venv/bin/whest'
# The machine guard. Since 2026-10-02 08:50Z the lane's own copy of H's lifecycle guard with the memory trigger on footprint and
# RSS as a 10 GB backstop (lead's decision 08:37Z; on this machine RSS counts freed pages that are only reclaimed under memory
# pressure: 6.0-6.8 GB in a quiet window against 2.9-3.9 GB loaded at the same 2.1-2.7 GB footprint). The original, which
# killed at 7.5 GB of group RSS: P2/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py'.
# Since 2026-10-12 08:20Z guard_footprint_v2.py (same limits; a failed ps/sysctl reading is retried, one reading a second): v1 retired
# every S51 worker on one ps call slower than 3 s at 08:06Z. The file before this line: _stale/common_pre_guard_v2_20261012T081938Z.py.
# Since 2026-10-12 23:47Z guard_footprint_v3.py (lead 23:4xZ: the swap rule needs memory_pressure free < 35% too, or 1 GiB; v2 retired every group
# on the box when sticky swap grew 269 MiB at 74% free).
GUARD = R/'guard_footprint_v3.py'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'fable_refit_v1/truth688'
RUNS = R/'runs'
BANK_S35 = H/'hill_climb_prefix128_fusion_v1/candidate.py'   # S35, submission #333111: the lane's first reference file
RRPI = H/'hill_climb_pair_rr_inverse_v1/candidate.py'        # research parent
VALID40 = [37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130, 132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163,
           169, 173, 175, 181, 192, 198, 199, 204, 219, 224, 227, 233, 234]
WORKER20 = [35, 100, 37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 76, 80, 84, 88, 92, 96, 104, 108]
# 40 networks nobody has trained or selected on: the first 40 of the trainer's 100 held-out newer ranks (O/trainer/cert/runB_val.txt
# as of 2026-10-01 21:50 UTC). The current parameters were tuned on the old 160 training ranks and SELECTED on valid40, so on
# valid40 any perturbation of the estimator reads slightly worse than it is (trainer, TEAM_LOG 21:50 UTC).
FRESH40 = [651, 378, 424, 328, 287, 641, 548, 557, 316, 390, 293, 277, 320, 312, 550, 672, 333, 310, 530, 259, 623, 459, 16, 473,
           474, 648, 633, 30, 1, 588, 673, 302, 303, 11, 324, 539, 579, 272, 480, 321]
# Limits each stage is held to (the ones used for S35).
LIMITS = dict(footprint_bytes=7_500_000_000, rss_backstop_bytes=10_000_000_000, swap_bytes=0, handles=70_000, setup_s=4.1,
              residual_s=0.4, worst_wall_s=105.0)
# What the platform measured for S35, #333111 (H/eval_333111.json, 100 networks, 5 workers).
PLATFORM_S35 = dict(sid=333111, predict_mean=82.469, predict_max=94.930, kernel_mean=73.412, overhead_mean=8.753,
                    residual_mean=0.3042, residual_max=0.3387, setup_max=4.290, flops_mean=263.955e9)
# THE BANK = the graded file every stage pairs a candidate with: stage C runs its legs beside the candidate's and scales ITS
# platform record, stage A takes its einsum signatures as platform-proven, stage B uses it as the native parent. It is the newest
# graded file (the nearest parent: the less a projection has to extrapolate, the better it is; three fires projected from S35
# read -1.1, +0.2 and -2.1 s in mean wall). calibration/bank.json is written by set_bank.py; without that file the bank is S35.
_BANK_FILE = R/'calibration'/'bank.json'
if _BANK_FILE.exists():
    _bank = json.loads(_BANK_FILE.read_text())
    BANK, PLATFORM, BANK_NAME = Path(_bank['source']), _bank['platform'], _bank['name']
else:
    BANK, PLATFORM, BANK_NAME = BANK_S35, PLATFORM_S35, 'S35'
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dir(tag):
    """A fresh output directory O/release/runs/<tag>; refuses to reuse one (never overwrite)."""
    assert Path(tag).name == tag, tag
    out = RUNS/tag
    out.mkdir(parents=True)
    return out


def guarded(record, seconds, command, log, env=None):
    """Run command under the machine guard (kills on group footprint >= 7.5 GB, group RSS >= 10 GB, +256 MiB system swap,
    or the time limit). Returns the guard's record (max_footprint_bytes, max_rss_bytes, max_swap_growth_bytes, completed,
    elapsed_s, ...)."""
    full = [str(LIGHT_PY), '-B', str(GUARD), '--record', str(record), '--seconds', str(seconds),
            '--footprint-limit', str(LIMITS['footprint_bytes']), '--rss-backstop', str(LIMITS['rss_backstop_bytes']),
            *map(str, command)]
    with open(log, 'x') as stream:
        subprocess.run(full, env=dict(os.environ, **THREADS, **(env or {})), stdout=stream, stderr=subprocess.STDOUT)
    return json.loads(Path(record).read_text())


LOAD_DEPENDENT = ('swap', 'setup')   # gates that only mean something when nobody else is using the machine
QUIET_LOAD = 3.0                     # 1-minute load average before launch at or below which the window counts as quiet


def peak_footprint(sessions):
    """Server + client peak physical footprint of worker sessions, in bytes: for each session the sum of the two processes'
    lifetime peaks as the kernel reports them (never sampled, compressed pages included; an upper bound of the simultaneous
    peak), then the largest over the sessions. This is the memory the platform's 8 GB limit is compared with."""
    return max(max(r['server_footprint_max'] for r in s['rows']) + max(r['client']['footprint_max'] for r in s['rows'])
               for s in sessions)


def verdict(stage, ok, text):
    line = f"[{stage}] {'PASS' if ok else 'FAIL'}: {text}"
    print(line, flush=True)
    return line


def judge(stage, gates, load_before, text):
    """PASS = every gate passed. FAIL = a gate that does not depend on machine load failed (or a load-dependent one failed in a
    quiet window). PASS* = only swap / setup-wall gates failed AND the window was loaded: those two are system-wide or wall-clock
    measurements that other lanes inflate, so they are reported as not judged and must be re-run in a quiet window before release."""
    failed = [k for k, v in gates.items() if not v]
    loaded = load_before > QUIET_LOAD
    hard = [k for k in failed if k not in LOAD_DEPENDENT or not loaded]
    status = 'FAIL' if hard else ('PASS*' if failed else 'PASS')
    note = f"; failed: {', '.join(hard)}" if hard else ''
    if failed and not hard:
        note = f"; NOT JUDGED (machine shared, load {load_before:.1f} before launch): {', '.join(failed)} -> re-run in a quiet window"
    line = f"[{stage}] {status}: {text}{note}"
    print(line, flush=True)
    return status, line


def write_json(path, value):
    with open(path, 'x') as stream:
        json.dump(value, stream, indent=1)
        stream.write('\n')


if __name__ == '__main__':
    for name in ('SERVER_PY', 'CLIENT_PY', 'LIGHT_PY', 'WHEST', 'GUARD', 'STORE', 'TRUTH', 'BANK', 'RRPI'):
        path = globals()[name]
        print(f"{name:10s} {'ok ' if Path(path).exists() else 'MISSING'} {path}")
    sys.exit(0)
