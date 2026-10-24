"""Rank35-only observation adapter; inert until --execute-reviewed-capture.

Exports the exact frozen v2 capture source, then adds observation/provenance
checks only. No inverse, predictor, loss, fit, Torch or MC is implemented.
Original pre13 K21 is read only for unchanged-bank byte parity and is never
written into the new causal input artifact. Root reviews and executes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "rows_reflected_residual_copula_tree_v1" / "row_0035"
OLD = HERE / "rows_two_memory_teacher_action_v2" / "row_0035"
NOTE = "REFLECTED_RESIDUAL_COPULA_TRACE_CAPTURE_OWNERSHIP_V1.md"
PROTOCOL = "REFLECTED_RESIDUAL_COPULA_TREE_PROTOCOL_V1.md"
QUALIFICATION = "two_memory_capture_replay_v2_qualification.json"
PINS = {
    "capture_two_memory_teacher_action_v1.py":
        "4eb6a294651800c137614cbc1ad8a1bab99a206ad220d76e86f74b184056d608",
    "capture_two_memory_teacher_action_replay_v2.py":
        "82103b513f517cd555803260dd6f2c54aeeeb07a3a9e55254b12162baf646d7e",
    "TWO_MEMORY_RESOURCE_REPLAY_V2.md":
        "f32c9e8097d1845b4882930f9d45ae8aaab89499d9f4493f964123a763a05d4c",
    "qualify_two_memory_capture_replay_v2.py":
        "3b5e25198d342e760aea5827b5100e99630bdbdb89578f1adc05945b21597247",
    QUALIFICATION:
        "af8c2b0cb66b6730af9d93b5aa0e56efcc97ba3e825093c4926f4bbd3805f328",
    PROTOCOL:
        "a0c4fa413bcd607a7df2f748c86c359489b88c84672ce97314d90dec448c0c50",
    "REFLECTED_RESIDUAL_COPULA_ROOT_ACCEPTANCE_V1.md":
        "a0c25bea1700f484fcab872f2c79c4f4ec2d9cbcdea9bc8d9960d4d2b365e8f2",
}
CAUSAL_KEYS = (
    "post_C", "post_B", "post_b", "post_m", "post_variance", "post_K21",
    "post_alive", "post_radial", "pre13_Bpre", "pre13_bpre", "pre13_alive",
    "pre13_alpha", "pre13_sigma",
)
THREAD_ENV_KEYS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replace_once(source, old, new):
    require(source.count(old) == 1, f"Frozen observation anchor changed: {old!r}")
    return source.replace(old, new)


class TraceObservation:
    def __init__(self):
        self.checked = {}
        self.post_checks = []
        self.pre13_checks = []
        self.final_checks = []
        self.causal = {}
        self.causal_saved = False
        self.started = time.perf_counter()
        self.original_v2_generated_capture_sha256 = None
        self.extended_capture_sha256 = None

    def check_file(self, path, expected=None):
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"Authenticated file changed: {path.name}")
        self.checked[str(path.relative_to(HERE))] = digest
        return digest

    def sources(self, source_hashes):
        for name, expected in PINS.items():
            require(source_hashes[name] == expected, f"Frozen source/receipt mismatch: {name}")
        for name in (Path(__file__).name, NOTE):
            require(name in source_hashes, f"Unbound observation source: {name}")
        qualification = json.loads((HERE / QUALIFICATION).read_text())
        require(qualification["passed"] is True and qualification["ranks"] == [35, 36, 37],
                "The fixed three-rank v2 replay did not qualify")
        rows = [row for row in qualification["rows"] if row["rank"] == 35]
        require(len(rows) == 1, "Ambiguous rank35 qualification")
        self.check_file(OLD / "result.json", rows[0]["v2_receipt_sha256"])
        receipt = json.loads((OLD / "result.json").read_text())
        require(receipt["passed"] is True and receipt["rank"] == 35 and
                receipt["target_access"] is False and receipt["peak_rss_bytes"] <= 6*(1 << 30),
                "The original rank35 v2 row is not qualified")
        for name, digest in receipt["source_hashes"].items():
            self.check_file(HERE / name, digest)
        for name, record in receipt["artifacts"].items():
            path = OLD / name
            key = str(path.relative_to(HERE))
            require(qualification["checked_files"][key] == record["sha256"],
                    f"Old v2 artifact is outside the authenticated replay: {name}")
            self.check_file(path, record["sha256"])
            require(path.stat().st_size == record["bytes"], f"Old artifact size changed: {name}")
        require("pre_13.npz" in receipt["artifacts"] and "post_12.npz" in receipt["artifacts"],
                "Missing authenticated old causal/action artifact")
        self.old_receipt = receipt
        self.old_receipt_sha256 = sha(OLD / "result.json")
        self.old_peak_rss_bytes = receipt["peak_rss_bytes"]
        self.initial_source_hashes = dict(source_hashes)

    def inputs(self, input_hashes):
        require(input_hashes == self.old_receipt["input_hashes"],
                "Weight/prefix/final inputs differ from the qualified v2 row")
        self.initial_input_hashes = dict(input_hashes)

    @staticmethod
    def byte_equal(np, new, old, label):
        require(new.shape == old.shape and new.dtype == old.dtype,
                f"Shape or dtype changed: {label}")
        require(np.isfinite(new).all() and np.isfinite(old).all(),
                f"Nonfinite parity input: {label}")
        require(new.tobytes() == old.tobytes(), f"Bank byte parity failed: {label}")

    def post(self, np, layer, m, C, K21, b, Bpost, alive, radial):
        require(layer == len(self.post_checks), "Post observation order changed")
        variance = np.diag(C).copy()
        fields = dict(layer=np.array(layer), m=m, variance=variance,
                      physical_K21=K21, alive=alive, radial=np.array(radial))
        with np.load(OLD / f"post_{layer:02d}.npz", allow_pickle=False) as old:
            require(set(old.files) == set(fields), "Old post schema changed")
            for key, value in fields.items():
                reference = old[key]
                if key == "physical_K21":
                    require(value.shape == reference.shape and value.dtype == reference.dtype,
                            "Post K21 shape/dtype changed")
                    require(np.isfinite(value).all() and np.isfinite(reference).all(),
                            "Nonfinite normalized post K21 comparison")
                    relative = float(np.linalg.norm(value-reference) /
                                     max(1., float(np.linalg.norm(reference))))
                    require(relative <= 1e-10, "Normalized post K21 parity failed")
                else:
                    self.byte_equal(np, value, reference, f"post{layer}/{key}")
        Bcheck = C.copy()
        np.fill_diagonal(Bcheck, variance*alive)
        self.byte_equal(np, Bpost, Bcheck, f"post{layer}/actual_Bpost_ownership")
        require(np.count_nonzero(b[~alive]) == 0, "Trace-vector cold support changed")
        self.post_checks.append(dict(layer=layer, other_fields_byte_equal=True,
                                     physical_K21_normalized_l2=relative,
                                     actual_Bpost_byte_equal_expected_bank_metric=True))
        if layer == 12:
            require(not self.causal, "Duplicate post12 capture")
            self.causal = dict(post_C=C.copy(), post_B=Bpost.copy(), post_b=b.copy(),
                post_m=m.copy(), post_variance=variance, post_K21=K21.copy(),
                post_alive=alive.copy(), post_radial=np.array(radial))
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 6*(1 << 30),
                "Fixed6GiB observation RSS qualification exceeded")

    def pre(self, layer, Bpre, bpre, alive):
        if layer != 13:
            return
        require(set(self.causal) == set(CAUSAL_KEYS[:8]), "Missing causal post12 state")
        self.causal.update(pre13_Bpre=Bpre.copy(), pre13_bpre=bpre.copy(),
                           pre13_alive=alive.copy())

    def save_causal(self, np, aux13, save_npz):
        require(not self.causal_saved, "Duplicate causal artifact")
        # This is equality qualification only. No new predictor or loss exists
        # in this script. The old action array is never added to causal fields.
        with np.load(OLD / "pre_13.npz", allow_pickle=False) as old:
            require(set(old.files) == set(aux13), "Old pre13 parity schema changed")
            for key, value in aux13.items():
                self.byte_equal(np, value, old[key], f"pre13/{key}")
                self.pre13_checks.append(dict(array=key, byte_equal=True))
        self.byte_equal(np, self.causal["pre13_alive"], aux13["pre_alive"],
                        "pre13/trace_vs_host_alive")
        self.causal.update(pre13_alpha=aux13["aux_alpha"].copy(),
                           pre13_sigma=aux13["aux_sigma"].copy())
        require(set(self.causal) == set(CAUSAL_KEYS), "Causal contract changed")
        require("pre_K21_loss_only" not in self.causal, "Current-action loss leaked into inputs")
        save_npz("causal_post12_pre13.npz", **self.causal)
        self.causal_saved = True
        self.causal.clear()

    def finish(self, np, out):
        require(self.causal_saved and not self.causal and len(self.post_checks) == 13,
                "Incomplete causal observation")
        require(not (out / "pre_13.npz").exists(), "Do not duplicate sealed current-action loss")
        for name in ("pre_vectors_00_14.npz", "outputs_and_final_Q.npz"):
            with np.load(OLD / name, allow_pickle=False) as old, \
                    np.load(out / name, allow_pickle=False) as new:
                require(set(old.files) == set(new.files), f"Final parity schema changed: {name}")
                for key in old.files:
                    self.byte_equal(np, new[key], old[key], f"{name}/{key}")
                    self.final_checks.append(dict(artifact=name, array=key, byte_equal=True))
        for name, digest in self.checked.items():
            require(sha(HERE / name) == digest, f"Authenticated old input changed during capture: {name}")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 6*(1 << 30),
                "Fixed6GiB final observation qualification exceeded")

    def report(self):
        return dict(causal_artifact="causal_post12_pre13.npz", causal_keys=list(CAUSAL_KEYS),
            action_loss_path="rows_two_memory_teacher_action_v2/row_0035/pre_13.npz",
            action_loss_key="pre_K21_loss_only", current_action_loss_in_causal_artifact=False,
            old_action_array_read_for_bank_byte_parity_only=True,
            predictor_evaluated=False, inverse_evaluated=False, current_action_loss_evaluated=False,
            captured_actual_Bpost=True, captured_actual_Bpre_bpre=True,
            trace_covariance_symmetrized=False, covariance_repaired=False,
            old_v2_receipt_sha256=self.old_receipt_sha256,
            old_v2_peak_rss_bytes=self.old_peak_rss_bytes, checked_files=self.checked,
            post_checks=self.post_checks, pre13_byte_parity=self.pre13_checks,
            final_byte_parity=self.final_checks,
            original_v2_generated_capture_sha256=self.original_v2_generated_capture_sha256,
            extended_capture_sha256=self.extended_capture_sha256,
            requested_thread_environment={k: os.environ.get(k) for k in THREAD_ENV_KEYS},
            requested_blas_threads=2, actual_effective_thread_count_not_assumed=True,
            adapter_wall_seconds=time.perf_counter()-self.started)


def export_frozen_v2_source():
    wrapper = HERE / "capture_two_memory_teacher_action_replay_v2.py"
    require(sha(wrapper) == PINS[wrapper.name], "Frozen v2 wrapper changed")
    original = wrapper.read_text()
    footer = (
        "    namespace = dict(__file__=str(Path(__file__).resolve()),\n"
        "                     __name__='two_memory_resource_replay_v2_frozen_parent')\n"
        "    exec(compile(source, str(PARENT)+'[resource-replay-v2]', 'exec'), namespace)\n"
        "    namespace['main']()\n"
    )
    exported = replace_once(original, footer, "    return source\n")
    require(exported.replace("    return source\n", footer, 1) == original,
            "The v2 source-export adapter is not reversible")
    namespace = dict(__file__=str(wrapper), __name__="frozen_v2_capture_source_export")
    exec(compile(exported, str(wrapper)+"[source-export-only]", "exec"), namespace)
    return namespace["main"]()


def extend_observation_source(source):
    source = replace_once(source, '"rows_two_memory_teacher_action_v2"',
                          '"rows_reflected_residual_copula_tree_v1"')
    source = replace_once(source,
        "    # Numerical imports and every array access are behind --execute.\n",
        "    require(args.rank == 35, 'This observation capture is rank35 only')\n"
        "    # Numerical imports and every array access are behind the reviewed flag.\n")
    source = replace_once(source,
        "    source_hashes = {name: sha(HERE / name) for name in source_names}",
        "    source_names = list(dict.fromkeys(source_names + _TRACE_SOURCE_NAMES))\n"
        "    source_hashes = {name: sha(HERE / name) for name in source_names}")
    source = replace_once(source,
        "    if args.rank != 35:\n",
        "    _trace_observer.sources(source_hashes)\n"
        "    if args.rank != 35:\n")
    source = replace_once(source,
        "    input_hashes = {name: sha(path) for name, path in input_paths.items()}",
        "    input_hashes = {name: sha(path) for name, path in input_paths.items()}\n"
        "    _trace_observer.inputs(input_hashes)")
    source = replace_once(source,
        "    def capture_pre(l, previous_m, previous_C, mu, Cpre, K21, alive, radial):",
        "    def capture_pre(l, previous_m, previous_C, mu, Cpre, K21, alive, radial, Bpre, bpre):")
    source = replace_once(source,
        "        if l == 13:\n            # Keep the exact bank read.",
        "        _trace_observer.pre(l, Bpre, bpre, alive)\n"
        "        if l == 13:\n            # Keep the exact bank read.")
    source = replace_once(source,
        "    def capture_post(l, m, C, p, lam, star, b, alive, radial, map_target):",
        "    def capture_post(l, m, C, p, lam, star, b, alive, radial, map_target, Bpost):")
    source = replace_once(source,
        "        physical_K21 = cp + decoded_star + trace",
        "        physical_K21 = cp + decoded_star + trace\n"
        "        _trace_observer.post(np, l, m, C, physical_K21, b, Bpost, alive, radial)")
    source = replace_once(source,
        '            save_npz("pre_13.npz", **aux13)',
        "            _trace_observer.save_causal(np, aux13, save_npz)")
    source = replace_once(source,
        "_capture_pre(l, m, C, mu, Cpre, K21, alive, radial)",
        "_capture_pre(l, m, C, mu, Cpre, K21, alive, radial, Bpre, bpre)")
    source = replace_once(source,
        "_capture_post(l, m, C, p, lam, star, b, alive, radial_new, target)",
        "_capture_post(l, m, C, p, lam, star, b, alive, radial_new, target, Bpost)")
    source = replace_once(source,
        '                 prediction_served_float32=final_saved["prediction"].astype(np.float32))\n',
        '                 prediction_served_float32=final_saved["prediction"].astype(np.float32))\n'
        "        _trace_observer.finish(np, out)\n")
    source = replace_once(source,
        '              interpretation="Target-free actual-bank action data;',
        '              trace_ownership=_trace_observer.report(),\n'
        '              interpretation="Target-free actual-bank action data;')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=35, choices=(35,))
    parser.add_argument("--execute-reviewed-capture", action="store_true")
    args = parser.parse_args()
    if not args.execute_reviewed_capture:
        print(json.dumps(dict(execute=False, rank=35, output=str(OUT),
            causal_artifact="causal_post12_pre13.npz", causal_keys=list(CAUSAL_KEYS),
            arrays_opened=False, numeric_imports=False, current_action_loss_evaluated=False)))
        return
    require(not OUT.exists(), "Preserve existing observation output; no overwrite/replay")
    observer = TraceObservation()
    try:
        require("/anaconda3/" not in sys.executable, "Use NumPy-only repository .venv Python")
        require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
                "Torch must not be loaded")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch nice >=10")
        for key in THREAD_ENV_KEYS:
            os.environ[key] = "2"
        for name, expected in PINS.items():
            observer.check_file(HERE / name, expected)
        observer.check_file(Path(__file__).resolve())
        observer.check_file(HERE / NOTE)
        original = export_frozen_v2_source()
        source = extend_observation_source(original)
        observer.original_v2_generated_capture_sha256 = hashlib.sha256(original.encode()).hexdigest()
        observer.extended_capture_sha256 = hashlib.sha256(source.encode()).hexdigest()
        namespace = dict(__file__=str(Path(__file__).resolve()),
            __name__="reflected_residual_copula_trace_observation_v1",
            _trace_observer=observer, _TRACE_SOURCE_NAMES=[*PINS, NOTE])
        exec(compile(source, str(Path(__file__).resolve())+"[frozen-v2-observation-adapter]", "exec"),
             namespace)
        namespace["execute"](args)
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = OUT / "trace_observation_failure_v1.json"
        if not failure.exists():
            write_json(failure, dict(passed=False, rank=35,
                exception_type=type(exc).__name__, message=str(exc),
                arrays_used_only_for_bank_capture_and_parity=True,
                inverse_evaluated=False, predictor_evaluated=False,
                current_action_loss_evaluated=False, checked_files=observer.checked,
                source_sha256=sha(Path(__file__).resolve()),
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                wall_seconds=time.perf_counter()-observer.started))
        raise


if __name__ == "__main__":
    main()
