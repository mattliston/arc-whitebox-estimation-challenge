"""Read-only archived geometry pricing; no models, arrays, targets or fitting.

Reproduces the old P0/Q first-transport pricing frame. This is NOT a
current-response candidate bill or a raw/adjusted score measurement.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "best_cube_archived_metadata_bound_v1.json"
INPUTS = (
    "pair_jet_opportunity_stage_a_v1.json",
    "pair_jet_coalescence_tiny_cost_v2.json",
    "direct_kernel_schedule_prices_v1.json",
)


def main():
    assert not OUT.exists(), "Versioned output already exists; do not overwrite."
    old, tiny, kernels = [json.loads((HERE / p).read_text()) for p in INPUTS]
    coeff = kernels["coefficients"]

    def mm(m, k, n):
        if not m or not k or not n:
            return 0.0
        assert all(v % 32 == 0 for v in (m, k, n))
        if n > 8192:
            return sum(mm(m, k, min(8192, n-j)) for j in range(0, n, 8192)) + m*n
        if k > 8192:
            return sum(mm(m, min(8192, k-j), n) for j in range(0, k, 8192)) + (len(range(0, k, 8192))-1)*m*n
        depths = (3, 4, 5) if max(m, k, n) <= 1024 else (4,)
        return min(2*m*k*n-m*n, *[
            sum(a*b for a, b in zip(coeff[str(d)], (m*k*n, m*k, k*n, m*n)))
            for d in depths
        ])

    # Exact rational checks for the conservative <0.23 ideal-algebra proof.
    assert Fraction(217, 1000)**3 > Fraction(1, 99)
    assert Fraction(884, 1000)**2 < Fraction(783, 1000)
    assert Fraction(994, 1000)**2 < Fraction(99, 100)
    assert Fraction(994, 1000)*Fraction(783, 1000)*Fraction(884, 1000)-Fraction(1, 10) > Fraction(588, 1000)
    assert Fraction(100, 441) < Fraction(23, 100)

    results = []
    for row in old["records"]:
        frame = next(x for x in tiny["cost"]["rows"] if x["rank"] == row["rank"])
        counts, active = frame["birth_counts"], frame["active"]
        assert counts == row["birth_counts"]

        def products(widths):
            total = 0.0
            for l in range(1, 16):
                inherited, newborn = sum(widths[:l-1]), counts[l-1]
                total += mm(1024, active[l-1], inherited) + mm(1024, active[l-1], newborn)
                if l < 15:
                    total += mm(active[l], inherited+2*newborn, active[l])
            return total

        original = products([2*n for n in counts])
        assert original == row["projection"]["parent_cp_gemm_projection"]
        assert original == frame["original_cp_gemm_projection"]
        bill = row["projection"]["parent_measured_first_context_flops"]
        for cutoff in (0.23, 0.24):
            caps = [0] * len(counts)
            widths = [2*n for n in counts]
            restoration = 0.0
            for site in row["sites"]:
                b, n = site["birth_layer"], site["original_pairs"]
                q = site["relative_tensor_error_quantiles"]
                # A deliberately loose maximum-only case admits all n pairs.
                cap = n
                if q[0] > cutoff:
                    cap = 0
                else:
                    for index, fraction in ((1, .25), (2, .5), (3, .75)):
                        if q[index] > cutoff:
                            cap = math.ceil(fraction*(n-1))
                            break
                caps[b] = cap
                widths[b] = 32*math.ceil((2*n-cap)/32)
                restoration += mm(site["source_active"], 64*math.ceil(cap/32), site["source_active"])
            reduced = products(widths)
            gross = original-reduced
            results.append(dict(
                rank=row["rank"], comparison_cutoff=cutoff,
                eligible_count_caps=caps, padded_cohort_widths=widths,
                original_cp_gemm_projection=original,
                reduced_cp_gemm_projection_at_caps=reduced,
                old_measured_first_context_flops=bill,
                gross_cp_saving_at_caps=gross,
                gross_fraction_of_old_bill=gross/bill,
                rank_2n_restoration_at_caps=restoration,
                saving_after_restoration_at_caps=gross-restoration,
                fraction_after_restoration_at_caps=(gross-restoration)/bill,
            ))
    receipt = dict(
        utc_finished=datetime.now(timezone.utc).isoformat(),
        ideal_bound="best relative Frobenius error <= 0.1 implies jet < 100/441 < 0.23",
        proof_constant_rational_checks=True,
        input_sha256={p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in INPUTS},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=results,
        qualification=(
            "Scalar archived-metadata calculation only. No model/array/target access, "
            "capture, candidate, fit, raw or adjusted score. Old P0/Q active and birth "
            "geometry is NOT current retained P-all16 response geometry. Cutoffs include "
            "a comparison margin but are not a certified bound on the old FP64 reductions. "
            "Restoration-at-caps is a priced scenario, not an optimization over every "
            "subset. Root solving, Gram acquisition, assembly, copies, FP32 repair and "
            "feedback are unpaid. No closed family or criterion is changed."
        ),
    )
    with OUT.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
