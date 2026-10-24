"""Compare completed science32 runs as paired relative score ratios.

Keeps the legacy compare32 percentage-point delta for the existing fire rule.
Default reference is science32_rel_v4a_m15. No submission authority.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SEVEN = {35, 150, 200, 250, 259, 277, 287}


def grade(path):
    result = json.loads((path / "RESULT32_v1.json").read_text())
    assert result["all_groups_retired"] and result["all32"]["n"] == 32
    rows = json.loads((path / "candidate_all32/GRADE_STORE_v1.json").read_text())
    assert len(rows) == 32 and len({r["rank"] for r in rows}) == 32
    return result, {r["rank"]: r for r in rows}


def compare(candidate, reference):
    ar, a = grade(reference)
    br, b = grade(candidate)
    assert set(a) == set(b)
    assert ar["baseline_sha256"] == br["baseline_sha256"]
    values = []
    for rank in sorted(a):
        x, y = a[rank], b[rank]
        assert x["u0"] == y["u0"]
        row = {"rank": rank}
        for field in ("raw_pct", "bill_pct", "adjusted_pct"):
            row[field] = 100 * ((1 + y[field] / 100) / (1 + x[field] / 100) - 1)
        row["legacy_adjusted_delta_pp"] = y["adjusted_pct"] - x["adjusted_pct"]
        assert all(math.isfinite(v) for v in row.values())
        values.append(row)
    out = dict(reference=str(reference), candidate=str(candidate),
               candidate_sha256=br["candidate_sha256"], rows=values)
    for name, selected in (
        ("seven", [r for r in values if r["rank"] in SEVEN]),
        ("heldout25", [r for r in values if r["rank"] not in SEVEN]),
        ("all32", values),
    ):
        adj = [r["adjusted_pct"] for r in selected]
        out[name] = dict(n=len(adj), adjusted_pct=statistics.mean(adj),
                         adjusted_se=statistics.stdev(adj) / math.sqrt(len(adj)),
                         lower=sum(v < 0 for v in adj),
                         raw_pct=statistics.mean(r["raw_pct"] for r in selected),
                         bill_pct=statistics.mean(r["bill_pct"] for r in selected),
                         legacy_adjusted_delta_pp=statistics.mean(r["legacy_adjusted_delta_pp"] for r in selected))
    out["science_bar_cleared"] = (
        out["all32"]["adjusted_pct"] <= -0.35
        and out["all32"]["legacy_adjusted_delta_pp"] <= -0.35
        and out["heldout25"]["adjusted_pct"] < 0
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, default=HERE / "science32_rel_v4a_m15")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = compare(args.candidate.resolve(), args.reference.resolve())
    for name in ("seven", "heldout25", "all32"):
        r = out[name]
        print(f"{name}: adjusted {r['adjusted_pct']:+.4f}% (SE {r['adjusted_se']:.4f}, lower {r['lower']}/{r['n']}); "
              f"raw {r['raw_pct']:+.4f}%, bill {r['bill_pct']:+.4f}%; legacy {r['legacy_adjusted_delta_pp']:+.4f} pp")
    print("Science bar cleared:", out["science_bar_cleared"])
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n")
