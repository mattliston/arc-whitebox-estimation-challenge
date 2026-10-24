"""Target-free verifier for the accuracy-frontier design package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    matrix = json.loads((HERE / "VARIANT_MATRIX.json").read_text())
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    children = json.loads((HERE / "GRAPH_CHILDREN.json").read_text())
    manifest = json.loads((HERE / "DESIGN_MANIFEST.json").read_text())

    assert matrix["status"].startswith("TARGET_FREE_DESIGN_ONLY")
    assert children["status"] == "PROPOSAL_ONLY__CANONICAL_GRAPH_UNCHANGED"
    assert children["parent_nodes"] == ["P2R0E3", "P2R0H"]
    assert [arm["id"] for arm in matrix["priority_variants"]] == [
        "A1_NO_CP4",
        "A2_THIRD_ONLY",
        "A3_CP3_ONLY",
        "A4_NO_L3SQ",
    ]

    arms = {arm["id"]: arm for arm in matrix["priority_variants"]}
    assert arms["A1_NO_CP4"]["exact_dominant_kernel_delta_flops"] == -233_997_074_432
    assert arms["A2_THIRD_ONLY"]["delta_partition"] == {
        "h4_gemms": -182_446_981_120,
        "q32_pair_einsums_four_moments_to_two": -97_860_116_480,
        "h1_gaussian_pair_einsums_three_to_one": -411_176_960,
    }
    assert arms["A2_THIRD_ONLY"]["exact_dominant_kernel_delta_flops"] == -280_718_274_560
    assert arms["A3_CP3_ONLY"]["delta_partition"] == {
        "h3_gemms": -122_346_799_104,
        "q32_pair_einsums_two_moments_to_one": -48_930_058_240,
        "h1_gaussian_pair_einsums_one_to_zero": -205_588_480,
    }
    assert arms["A3_CP3_ONLY"]["exact_dominant_kernel_delta_flops"] == -171_482_445_824
    assert arms["A4_NO_L3SQ"]["exact_q32_pair_einsum_delta_flops"] == -80_590_684_160

    diagnostics = {item["id"]: item for item in matrix["target_free_diagnostics"]}
    assert diagnostics["D_Q24_Q32"]["exact_pair_einsum_delta_flops"] == -48_082_370_560
    assert diagnostics["D_PSD_INTERVENTION"]["no_psd_off_arm"] is True
    assert diagnostics["D_PSD_INTERVENTION"]["spectral_eigh_plus_reconstruction_flops"] == 330_683_121_664

    graph_nodes = {item["id"]: item for item in children["proposed_nodes"]}
    assert {"P2R0E3A5", "P2R0E3A6", "P2R0E3A7", "P2R0H1"} <= graph_nodes.keys()
    assert graph_nodes["P2R0E3A5"]["kind"] == "phase2_actual_moment_feasibility_diagnostic"
    assert graph_nodes["P2R0H1"]["kind"] == "phase2_pure_closure_evaluation_and_promotion"
    assert all(parent != "P2R0E4" for parent, _ in children["proposed_edges"])

    immutable_checked = 0
    mutable_observed = 0
    for binding in bindings["bindings"]:
        path = (HERE / binding["path"]).resolve()
        assert path.is_file(), path
        if binding.get("mutable"):
            mutable_observed += 1
            continue
        assert _sha256(path) == binding["sha256"], path
        immutable_checked += 1

    assert not list(HERE.glob("estimator*.py"))
    assert bindings["negative_controls"] == {
        "targets_accessed": 0,
        "fits": 0,
        "samples": 0,
        "submissions": 0,
        "graph_files_modified": 0,
        "estimator_sources_created": 0,
    }
    for item in manifest["files"]:
        assert _sha256(HERE / item["path"]) == item["sha256"], item["path"]
    print(
        "PASS_TARGET_FREE_ACCURACY_FRONTIER_DESIGN "
        f"immutable_bindings={immutable_checked} mutable_observations={mutable_observed}"
    )


if __name__ == "__main__":
    main()
