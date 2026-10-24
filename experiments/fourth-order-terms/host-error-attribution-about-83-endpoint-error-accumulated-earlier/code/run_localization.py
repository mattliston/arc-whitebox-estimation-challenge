#!/Users/matt/anaconda3/bin/python3
"""P2R0F2S read-only host-error localization: frozen metrics M1-M6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
OUTPUT = HERE / "LOCALIZATION_RESULT.json"
DEV_ERROR = P2 / "public_full688_dynamic_multistate_residual_cv_v1/cache/patched_error.npy"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
AUG_ROWS = P2 / "public_aug7_source_observable_tape_v1/rows/observer_fit"
NATIVE_ROWS = P2 / "public_q24_native_transport_energy_pilot_v1/runs/native_rows_v1"
R_ROWS = P2 / "public_native_r10_source_uncertainty_pilot_v1/rows"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"

DEPTH = 16
WIDTH = 1024
DEV_CHANNELS = 7
NATIVE_CHANNELS = 10
HARD_ROWS = [201, 310, 599, 643]
DEV_CHANNEL_NAMES = [
    "discarded_k3_signed", "discarded_k3_absolute", "cov_series_order10_row_share",
    "max_offdiag_abs_correlation", "coincident_bond_degree", "rho2_neighbor_skew",
    "rho2_neighbor_alpha",
]
NATIVE_CHANNEL_NAMES = [
    "cp4_dropped_diagonal", "cp4_dropped_diagonal_absolute", "cubic_star_row_rms",
    "quartic_star_row_rms", "quartic_star_diagonal", "pairing_alpha",
    "mehler_t24_row_coupling", "max_offdiagonal_abs_rho", "rho2_neighbor_alpha",
    "rho8_row_concentration",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CorrAccumulator:
    """Streaming Pearson accumulator over (layer, channel) cells."""

    def __init__(self, layers: int, channels: int) -> None:
        shape = (layers, channels)
        self.n = np.zeros(shape)
        self.sx = np.zeros(shape)
        self.sy = np.zeros(shape)
        self.sxx = np.zeros(shape)
        self.syy = np.zeros(shape)
        self.sxy = np.zeros(shape)

    def add(self, layer: int, x: np.ndarray, y: np.ndarray) -> None:
        # x: [width, channels]; y: [width]
        self.n[layer] += x.shape[0]
        self.sx[layer] += x.sum(axis=0)
        self.sxx[layer] += np.square(x).sum(axis=0)
        self.sy[layer] += y.sum()
        self.syy[layer] += np.square(y).sum()
        self.sxy[layer] += (x * y[:, None]).sum(axis=0)

    def corr(self) -> np.ndarray:
        n = np.maximum(self.n, 1.0)
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = np.maximum(self.sxx / n - np.square(self.sx / n), 1e-300)
        vy = np.maximum(self.syy / n - np.square(self.sy / n), 1e-300)
        return cov / np.sqrt(vx * vy)


def frame_analysis(
    label: str,
    ranks: list[int],
    load_error_row,
    load_phi_row,
    load_weights_row,
    load_channels_row,
    channel_count: int,
    channel_names: list[str],
) -> dict[str, object]:
    rows = len(ranks)
    layer_sse = np.zeros(DEPTH)
    unit_sse = np.zeros((DEPTH, WIDTH))
    half_unit_sse = np.zeros((2, DEPTH, WIDTH))
    proj_num = np.zeros(DEPTH)
    proj_den = np.zeros(DEPTH)
    cache_p: dict[int, np.ndarray] = {}
    row_final = np.zeros(rows)
    row_channel_mean = np.zeros((rows, channel_count))
    corr_local = CorrAccumulator(DEPTH, channel_count)

    # Pass 1: profiles, projection scalars, row summaries.
    for index, rank in enumerate(ranks):
        error = load_error_row(rank).astype(np.float64)
        phi = load_phi_row(rank).astype(np.float64)
        weights = load_weights_row(rank)
        channels = np.abs(load_channels_row(rank).astype(np.float64))
        layer_sse += np.mean(np.square(error), axis=1)
        unit_sse[:] += np.square(error)
        half_unit_sse[index % 2] += np.square(error)
        row_final[index] = float(np.mean(np.square(error[-1])))
        row_channel_mean[index] = channels.mean(axis=(0, 1))
        for layer in range(1, DEPTH):
            w = np.asarray(weights[layer], dtype=np.float64)
            p = phi[layer] * (error[layer - 1] @ w)
            proj_num[layer] += float(np.dot(error[layer], p))
            proj_den[layer] += float(np.dot(p, p))
            corr_local.add(layer, channels[layer], np.square(error[layer]))

    a = np.where(proj_den > 0, proj_num / np.maximum(proj_den, 1e-300), 0.0)

    # Pass 2: born error with the fitted global scalars.
    born_sse = np.zeros(DEPTH)
    corr_born = CorrAccumulator(DEPTH, channel_count)
    row_born_total = np.zeros(rows)
    for index, rank in enumerate(ranks):
        error = load_error_row(rank).astype(np.float64)
        phi = load_phi_row(rank).astype(np.float64)
        weights = load_weights_row(rank)
        channels = np.abs(load_channels_row(rank).astype(np.float64))
        for layer in range(1, DEPTH):
            w = np.asarray(weights[layer], dtype=np.float64)
            p = phi[layer] * (error[layer - 1] @ w)
            born = error[layer] - a[layer] * p
            born_sq = np.square(born)
            born_sse[layer] += float(np.mean(born_sq))
            row_born_total[index] += float(np.mean(born_sq))
            corr_born.add(layer, channels[layer], born_sq)

    layer_mse = layer_sse / rows
    born_mse = born_sse / rows
    increments = np.diff(layer_mse, prepend=0.0)
    unit_mse = unit_sse / rows
    top_shares = {}
    persistence = {}
    for layer in (7, 11, 15):
        values = np.sort(unit_mse[layer])[::-1]
        total = float(values.sum())
        top_shares[str(layer)] = {
            "top_1pct": float(values[: WIDTH // 100].sum() / total),
            "top_10pct": float(values[: WIDTH // 10].sum() / total),
        }
        h0 = half_unit_sse[0, layer]
        h1 = half_unit_sse[1, layer]
        persistence[str(layer)] = float(np.corrcoef(h0, h1)[0, 1])

    def row_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    pooled_local = corr_local.corr()[2:16].mean(axis=0)
    pooled_born = corr_born.corr()[2:16].mean(axis=0)
    sign_consistency_born = (
        np.sign(corr_born.corr()[2:16]) == np.sign(pooled_born)[None, :]
    ).sum(axis=0)

    return {
        "rows": rows,
        "m1_layer_mse": layer_mse.tolist(),
        "m1_layer_increment": increments.tolist(),
        "m1_final_mse": float(layer_mse[-1]),
        "m2_top_unit_shares": top_shares,
        "m2_cross_half_persistence": persistence,
        "m3_propagation_scalar_a": a.tolist(),
        "m3_born_mse": born_mse.tolist(),
        "m3_born_fraction": (born_mse[1:] / np.maximum(layer_mse[1:], 1e-300)).tolist(),
        "m4_local_error_correlation_pooled": {
            name: float(pooled_local[i]) for i, name in enumerate(channel_names)
        },
        "m4_born_error_correlation_pooled": {
            name: float(pooled_born[i]) for i, name in enumerate(channel_names)
        },
        "m4_born_correlation_by_layer": {
            name: corr_born.corr()[2:16, i].tolist()
            for i, name in enumerate(channel_names)
        },
        "m4_born_sign_consistency_of_14": {
            name: int(sign_consistency_born[i]) for i, name in enumerate(channel_names)
        },
        "m5_row_final_vs_channel_mean": {
            name: row_corr(row_final, row_channel_mean[:, i])
            for i, name in enumerate(channel_names)
        },
        "m5_row_final_vs_born_total": row_corr(row_final, row_born_total),
        "label": label,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prereg_sha = sha256(PREREGISTRATION)

    dev_error = np.load(DEV_ERROR, mmap_mode="r", allow_pickle=False)

    def dev_err(rank):
        return np.asarray(dev_error[rank])

    def dev_phi(rank):
        raw = np.load(
            FEATURE_STORE / f"rows/observer_fit/{rank:04d}/cp1792_raw_locals.npy",
            mmap_mode="r", allow_pickle=False,
        )
        return np.asarray(raw[:, :, 5])

    def dev_weights(rank):
        return np.load(
            FEATURE_STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
            mmap_mode="r", allow_pickle=False,
        )

    def dev_channels(rank):
        return np.asarray(
            np.load(AUG_ROWS / f"{rank:04d}/aug7_locals.npy", mmap_mode="r", allow_pickle=False)
        )

    development = frame_analysis(
        "development_688",
        list(range(688)),
        dev_err, dev_phi, dev_weights, dev_channels,
        DEV_CHANNELS, DEV_CHANNEL_NAMES,
    )
    hard = {
        str(rank): {
            "final_mse": float(np.mean(np.square(np.asarray(dev_error[rank][-1], dtype=np.float64)))),
            "layer_mse": np.mean(np.square(np.asarray(dev_error[rank], dtype=np.float64)), axis=1).tolist(),
            "channel_row_mean_abs": np.abs(
                np.asarray(dev_channels(rank), dtype=np.float64)
            ).mean(axis=(0, 1)).tolist(),
        }
        for rank in HARD_ROWS
    }
    population_channel_median = np.median(
        np.stack([
            np.abs(np.asarray(dev_channels(rank), dtype=np.float64)).mean(axis=(0, 1))
            for rank in range(0, 688, 8)
        ]), axis=0,
    ).tolist()

    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, DEPTH, WIDTH)

    def native_err(rank):
        host = np.load(NATIVE_ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False)
        return np.asarray(targets[rank], dtype=np.float64) - np.asarray(host, dtype=np.float64)

    def native_phi(rank):
        raw = np.load(
            NATIVE_ROWS / f"row_{rank:04d}/raw20.npy", mmap_mode="r", allow_pickle=False
        )
        return np.asarray(raw[:, :, 5])

    def native_channels(rank):
        return np.asarray(
            np.load(R_ROWS / f"row_{rank:04d}/r10_locals.npy", mmap_mode="r", allow_pickle=False)
        )

    native = frame_analysis(
        "native_200",
        list(range(35, 235)),
        native_err, native_phi, dev_weights, native_channels,
        NATIVE_CHANNELS, NATIVE_CHANNEL_NAMES,
    )

    payload = {
        "schema": "public-host-error-localization-result-v1",
        "node": "P2R0F2S",
        "status": "COMPLETE_READ_ONLY",
        "preregistration_sha256": prereg_sha,
        "program_sha256": sha256(Path(__file__)),
        "bindings": {
            "dev_error_sha256": sha256(DEV_ERROR),
            "targets_sha256": sha256(TARGETS),
        },
        "development": development,
        "development_hard_rows": {
            "rows": hard,
            "population_channel_median_row_mean_abs": population_channel_median,
        },
        "native": native,
        "firewalls": {
            "observer_selection_targets_opened": 0,
            "public_name_joins": 0,
            "network_access": 0,
            "host_source_modifications": 0,
            "models_trained": 0,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(json.dumps({
        "dev_final": development["m1_final_mse"],
        "native_final": native["m1_final_mse"],
        "dev_born_fraction_final": development["m3_born_fraction"][-1],
        "native_born_fraction_final": native["m3_born_fraction"][-1],
        "native_born_corr": native["m4_born_error_correlation_pooled"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
