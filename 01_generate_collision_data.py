import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================
# Config
# ==========================================
MU = 398600.4418
R_EQ = 6378.137
J2 = 1.08262668e-3
TIME_SPAN = 48 * 3600.0

LOW_A_MIN = R_EQ + 200.0
LOW_A_MAX = R_EQ + 2000.0
HIGH_A_MIN = LOW_A_MAX
HIGH_A_MAX = 42364.0

COARSE_STEP = float(os.environ.get("COARSE_STEP", "60.0"))
TOP_CANDIDATES = int(os.environ.get("TOP_CANDIDATES", "10"))
ORBIT_RTOL = float(os.environ.get("ORBIT_RTOL", "1e-10"))
MINIMIZE_XATOL = float(os.environ.get("MINIMIZE_XATOL", "1e-3"))

Q_POINTS = int(os.environ.get("Q_POINTS", "5"))
RANDOM_PER_GROUP = int(os.environ.get("RANDOM_PER_GROUP", "1"))
WORKERS = int(os.environ.get("WORKERS", str(min(32, os.cpu_count() or 4))))
SHARD_GROUPS = int(os.environ.get("SHARD_GROUPS", "200"))
COMPRESS_SHARDS = os.environ.get("COMPRESS_SHARDS", "0") == "1"

DATA_DIR = Path("data")
DATASET_NAME = os.environ.get("DATASET_NAME", "high_precision_100k_v1")
SHARD_DIR = DATA_DIR / "shards" / DATASET_NAME


@dataclass(frozen=True)
class Layer:
    key: str
    name: str
    source_code: int
    groups: int
    distance_range_km: tuple[float, float] | None
    seed_offset: int
    samples_per_group: int


LAYERS = [
    Layer("danger_0_5", "constructed_danger_0_5km", 1, int(os.environ.get("DANGER_0_5_GROUPS", "2000")), (0.0, 5.0), 0, Q_POINTS),
    Layer("danger_5_10", "constructed_danger_5_10km", 2, int(os.environ.get("DANGER_5_10_GROUPS", "2000")), (5.0, 10.0), 100_000, Q_POINTS),
    Layer("near_safe", "constructed_near_safe_10_30km", 3, int(os.environ.get("NEAR_SAFE_GROUPS", "2000")), (10.0, 30.0), 200_000, Q_POINTS),
    Layer("mid_safe", "constructed_mid_safe_30_100km", 4, int(os.environ.get("MID_SAFE_GROUPS", "2000")), (30.0, 100.0), 400_000, Q_POINTS),
    Layer("random", "random_background_low_high", 0, int(os.environ.get("RANDOM_GROUPS", "2000")), None, 1_000_000, RANDOM_PER_GROUP),
]
SOURCE_NAMES = {layer.source_code: layer.name for layer in LAYERS}


# ==========================================
# Orbit and label calculation
# ==========================================
def j2_derivative(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x * x + y * y + z * z)
    zr = z * z / (r * r)
    j2 = 1.5 * J2 * MU * R_EQ**2 / r**5
    return np.array([
        vx,
        vy,
        vz,
        -MU * x / r**3 + j2 * x * (5 * zr - 1),
        -MU * y / r**3 + j2 * y * (5 * zr - 1),
        -MU * z / r**3 + j2 * z * (5 * zr - 3),
    ])


def coe2rv(a, e, inc, raan, argp, mean_anomaly):
    E = mean_anomaly
    for _ in range(15):
        E -= (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))

    r_pqw = np.array([a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E), 0.0])
    v_scale = np.sqrt(MU / a) / (1.0 - e * np.cos(E))
    v_pqw = np.array([-v_scale * np.sin(E), v_scale * np.sqrt(1.0 - e * e) * np.cos(E), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    rot = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pqw, rot @ v_pqw


def rv2coe(r, v):
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    vr = np.dot(r, v) / rn
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    inc = np.arccos(np.clip(h[2] / hn, -1.0, 1.0))

    node = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(node)
    raan = np.arccos(np.clip(node[0] / nn, -1.0, 1.0)) if nn > 0 else 0.0
    if node[1] < 0:
        raan = 2.0 * np.pi - raan

    e_vec = ((vn * vn - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(e_vec)
    argp = np.arccos(np.clip(np.dot(node, e_vec) / (nn * e), -1.0, 1.0)) if nn > 0 and e > 0 else 0.0
    if e_vec[2] < 0:
        argp = 2.0 * np.pi - argp

    true_anomaly = np.arccos(np.clip(np.dot(e_vec, r) / (e * rn), -1.0, 1.0)) if e > 0 else 0.0
    if vr < 0:
        true_anomaly = 2.0 * np.pi - true_anomaly

    a = 1.0 / (2.0 / rn - vn * vn / MU)
    E = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(true_anomaly / 2.0),
        np.sqrt(1.0 + e) * np.cos(true_anomaly / 2.0),
    )
    mean_anomaly = (E - e * np.sin(E)) % (2.0 * np.pi)
    return np.array([a, e, inc, raan, argp, mean_anomaly])


def propagate_state(r0, v0, t0, t1, dense_output=False):
    return solve_ivp(
        j2_derivative,
        (t0, t1),
        np.concatenate([r0, v0]),
        method="DOP853",
        dense_output=dense_output,
        rtol=ORBIT_RTOL,
    )


def closest_approach(sol_a, sol_b, time_span=TIME_SPAN):
    t_grid = np.arange(0.0, time_span + COARSE_STEP, COARSE_STEP)
    t_grid[-1] = time_span
    d_grid = np.linalg.norm(sol_a.sol(t_grid)[:3].T - sol_b.sol(t_grid)[:3].T, axis=1)

    local_idx = np.where((d_grid[1:-1] <= d_grid[:-2]) & (d_grid[1:-1] <= d_grid[2:]))[0] + 1
    candidate_idx = np.unique(np.concatenate([local_idx, [0, len(t_grid) - 1]]))
    if len(candidate_idx) > TOP_CANDIDATES:
        candidate_idx = candidate_idx[np.argsort(d_grid[candidate_idx])[:TOP_CANDIDATES]]

    def distance_at(t):
        return np.linalg.norm(sol_a.sol(t)[:3] - sol_b.sol(t)[:3])

    best_d, best_t = float("inf"), 0.0
    for idx in candidate_idx:
        left = max(0.0, t_grid[idx] - COARSE_STEP)
        right = min(time_span, t_grid[idx] + COARSE_STEP)
        result = minimize_scalar(distance_at, bounds=(left, right), method="bounded", options={"xatol": MINIMIZE_XATOL})
        if result.success and result.fun < best_d:
            best_d, best_t = float(result.fun), float(result.x)
    return best_d, best_t


def label_pair(coe_a, coe_b):
    r_a, v_a = coe2rv(*coe_a)
    r_b, v_b = coe2rv(*coe_b)
    sol_a = propagate_state(r_a, v_a, 0.0, TIME_SPAN, dense_output=True)
    sol_b = propagate_state(r_b, v_b, 0.0, TIME_SPAN, dense_output=True)
    return closest_approach(sol_a, sol_b)


# ==========================================
# Sampling
# ==========================================
def sample_valid_coe(orbit_type=None):
    if orbit_type == "low":
        a = np.random.uniform(LOW_A_MIN, LOW_A_MAX)
    elif orbit_type == "high":
        a = np.random.uniform(HIGH_A_MIN, HIGH_A_MAX)
    else:
        a = np.random.uniform(LOW_A_MIN, HIGH_A_MAX)

    e = np.random.uniform(0.01, 0.9)
    while a * (1.0 - e) < R_EQ + 200.0:
        e *= 0.9

    return np.array([
        a,
        e,
        np.random.uniform(0.0, np.pi),
        np.random.uniform(0.0, 2.0 * np.pi),
        np.random.uniform(0.0, 2.0 * np.pi),
        np.random.uniform(0.0, 2.0 * np.pi),
    ])


def sample_offset(r_min, r_max):
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)
    radius = (r_min**3 + np.random.rand() * (r_max**3 - r_min**3)) ** (1.0 / 3.0)
    return direction * radius


def generate_random_group(seed, layer):
    np.random.seed(layer.seed_offset + seed)
    X, Y = [], []
    for _ in range(layer.samples_per_group):
        first_is_low = np.random.rand() < 0.5
        coe_a = sample_valid_coe("low" if first_is_low else "high")
        coe_b = sample_valid_coe("high" if first_is_low else "low")
        dc, tc = label_pair(coe_a, coe_b)
        X.append(np.concatenate([coe_a, coe_b]))
        Y.append([dc, tc, layer.source_code])
    return X, Y


def generate_constructed_group(seed, layer, max_retry_per_point=12):
    np.random.seed(layer.seed_offset + seed)
    X, Y = [], []
    r_min, r_max = layer.distance_range_km

    base_coe = sample_valid_coe()
    r_col_a, v_col_a = coe2rv(*base_coe)
    t_enc = np.random.uniform(3600.0, TIME_SPAN)

    sol_a_back = propagate_state(r_col_a, v_col_a, t_enc, 0.0)
    r_a0, v_a0 = sol_a_back.y[:3, -1], sol_a_back.y[3:, -1]
    coe_a0 = rv2coe(r_a0, v_a0)
    sol_a = propagate_state(r_a0, v_a0, 0.0, TIME_SPAN, dense_output=True)

    attempts = 0
    max_attempts = layer.samples_per_group * max_retry_per_point
    while len(X) < layer.samples_per_group and attempts < max_attempts:
        attempts += 1
        r_col_b = r_col_a + sample_offset(r_min, r_max)
        v_col_b = v_col_a + np.random.uniform(-0.01, 0.01, 3)

        sol_b_back = propagate_state(r_col_b, v_col_b, t_enc, 0.0)
        r_b0, v_b0 = sol_b_back.y[:3, -1], sol_b_back.y[3:, -1]
        coe_b0 = rv2coe(r_b0, v_b0)
        sol_b = propagate_state(r_b0, v_b0, 0.0, TIME_SPAN, dense_output=True)

        dc, tc = closest_approach(sol_a, sol_b)
        if r_min <= dc <= r_max:
            X.append(np.concatenate([coe_a0, coe_b0]))
            Y.append([dc, tc, layer.source_code])
    return X, Y


def generate_group(layer, seed):
    if layer.distance_range_km is None:
        return generate_random_group(seed, layer)
    return generate_constructed_group(seed, layer)


# ==========================================
# Save, merge, metadata
# ==========================================
def group_id(layer, seed):
    return layer.source_code * 10_000_000 + seed


def save_layer_shards(layer):
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_count = (layer.groups + SHARD_GROUPS - 1) // SHARD_GROUPS
    shard_paths = []

    for shard_idx in range(shard_count):
        start = shard_idx * SHARD_GROUPS
        end = min(layer.groups, start + SHARD_GROUPS)
        path = SHARD_DIR / f"{layer.key}_{shard_idx:05d}.npz"
        shard_paths.append(path)

        if path.exists():
            print(f"Skip existing shard: {path}")
            continue

        X_part, Y_part, G_part = [], [], []
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            future_to_seed = {executor.submit(generate_group, layer, seed): seed for seed in range(start, end)}
            for future in tqdm(as_completed(future_to_seed), total=end - start, desc=f"{layer.key} {shard_idx + 1}/{shard_count}"):
                seed = future_to_seed[future]
                x_g, y_g = future.result()
                X_part.extend(x_g)
                Y_part.extend(y_g)
                G_part.extend([group_id(layer, seed)] * len(x_g))

        arrays = {
            "X": np.asarray(X_part, dtype=np.float64),
            "Y": np.asarray(Y_part, dtype=np.float64),
            "G": np.asarray(G_part, dtype=np.int64),
        }
        if COMPRESS_SHARDS:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        print(f"Saved shard: {path}, samples: {len(X_part)}")

    return shard_paths


def merge_shards(paths):
    X_parts, Y_parts, G_parts = [], [], []
    for path in tqdm(paths, desc="Merge shards"):
        data = np.load(path)
        if len(data["X"]) == 0:
            continue
        X_parts.append(data["X"])
        Y_parts.append(data["Y"])
        G_parts.append(data["G"])

    if not X_parts:
        raise RuntimeError("No generated samples were found.")
    return np.concatenate(X_parts), np.concatenate(Y_parts), np.concatenate(G_parts)


def count_distance_bins(dc):
    return {
        "0_5km": int(np.sum((0.0 <= dc) & (dc <= 5.0))),
        "5_10km": int(np.sum((5.0 < dc) & (dc <= 10.0))),
        "10_30km": int(np.sum((10.0 < dc) & (dc <= 30.0))),
        "30_100km": int(np.sum((30.0 < dc) & (dc <= 100.0))),
        "gt_100km": int(np.sum(dc > 100.0)),
    }


def make_metadata(X, Y, G):
    planned = {layer.name: layer.groups * layer.samples_per_group for layer in LAYERS}
    source_counts = {SOURCE_NAMES[code]: int(np.sum(Y[:, 2] == code)) for code in SOURCE_NAMES}
    return {
        "dataset_name": DATASET_NAME,
        "sample_count": int(len(X)),
        "planned_counts": planned,
        "source_code_mapping": SOURCE_NAMES,
        "source_counts": source_counts,
        "distance_counts_by_true_dc": count_distance_bins(Y[:, 0]),
        "danger_class_definition": "Y_class = 1 if true Dc <= 10 km else 0",
        "file_schema": {
            "X_coe.npy": "[N, 12], satellite A and B orbital elements: a,e,i,Omega,omega,M",
            "Y_reg.npy": "[N, 3], true Dc_km, true Tc_s, source_code",
            "Y_class.npy": "[N], binary danger label from true Dc <= 10 km",
            "source_code.npy": "[N], copied from Y[:,2]",
            "group_id.npy": "[N], same constructed encounter group has same id",
        },
        "label_method": {
            "dynamics": "J2 numerical propagation",
            "time_window_s": TIME_SPAN,
            "coarse_step_s": COARSE_STEP,
            "top_candidates": TOP_CANDIDATES,
            "local_refinement": "minimize_scalar on dense_output",
            "minimize_xatol_s": MINIMIZE_XATOL,
            "orbit_rtol": ORBIT_RTOL,
        },
        "runtime": {
            "workers": WORKERS,
            "shard_groups": SHARD_GROUPS,
            "compress_shards": COMPRESS_SHARDS,
        },
    }


def save_final_dataset(X, Y, G):
    DATA_DIR.mkdir(exist_ok=True)
    perm = np.random.default_rng(42).permutation(len(X))
    X, Y, G = X[perm], Y[perm], G[perm]

    np.save(DATA_DIR / "X_coe.npy", X)
    np.save(DATA_DIR / "Y_reg.npy", Y)
    np.save(DATA_DIR / "Y_class.npy", (Y[:, 0] <= 10.0).astype(np.float64))
    np.save(DATA_DIR / "source_code.npy", Y[:, 2].astype(np.int64))
    np.save(DATA_DIR / "group_id.npy", G)

    metadata = make_metadata(X, Y, G)
    for path in [DATA_DIR / "dataset_metadata.json", DATA_DIR / f"{DATASET_NAME}_metadata.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    return X, Y, G, metadata


def main():
    planned_total = sum(layer.groups * layer.samples_per_group for layer in LAYERS)
    print("Start high-precision layered dataset generation.")
    print(f"Dataset: {DATASET_NAME}, planned samples: {planned_total}")
    print(f"Label settings: COARSE_STEP={COARSE_STEP}s, TOP_CANDIDATES={TOP_CANDIDATES}, ORBIT_RTOL={ORBIT_RTOL}")
    print(f"Workers={WORKERS}, shard_groups={SHARD_GROUPS}, shard_dir={SHARD_DIR}")
    print("Y[:,0]=true Dc_km, Y[:,1]=true Tc_s, Y[:,2]=source_code; Y_class is true Dc<=10km.")

    shard_paths = []
    for layer in LAYERS:
        print(f"\nLayer: {layer.name}, planned={layer.groups * layer.samples_per_group}")
        shard_paths.extend(save_layer_shards(layer))

    X, Y, G = merge_shards(shard_paths)
    X, Y, G, metadata = save_final_dataset(X, Y, G)
    danger = Y[:, 0] <= 10.0

    print("\nDataset generation finished.")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, group_id shape: {G.shape}")
    print(f"Danger samples: {int(np.sum(danger))}, safe samples: {int(np.sum(~danger))}")
    print(f"Distance counts: {metadata['distance_counts_by_true_dc']}")
    print(f"Source counts: {metadata['source_counts']}")
    print(f"Dc range: min={np.min(Y[:, 0]):.6f} km, max={np.max(Y[:, 0]):.6f} km")
    print("Saved: data/X_coe.npy, data/Y_reg.npy, data/Y_class.npy, data/source_code.npy, data/group_id.npy")
    print("Saved metadata: data/dataset_metadata.json")


if __name__ == "__main__":
    main()
