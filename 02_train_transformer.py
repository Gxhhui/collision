import csv
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_ENABLE_X64"] = "True"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
    PLOT_IMPORT_ERROR = ""
except Exception as exc:
    plt = None
    HAS_MATPLOTLIB = False
    PLOT_IMPORT_ERROR = str(exc)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


MU = 398600.4418
R_EQ = 6378.137
J2 = 1.08262668e-3
DANGER_DISTANCE_KM = 10.0
MAX_CYCLES = 40
TIME_SPAN = 48 * 3600.0
BATCH_SIZE = 256
EPOCHS = 150
PATIENCE = 30
INPUT_DIM = 18

CLASS_LOSS_WEIGHT = 1.0
DIST_LOSS_WEIGHT = 10.0
CYCLE_LOSS_WEIGHT = 2.0
PHASE_LOSS_WEIGHT = 6.0
PHYSICS_CONSISTENCY_WEIGHT = 1.0
TOP_K_CYCLE_REFINE = 3
TOP_K_CYCLE_DIAGNOSTIC = 5
CYCLE_REFINE_CONFIDENCE_THRESHOLD = 0.80
HIGH_CONF_WRONG_CYCLE_THRESHOLD = 0.95
CYCLE_SOFT_NEIGHBOR_WEIGHT = 0.10
SUSPICIOUS_REFINE_RAW_DISTANCE_KM = 100.0
SUSPICIOUS_REFINE_CANDIDATE_DISTANCE_KM = 30.0
SUSPICIOUS_REFINE_IMPROVEMENT_RATIO = 0.50
LOCAL_PHYSICAL_REFINE = os.environ.get("LOCAL_PHYSICAL_REFINE", "1") == "1"
LOCAL_REFINE_DANGER_PROB_THRESHOLD = float(os.environ.get("LOCAL_REFINE_DANGER_PROB_THRESHOLD", "0.5"))
LOCAL_REFINE_WINDOW_FRACTION = float(os.environ.get("LOCAL_REFINE_WINDOW_FRACTION", "0.35"))
LOCAL_REFINE_MAX_SAMPLES = int(os.environ.get("LOCAL_REFINE_MAX_SAMPLES", "0"))
LOCAL_REFINE_RTOL = float(os.environ.get("LOCAL_REFINE_RTOL", "1e-9"))
LOCAL_REFINE_XATOL = float(os.environ.get("LOCAL_REFINE_XATOL", "1e-3"))


def orbital_period(a_km):
    return 2.0 * np.pi * np.sqrt((a_km**3) / MU)


def pair_mean_period(X_raw):
    period_a = orbital_period(X_raw[:, 0])
    period_b = orbital_period(X_raw[:, 6])
    return 0.5 * (period_a + period_b)


def encode_angles(X_raw):
    """Encode two 6-element orbits into two 9-dimensional Transformer tokens."""
    if X_raw.shape[1] != 12:
        raise ValueError(f"X_coe.npy must have 12 columns, got {X_raw.shape[1]}.")
    aA, eA, iA, OmA, wA, MA = [X_raw[:, k] for k in range(6)]
    aB, eB, iB, OmB, wB, MB = [X_raw[:, k] for k in range(6, 12)]
    return jnp.stack(
        [
            aA,
            eA,
            iA,
            jnp.sin(OmA),
            jnp.cos(OmA),
            jnp.sin(wA),
            jnp.cos(wA),
            jnp.sin(MA),
            jnp.cos(MA),
            aB,
            eB,
            iB,
            jnp.sin(OmB),
            jnp.cos(OmB),
            jnp.sin(wB),
            jnp.cos(wB),
            jnp.sin(MB),
            jnp.cos(MB),
        ],
        axis=1,
    )


def build_features(X_raw):
    return np.array(encode_angles(jnp.array(X_raw)))


def masked_mean(values, mask):
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(values * mask) / denom


def phase_to_vector(phase):
    angle = 2.0 * jnp.pi * phase
    return jnp.stack([jnp.sin(angle), jnp.cos(angle)], axis=-1)


def vector_to_phase(phase_vec):
    phase = jnp.arctan2(phase_vec[..., 0], phase_vec[..., 1]) / (2.0 * jnp.pi)
    return jnp.mod(phase, 1.0)


def circular_phase_error(pred_phase, true_phase):
    delta = jnp.abs(pred_phase - true_phase)
    return jnp.minimum(delta, 1.0 - delta)


def topk_candidate_value(values, row_idx, candidate_idx, default_value):
    if candidate_idx < values.shape[1]:
        return values[row_idx, candidate_idx]
    return default_value


def j2_derivative_np(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x * x + y * y + z * z)
    zr = z * z / (r * r)
    j2 = 1.5 * J2 * MU * R_EQ**2 / r**5
    return np.array(
        [
            vx,
            vy,
            vz,
            -MU * x / r**3 + j2 * x * (5.0 * zr - 1.0),
            -MU * y / r**3 + j2 * y * (5.0 * zr - 1.0),
            -MU * z / r**3 + j2 * z * (5.0 * zr - 3.0),
        ]
    )


def coe2rv_np(a, e, inc, raan, argp, mean_anomaly):
    E = mean_anomaly
    for _ in range(15):
        E -= (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))

    r_pqw = np.array([a * (np.cos(E) - e), a * np.sqrt(1.0 - e * e) * np.sin(E), 0.0])
    v_scale = np.sqrt(MU / a) / (1.0 - e * np.cos(E))
    v_pqw = np.array([-v_scale * np.sin(E), v_scale * np.sqrt(1.0 - e * e) * np.cos(E), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    rot = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_pqw, rot @ v_pqw


def propagate_j2_dense(coe):
    r0, v0 = coe2rv_np(*coe)
    sol = solve_ivp(
        j2_derivative_np,
        (0.0, TIME_SPAN),
        np.concatenate([r0, v0]),
        method="DOP853",
        dense_output=True,
        rtol=LOCAL_REFINE_RTOL,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def local_physical_refine_pair(coe_pair, candidate_cycles, phase_value, t_ref):
    sol_a = propagate_j2_dense(coe_pair[:6])
    sol_b = propagate_j2_dense(coe_pair[6:])

    def distance_at(t_s):
        return np.linalg.norm(sol_a.sol(t_s)[:3] - sol_b.sol(t_s)[:3])

    best_d = float("inf")
    best_t = 0.0
    best_cycle = int(candidate_cycles[0])
    best_rank = -1
    half_window = max(60.0, LOCAL_REFINE_WINDOW_FRACTION * float(t_ref))

    for rank, cycle in enumerate(candidate_cycles, start=1):
        center_t = np.clip((float(cycle) + float(phase_value)) * float(t_ref), 0.0, TIME_SPAN)
        left = max(0.0, center_t - half_window)
        right = min(TIME_SPAN, center_t + half_window)
        if right <= left:
            continue
        result = minimize_scalar(distance_at, bounds=(left, right), method="bounded", options={"xatol": LOCAL_REFINE_XATOL})
        if result.success and float(result.fun) < best_d:
            best_d = float(result.fun)
            best_t = float(result.x)
            best_cycle = int(cycle)
            best_rank = rank

    return best_d, best_t, best_cycle, best_rank


def apply_local_physical_refinement(X_raw_eval, base_d_pred, base_t_pred, base_cycle_pred, top_cycles, phase_pred, t_ref, danger_prob):
    refined_d = np.array(base_d_pred, copy=True)
    refined_t = np.array(base_t_pred, copy=True)
    refined_cycle = np.array(base_cycle_pred, copy=True)
    refined_rank = np.full(len(base_t_pred), -1, dtype=int)
    refined_used = np.zeros(len(base_t_pred), dtype=bool)

    if not LOCAL_PHYSICAL_REFINE:
        return refined_d, refined_t, refined_cycle, refined_rank, refined_used

    candidate_indices = np.where(danger_prob >= LOCAL_REFINE_DANGER_PROB_THRESHOLD)[0]
    if LOCAL_REFINE_MAX_SAMPLES > 0:
        candidate_indices = candidate_indices[:LOCAL_REFINE_MAX_SAMPLES]

    total = len(candidate_indices)
    if total == 0:
        return refined_d, refined_t, refined_cycle, refined_rank, refined_used

    print(f"Local physical refinement: {total} predicted-danger samples, top-{top_cycles.shape[1]} candidates.")
    for count, idx in enumerate(candidate_indices, start=1):
        if count == 1 or count % 100 == 0 or count == total:
            print(f"  local refine progress: {count}/{total}")
        try:
            d_ref, t_refined, cycle_refined, rank_refined = local_physical_refine_pair(
                X_raw_eval[idx], top_cycles[idx], phase_pred[idx], t_ref[idx]
            )
        except Exception as exc:
            print(f"  local refine skipped index {idx}: {exc}")
            continue
        refined_d[idx] = d_ref
        refined_t[idx] = t_refined
        refined_cycle[idx] = cycle_refined
        refined_rank[idx] = rank_refined
        refined_used[idx] = True

    return refined_d, refined_t, refined_cycle, refined_rank, refined_used


def cycle_soft_targets(cycle_true):
    cycle_true = cycle_true.astype(jnp.int32)
    center_weight = 1.0 - CYCLE_SOFT_NEIGHBOR_WEIGHT
    cycle_ids = jnp.arange(MAX_CYCLES)

    center = (cycle_ids[None, :] == cycle_true[:, None]).astype(jnp.float64) * center_weight
    left_valid = (cycle_true > 0).astype(jnp.float64)
    right_valid = (cycle_true < MAX_CYCLES - 1).astype(jnp.float64)
    neighbor_count = jnp.maximum(left_valid + right_valid, 1.0)
    side_weight = CYCLE_SOFT_NEIGHBOR_WEIGHT / neighbor_count

    left_idx = jnp.maximum(cycle_true - 1, 0)
    right_idx = jnp.minimum(cycle_true + 1, MAX_CYCLES - 1)
    left = (cycle_ids[None, :] == left_idx[:, None]).astype(jnp.float64) * (side_weight * left_valid)[:, None]
    right = (cycle_ids[None, :] == right_idx[:, None]).astype(jnp.float64) * (side_weight * right_valid)[:, None]
    return center + left + right


def coe_feature_to_r_at_time(features, offset, t_s):
    a = features[:, offset + 0]
    e = features[:, offset + 1]
    inc = features[:, offset + 2]
    Omega = jnp.arctan2(features[:, offset + 3], features[:, offset + 4])
    w = jnp.arctan2(features[:, offset + 5], features[:, offset + 6])
    M0 = jnp.arctan2(features[:, offset + 7], features[:, offset + 8])
    n = jnp.sqrt(MU / (a**3))
    M = jnp.mod(M0 + n * t_s, 2.0 * jnp.pi)

    E = M
    for _ in range(8):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))

    x_p = a * (jnp.cos(E) - e)
    y_p = a * jnp.sqrt(jnp.maximum(1.0 - e**2, 1e-10)) * jnp.sin(E)

    cO, sO = jnp.cos(Omega), jnp.sin(Omega)
    cw, sw = jnp.cos(w), jnp.sin(w)
    ci, si = jnp.cos(inc), jnp.sin(inc)

    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    r31 = sw * si
    r32 = cw * si

    return jnp.stack(
        [
            r11 * x_p + r12 * y_p,
            r21 * x_p + r22 * y_p,
            r31 * x_p + r32 * y_p,
        ],
        axis=1,
    )


def physical_distance_at_predicted_time(x_batch, cycle_logits, phase_preds):
    cycle_prob = jax.nn.softmax(cycle_logits, axis=-1)
    cycle_values = jnp.arange(MAX_CYCLES, dtype=jnp.float64)
    cycle_expect = jnp.sum(cycle_prob * cycle_values[None, :], axis=-1)
    period_a = 2.0 * jnp.pi * jnp.sqrt((x_batch[:, 0] ** 3) / MU)
    period_b = 2.0 * jnp.pi * jnp.sqrt((x_batch[:, 9] ** 3) / MU)
    t_ref = 0.5 * (period_a + period_b)
    pred_t_s = jnp.clip((cycle_expect + phase_preds) * t_ref, 0.0, TIME_SPAN)
    r_a = coe_feature_to_r_at_time(x_batch, 0, pred_t_s)
    r_b = coe_feature_to_r_at_time(x_batch, 9, pred_t_s)
    return jnp.linalg.norm(r_b - r_a, axis=1)


def refine_cycle_by_physical_distance(x_batch, cycle_logits, phase_preds, top_k=TOP_K_CYCLE_REFINE):
    """Choose a cycle from top-k candidates with a conservative physical sanity check."""
    cycle_prob = jax.nn.softmax(cycle_logits, axis=-1)
    _, top_cycles = jax.lax.top_k(cycle_logits, top_k)
    period_a = 2.0 * jnp.pi * jnp.sqrt((x_batch[:, 0] ** 3) / MU)
    period_b = 2.0 * jnp.pi * jnp.sqrt((x_batch[:, 9] ** 3) / MU)
    t_ref = 0.5 * (period_a + period_b)
    t_candidates = jnp.clip((top_cycles.astype(jnp.float64) + phase_preds[:, None]) * t_ref[:, None], 0.0, TIME_SPAN)

    n, k = top_cycles.shape
    x_flat = jnp.repeat(x_batch, k, axis=0)
    t_flat = t_candidates.reshape(-1)
    r_a = coe_feature_to_r_at_time(x_flat, 0, t_flat)
    r_b = coe_feature_to_r_at_time(x_flat, 9, t_flat)
    distances = jnp.linalg.norm(r_b - r_a, axis=1).reshape(n, k)

    best_idx = jnp.argmin(distances, axis=1)
    raw_cycle = top_cycles[:, 0]
    raw_t = t_candidates[:, 0]
    raw_distance = distances[:, 0]
    raw_confidence = jnp.max(cycle_prob, axis=1)

    candidate_cycle = jnp.take_along_axis(top_cycles, best_idx[:, None], axis=1)[:, 0]
    candidate_t = jnp.take_along_axis(t_candidates, best_idx[:, None], axis=1)[:, 0]
    candidate_distance = jnp.take_along_axis(distances, best_idx[:, None], axis=1)[:, 0]

    high_confidence = raw_confidence >= CYCLE_REFINE_CONFIDENCE_THRESHOLD
    suspicious_raw = (
        (raw_distance >= SUSPICIOUS_REFINE_RAW_DISTANCE_KM)
        & (candidate_distance <= SUSPICIOUS_REFINE_CANDIDATE_DISTANCE_KM)
        & (candidate_distance <= raw_distance * SUSPICIOUS_REFINE_IMPROVEMENT_RATIO)
    )
    keep_raw = high_confidence & (~suspicious_raw)
    refined_cycle = jnp.where(keep_raw, raw_cycle, candidate_cycle)
    refined_t = jnp.where(keep_raw, raw_t, candidate_t)
    refined_distance = jnp.where(keep_raw, raw_distance, candidate_distance)
    return refined_cycle, refined_t, refined_distance, top_cycles, distances, suspicious_raw


print("正在读取数据集，并构造 18 维轨道根数 + 角度 sin/cos 特征...")
try:
    X_raw_data = np.load("data/X_coe.npy")
    Y_raw_data = np.load("data/Y_reg.npy")
except FileNotFoundError:
    print("未找到数据文件，请先运行 01_generate_collision_data.py。")
    raise SystemExit(1)

if Y_raw_data.shape[1] < 2:
    raise ValueError("Y_reg.npy 至少需要两列：[Dc_km, Tc_s]。")

X_feat_all = build_features(X_raw_data)
T_ref_all = pair_mean_period(X_raw_data)
danger_all = (Y_raw_data[:, 0] <= DANGER_DISTANCE_KM).astype(np.float64)

np.random.seed(42)
perm = np.random.permutation(len(X_feat_all))
X_feat_all = X_feat_all[perm]
X_raw_all = X_raw_data[perm]
Y_raw_data = Y_raw_data[perm]
T_ref_all = T_ref_all[perm]
danger_all = danger_all[perm]

n_total = len(X_feat_all)
train_end = int(0.70 * n_total)
val_end = int(0.85 * n_total)

X_train, Y_train = X_feat_all[:train_end], Y_raw_data[:train_end]
X_val, Y_val = X_feat_all[train_end:val_end], Y_raw_data[train_end:val_end]
X_test, Y_test = X_feat_all[val_end:], Y_raw_data[val_end:]
X_raw_train = X_raw_all[:train_end]
X_raw_val = X_raw_all[train_end:val_end]
X_raw_test = X_raw_all[val_end:]
T_ref_train = T_ref_all[:train_end]
T_ref_val = T_ref_all[train_end:val_end]
T_ref_test = T_ref_all[val_end:]
danger_train = danger_all[:train_end]
danger_val = danger_all[train_end:val_end]
danger_test = danger_all[val_end:]

print(f"数据集划分 | 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
print(
    f"危险样本比例 | 训练集: {danger_train.mean() * 100:.2f}% | "
    f"验证集: {danger_val.mean() * 100:.2f}% | 测试集: {danger_test.mean() * 100:.2f}%"
)

X_mean = X_train.mean(0)
X_std = np.where(X_train.std(0) == 0, 1.0, X_train.std(0))
X_m_j, X_s_j = jnp.array(X_mean), jnp.array(X_std)

danger_train_mask = danger_train > 0.5
if np.sum(danger_train_mask) == 0:
    raise ValueError("训练集中没有危险样本，无法训练 Dc/Tc 回归头。")

d_train_norm = np.clip(Y_train[:, 0] / DANGER_DISTANCE_KM, 0.0, 1.0)
d_val_norm = np.clip(Y_val[:, 0] / DANGER_DISTANCE_KM, 0.0, 1.0)
d_test_norm = np.clip(Y_test[:, 0] / DANGER_DISTANCE_KM, 0.0, 1.0)
cycle_train = jnp.clip(jnp.floor(Y_train[:, 1] / T_ref_train).astype(jnp.int32), 0, MAX_CYCLES - 1)
cycle_val = jnp.clip(jnp.floor(Y_val[:, 1] / T_ref_val).astype(jnp.int32), 0, MAX_CYCLES - 1)
cycle_test = jnp.clip(jnp.floor(Y_test[:, 1] / T_ref_test).astype(jnp.int32), 0, MAX_CYCLES - 1)
phase_train = jnp.array((Y_train[:, 1] / T_ref_train) - np.floor(Y_train[:, 1] / T_ref_train))
phase_val = jnp.array((Y_val[:, 1] / T_ref_val) - np.floor(Y_val[:, 1] / T_ref_val))
phase_test = jnp.array((Y_test[:, 1] / T_ref_test) - np.floor(Y_test[:, 1] / T_ref_test))


class OrbitTransformer(nn.Module):
    d_model: int = 256

    @nn.compact
    def __call__(self, x_raw):
        x = (x_raw - X_m_j) / X_s_j
        tokens = x.reshape(2, 9)

        x = nn.Dense(self.d_model)(tokens)
        attn_out = nn.SelfAttention(num_heads=4)(x)
        x = nn.LayerNorm()(x + attn_out).reshape(-1)

        x = nn.swish(nn.Dense(512)(x))
        x = nn.swish(nn.Dense(256)(x))
        x = nn.swish(nn.Dense(128)(x))

        dc_h = nn.swish(nn.Dense(128)(x))
        dc_h = nn.swish(nn.Dense(64)(dc_h))
        dc_h = nn.swish(nn.Dense(32)(dc_h))
        d_pred = nn.sigmoid(nn.Dense(1)(dc_h)[0])

        cycle_logits = nn.Dense(MAX_CYCLES)(x)
        phase_vec_raw = nn.Dense(2)(x)
        phase_vec = phase_vec_raw / jnp.maximum(jnp.linalg.norm(phase_vec_raw), 1e-8)
        phase_pred = vector_to_phase(phase_vec)
        danger_logit = nn.Dense(1)(x)[0]

        return d_pred, cycle_logits, phase_pred, phase_vec, danger_logit


model = OrbitTransformer()


def hybrid_loss(params, x_batch, d_true_norm, cycle_true, phase_true, danger_true):
    d_preds, cycle_logits, phase_preds, phase_vec_preds, danger_logits = jax.vmap(
        lambda x: model.apply(params, x)
    )(x_batch)
    danger_mask = danger_true.astype(jnp.float64)

    loss_cls = jnp.mean(optax.sigmoid_binary_cross_entropy(danger_logits, danger_true))
    dist_weights = 1.0 + 3.0 * jnp.exp(-5.0 * d_true_norm)
    loss_d = masked_mean(((d_preds - d_true_norm) ** 2) * dist_weights, danger_mask)
    loss_cycle_each = optax.softmax_cross_entropy(cycle_logits, cycle_soft_targets(cycle_true))
    loss_cycle = masked_mean(loss_cycle_each, danger_mask)
    true_phase_vec = phase_to_vector(phase_true)
    loss_phase = masked_mean(jnp.sum((phase_vec_preds - true_phase_vec) ** 2, axis=-1), danger_mask)

    d_phys_norm = jnp.clip(
        physical_distance_at_predicted_time(x_batch, cycle_logits, phase_preds) / DANGER_DISTANCE_KM,
        0.0,
        2.0,
    )
    loss_physics = masked_mean(optax.huber_loss(d_phys_norm, d_preds, delta=0.2), danger_mask)

    total_loss = (
        CLASS_LOSS_WEIGHT * loss_cls
        + DIST_LOSS_WEIGHT * loss_d
        + CYCLE_LOSS_WEIGHT * loss_cycle
        + PHASE_LOSS_WEIGHT * loss_phase
        + PHYSICS_CONSISTENCY_WEIGHT * loss_physics
    )
    return total_loss, (loss_cls, loss_d, loss_cycle, loss_phase)


num_batches = len(X_train) // BATCH_SIZE
if num_batches == 0:
    raise ValueError("训练集样本数小于 BATCH_SIZE。")

lr_sched = optax.cosine_decay_schedule(1e-3, EPOCHS * num_batches)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_sched, weight_decay=1e-4))

params = model.init(jax.random.PRNGKey(42), jnp.ones((INPUT_DIM,)))
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, x, d, cycle, phase, danger):
    (loss, aux), grads = jax.value_and_grad(hybrid_loss, has_aux=True)(params, x, d, cycle, phase, danger)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


@jax.jit
def predict_batch(params, x):
    return jax.vmap(lambda row: model.apply(params, row))(x)


def eval_metrics(params, X_eval, Y_eval, d_eval_norm, cycle_eval, phase_eval, T_ref_eval, danger_eval):
    _, (loss_cls, loss_d, loss_cycle, loss_phase) = hybrid_loss(
        params,
        jnp.array(X_eval),
        jnp.array(d_eval_norm),
        jnp.array(cycle_eval),
        jnp.array(phase_eval),
        jnp.array(danger_eval),
    )
    d_norm_pred, cycle_logits, phase_pred, _, danger_logits = predict_batch(params, jnp.array(X_eval))
    danger_prob = jax.nn.sigmoid(danger_logits)
    danger_pred = danger_prob >= 0.5
    danger_true = jnp.array(danger_eval) >= 0.5
    danger_mask = danger_true.astype(jnp.float64)
    safe_mask = (~danger_true).astype(jnp.float64)

    d_pred = jnp.clip(d_norm_pred, 0.0, 1.0) * DANGER_DISTANCE_KM
    cycle_pred, t_pred, _, _, _, _ = refine_cycle_by_physical_distance(
        jnp.array(X_eval), cycle_logits, phase_pred, TOP_K_CYCLE_REFINE
    )

    cls_acc = jnp.mean(danger_pred == danger_true)
    tp = jnp.sum((danger_pred == 1) & (danger_true == 1))
    fp = jnp.sum((danger_pred == 1) & (danger_true == 0))
    fn = jnp.sum((danger_pred == 0) & (danger_true == 1))
    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-8)
    false_alarm = fp / jnp.maximum(jnp.sum(safe_mask), 1.0)
    miss_rate = fn / jnp.maximum(jnp.sum(danger_mask), 1.0)

    d_mae = masked_mean(jnp.abs(d_pred - jnp.array(Y_eval[:, 0])), danger_mask)
    t_mae_min = masked_mean(jnp.abs(t_pred - jnp.array(Y_eval[:, 1])), danger_mask) / 60.0
    k_pred = t_pred / jnp.array(T_ref_eval)
    k_true = jnp.array(Y_eval[:, 1]) / jnp.array(T_ref_eval)
    k_mae = masked_mean(jnp.abs(k_pred - k_true), danger_mask)
    cycle_acc = masked_mean((cycle_pred == cycle_eval).astype(jnp.float64), danger_mask)
    cycle_mae = masked_mean(jnp.abs(cycle_pred - cycle_eval), danger_mask)
    phase_mae = masked_mean(circular_phase_error(phase_pred, phase_eval), danger_mask)

    return {
        "loss_cls": loss_cls,
        "loss_d": loss_d,
        "loss_cycle": loss_cycle,
        "loss_phase": loss_phase,
        "cls_acc": cls_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm": false_alarm,
        "miss_rate": miss_rate,
        "d_mae": d_mae,
        "t_mae_min": t_mae_min,
        "k_mae": k_mae,
        "cycle_acc": cycle_acc,
        "cycle_mae": cycle_mae,
        "phase_mae": phase_mae,
    }


def val_metrics(params):
    return eval_metrics(params, X_val, Y_val, d_val_norm, cycle_val, phase_val, T_ref_val, danger_val)


def test_metrics(params):
    return eval_metrics(params, X_test, Y_test, d_test_norm, cycle_test, phase_test, T_ref_test, danger_test)


def to_float_dict(metrics):
    return {k: float(v) for k, v in metrics.items()}


def generate_report_figures():
    if not HAS_MATPLOTLIB:
        print(f"跳过报告图生成：matplotlib 不可用，原因：{PLOT_IMPORT_ERROR}")
        return ""

    fig_dir = os.path.join("logs", "report_figures")
    os.makedirs(fig_dir, exist_ok=True)
    metrics = np.genfromtxt(metrics_path, delimiter=",", names=True)
    if metrics.ndim == 0:
        metrics = np.array([metrics])

    epochs = metrics["epoch"]
    best_epoch_for_plot = int(best_epoch)

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, metrics["val_cls_acc"] * 100.0, label="Val accuracy (%)", lw=2)
    ax.plot(epochs, metrics["val_recall"] * 100.0, label="Val danger recall (%)", lw=2)
    ax.plot(epochs, metrics["val_false_alarm"] * 100.0, label="Val false alarm (%)", lw=2)
    ax.axvline(best_epoch_for_plot, color="#d62728", ls="--", lw=1.5, label=f"Best epoch {best_epoch_for_plot}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.suptitle("Classification Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "classification_curves.png"))
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(epochs, metrics["danger_d_mae_km"], color="#ff7f0e", lw=2, label="Danger D_MAE (km)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Distance MAE (km)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, metrics["danger_t_mae_min"], color="#1f77b4", lw=2, label="Danger T_MAE (min)")
    ax2.set_ylabel("Time MAE (min)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.suptitle("Danger Regression Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "danger_regression_curves.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, metrics["val_cls_loss"], lw=2, label="Classification loss")
    ax.plot(epochs, metrics["val_dist_loss"], lw=2, label="Distance loss")
    ax.plot(epochs, metrics["val_cycle_loss"], lw=2, label="Cycle loss")
    ax.plot(epochs, metrics["val_phase_loss"], lw=2, label="Phase loss")
    ax.axvline(best_epoch_for_plot, color="#d62728", ls="--", lw=1.5, label=f"Best epoch {best_epoch_for_plot}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.suptitle("Validation Loss Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "validation_loss_curves.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(Y_raw_data[danger_all > 0.5, 0], bins=40, alpha=0.8, label="Danger")
    axes[0].hist(Y_raw_data[danger_all <= 0.5, 0], bins=40, alpha=0.55, label="Safe/random")
    axes[0].set_xlabel("True closest distance Dc (km)")
    axes[0].set_ylabel("Sample count (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[1].bar(["Train", "Val", "Test"], [danger_train.mean(), danger_val.mean(), danger_test.mean()])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Danger ratio")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Dataset Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mixed_dataset_distribution.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["Accuracy", "Recall", "False alarm", "D_MAE km", "K_MAE", "Cycle acc"]
    vals = [
        float(test_m["cls_acc"]) * 100.0,
        float(test_m["recall"]) * 100.0,
        float(test_m["false_alarm"]) * 100.0,
        float(test_m["d_mae"]),
        float(test_m["k_mae"]),
        float(test_m["cycle_acc"]) * 100.0,
    ]
    bars = ax.bar(labels, vals, color=["#4c78a8", "#54a24b", "#e45756", "#f58518", "#72b7b2", "#59a14f"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Test Metrics, Best Epoch {best_epoch_for_plot}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "best_test_metrics.png"))
    plt.close(fig)

    return fig_dir


def analyze_test_time_errors():
    fig_dir = os.path.join("logs", "report_figures")
    os.makedirs(fig_dir, exist_ok=True)

    d_norm_pred, cycle_logits, phase_pred, _, danger_logits = predict_batch(params, jnp.array(X_test))
    danger_prob = np.array(jax.nn.sigmoid(danger_logits))
    cycle_prob = np.array(jax.nn.softmax(cycle_logits, axis=-1))
    pred_danger = danger_prob >= 0.5
    true_danger = danger_test >= 0.5
    danger_mask = true_danger

    raw_cycle_pred = np.array(jnp.argmax(cycle_logits, axis=-1))
    refined_cycle, refined_t_s, refined_proxy_distance, _, _, suspicious_recheck = refine_cycle_by_physical_distance(
        jnp.array(X_test), cycle_logits, phase_pred, TOP_K_CYCLE_REFINE
    )
    _, _, _, top_cycles_j, top_distances_j, _ = refine_cycle_by_physical_distance(
        jnp.array(X_test), cycle_logits, phase_pred, TOP_K_CYCLE_DIAGNOSTIC
    )
    cycle_pred = np.array(refined_cycle)
    top_cycles_np = np.array(top_cycles_j)
    top_distances_np = np.array(top_distances_j)
    phase_pred_np = np.array(phase_pred)
    true_cycle = np.array(cycle_test)
    true_phase = np.array(phase_test)
    pred_t_s = np.array(refined_t_s)
    d_pred_km = np.array(jnp.clip(d_norm_pred, 0.0, 1.0)) * DANGER_DISTANCE_KM
    refined_proxy_distance = np.array(refined_proxy_distance)
    suspicious_recheck = np.array(suspicious_recheck)
    true_t_s = Y_test[:, 1]
    raw_cycle_prob = cycle_prob[np.arange(len(cycle_prob)), raw_cycle_pred]
    refined_applied = raw_cycle_pred != cycle_pred
    top2_cycle = np.argsort(-cycle_prob, axis=1)[:, :2]
    true_cycle_prob = cycle_prob[np.arange(len(cycle_prob)), true_cycle]
    true_cycle_in_topk = np.any(top_cycles_np == true_cycle[:, None], axis=1)
    true_cycle_topk_rank = np.full(len(X_test), -1, dtype=int)
    true_cycle_topk_distance = np.full(len(X_test), np.nan, dtype=float)
    for row_idx in range(len(X_test)):
        match = np.where(top_cycles_np[row_idx] == true_cycle[row_idx])[0]
        if len(match) > 0:
            true_cycle_topk_rank[row_idx] = int(match[0]) + 1
            true_cycle_topk_distance[row_idx] = float(top_distances_np[row_idx, match[0]])

    local_refined_d_km, local_refined_t_s, local_refined_cycle, local_refined_rank, local_refined_used = apply_local_physical_refinement(
        X_raw_test,
        d_pred_km,
        pred_t_s,
        cycle_pred,
        top_cycles_np,
        phase_pred_np,
        T_ref_test,
        danger_prob,
    )
    d_pred_km = local_refined_d_km
    pred_t_s = local_refined_t_s
    cycle_pred = local_refined_cycle
    true_k = true_t_s / T_ref_test
    pred_k = pred_t_s / T_ref_test
    abs_k_err = np.abs(pred_k - true_k)
    abs_t_err_min = np.abs(pred_t_s - true_t_s) / 60.0
    abs_phase_err = np.minimum(np.abs(phase_pred_np - true_phase), 1.0 - np.abs(phase_pred_np - true_phase))
    cycle_correct = cycle_pred == true_cycle

    danger_err = abs_t_err_min[danger_mask]
    danger_cycle_correct = cycle_correct[danger_mask]
    danger_refined_applied = refined_applied[danger_mask]
    danger_suspicious_recheck = suspicious_recheck[danger_mask]
    danger_local_refined = local_refined_used[danger_mask]
    correct_err = danger_err[danger_cycle_correct]
    wrong_err = danger_err[~danger_cycle_correct]

    bins = [0, 5, 10, 30, 60, 120, 240, 1e9]
    labels = ["0-5", "5-10", "10-30", "30-60", "60-120", "120-240", ">240"]
    bin_counts = [
        int(np.sum((danger_err >= lo) & (danger_err < hi)))
        for lo, hi in zip(bins[:-1], bins[1:])
    ]

    csv_path = os.path.join("logs", "time_error_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "true_danger",
                "pred_danger",
                "danger_prob",
                "true_dc_km",
                "pred_dc_km",
                "true_tc_s",
                "pred_tc_s",
                "true_k",
                "pred_k",
                "abs_k_error",
                "abs_time_error_min",
                "true_cycle",
                "raw_pred_cycle",
                "pred_cycle",
                "refined_applied",
                "suspicious_physical_recheck",
                "local_physical_refined",
                "local_physical_rank",
                "cycle_correct",
                "refined_proxy_distance_km",
                "raw_cycle_prob",
                "pred_cycle_prob",
                "true_cycle_prob",
                "top2_cycle_1",
                "top2_cycle_2",
                "true_cycle_in_topk",
                "true_cycle_topk_rank",
                "true_cycle_topk_distance_km",
                "top1_cycle",
                "top1_distance_km",
                "top2_physical_cycle",
                "top2_distance_km",
                "top3_cycle",
                "top3_distance_km",
                "top4_cycle",
                "top4_distance_km",
                "top5_cycle",
                "top5_distance_km",
                "true_phase",
                "pred_phase",
                "abs_phase_error",
            ]
        )
        for i in range(len(X_test)):
            writer.writerow(
                [
                    i,
                    int(true_danger[i]),
                    int(pred_danger[i]),
                    float(danger_prob[i]),
                    float(Y_test[i, 0]),
                    float(d_pred_km[i]),
                    float(true_t_s[i]),
                    float(pred_t_s[i]),
                    float(true_k[i]),
                    float(pred_k[i]),
                    float(abs_k_err[i]),
                    float(abs_t_err_min[i]),
                    int(true_cycle[i]),
                    int(raw_cycle_pred[i]),
                    int(cycle_pred[i]),
                    int(refined_applied[i]),
                    int(suspicious_recheck[i]),
                    int(local_refined_used[i]),
                    int(local_refined_rank[i]),
                    int(cycle_correct[i]),
                    float(refined_proxy_distance[i]),
                    float(raw_cycle_prob[i]),
                    float(cycle_prob[i, cycle_pred[i]]),
                    float(true_cycle_prob[i]),
                    int(top2_cycle[i, 0]),
                    int(top2_cycle[i, 1]),
                    int(true_cycle_in_topk[i]),
                    int(true_cycle_topk_rank[i]),
                    float(true_cycle_topk_distance[i]),
                    int(top_cycles_np[i, 0]),
                    float(top_distances_np[i, 0]),
                    int(topk_candidate_value(top_cycles_np, i, 1, -1)),
                    float(topk_candidate_value(top_distances_np, i, 1, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 2, -1)),
                    float(topk_candidate_value(top_distances_np, i, 2, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 3, -1)),
                    float(topk_candidate_value(top_distances_np, i, 3, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 4, -1)),
                    float(topk_candidate_value(top_distances_np, i, 4, float("nan"))),
                    float(true_phase[i]),
                    float(phase_pred_np[i]),
                    float(abs_phase_err[i]),
                ]
            )

    cycle_case_path = os.path.join("logs", "cycle_error_cases.csv")
    danger_indices = np.where(danger_mask)[0]
    hard_indices = [
        int(i)
        for i in danger_indices
        if (not cycle_correct[i]) or abs_t_err_min[i] >= 60.0
    ]
    hard_indices = sorted(hard_indices, key=lambda i: abs_t_err_min[i], reverse=True)

    high_conf_wrong_indices = [
        int(i)
        for i in danger_indices
        if (raw_cycle_prob[i] >= HIGH_CONF_WRONG_CYCLE_THRESHOLD) and (not cycle_correct[i])
    ]
    high_conf_wrong_indices = sorted(high_conf_wrong_indices, key=lambda i: abs_t_err_min[i], reverse=True)

    true_cycle_counts = np.bincount(true_cycle[high_conf_wrong_indices], minlength=MAX_CYCLES)
    raw_cycle_counts = np.bincount(raw_cycle_pred[high_conf_wrong_indices], minlength=MAX_CYCLES)
    offset_values = cycle_pred[high_conf_wrong_indices] - true_cycle[high_conf_wrong_indices]
    unique_offsets, offset_counts = np.unique(offset_values, return_counts=True)

    with open(cycle_case_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "true_dc_km",
                "pred_dc_km",
                "true_tc_s",
                "pred_tc_s",
                "true_k",
                "pred_k",
                "abs_k_error",
                "abs_time_error_min",
                "true_cycle",
                "raw_pred_cycle",
                "pred_cycle",
                "refined_applied",
                "suspicious_physical_recheck",
                "local_physical_refined",
                "local_physical_rank",
                "cycle_offset",
                "refined_proxy_distance_km",
                "raw_cycle_prob",
                "pred_cycle_prob",
                "true_cycle_prob",
                "top2_cycle_1",
                "top2_cycle_2",
                "true_cycle_in_topk",
                "true_cycle_topk_rank",
                "true_cycle_topk_distance_km",
                "top1_cycle",
                "top1_distance_km",
                "top2_physical_cycle",
                "top2_distance_km",
                "top3_cycle",
                "top3_distance_km",
                "top4_cycle",
                "top4_distance_km",
                "top5_cycle",
                "top5_distance_km",
                "true_phase",
                "pred_phase",
                "abs_phase_error",
                "danger_prob",
            ]
        )
        for i in hard_indices:
            writer.writerow(
                [
                    i,
                    float(Y_test[i, 0]),
                    float(d_pred_km[i]),
                    float(true_t_s[i]),
                    float(pred_t_s[i]),
                    float(true_k[i]),
                    float(pred_k[i]),
                    float(abs_k_err[i]),
                    float(abs_t_err_min[i]),
                    int(true_cycle[i]),
                    int(raw_cycle_pred[i]),
                    int(cycle_pred[i]),
                    int(refined_applied[i]),
                    int(suspicious_recheck[i]),
                    int(local_refined_used[i]),
                    int(local_refined_rank[i]),
                    int(cycle_pred[i] - true_cycle[i]),
                    float(refined_proxy_distance[i]),
                    float(raw_cycle_prob[i]),
                    float(cycle_prob[i, cycle_pred[i]]),
                    float(true_cycle_prob[i]),
                    int(top2_cycle[i, 0]),
                    int(top2_cycle[i, 1]),
                    int(true_cycle_in_topk[i]),
                    int(true_cycle_topk_rank[i]),
                    float(true_cycle_topk_distance[i]),
                    int(top_cycles_np[i, 0]),
                    float(top_distances_np[i, 0]),
                    int(topk_candidate_value(top_cycles_np, i, 1, -1)),
                    float(topk_candidate_value(top_distances_np, i, 1, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 2, -1)),
                    float(topk_candidate_value(top_distances_np, i, 2, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 3, -1)),
                    float(topk_candidate_value(top_distances_np, i, 3, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 4, -1)),
                    float(topk_candidate_value(top_distances_np, i, 4, float("nan"))),
                    float(true_phase[i]),
                    float(phase_pred_np[i]),
                    float(abs_phase_err[i]),
                    float(danger_prob[i]),
                ]
            )

    high_conf_wrong_path = os.path.join("logs", "high_conf_wrong_cycle_cases.csv")
    with open(high_conf_wrong_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "true_dc_km",
                "pred_dc_km",
                "true_tc_s",
                "pred_tc_s",
                "true_k",
                "pred_k",
                "abs_k_error",
                "abs_time_error_min",
                "true_cycle",
                "raw_pred_cycle",
                "pred_cycle",
                "cycle_offset",
                "refined_applied",
                "suspicious_physical_recheck",
                "local_physical_refined",
                "local_physical_rank",
                "raw_cycle_prob",
                "pred_cycle_prob",
                "true_cycle_prob",
                "top2_cycle_1",
                "top2_cycle_2",
                "true_cycle_in_topk",
                "true_cycle_topk_rank",
                "true_cycle_topk_distance_km",
                "top1_cycle",
                "top1_distance_km",
                "top2_physical_cycle",
                "top2_distance_km",
                "top3_cycle",
                "top3_distance_km",
                "top4_cycle",
                "top4_distance_km",
                "top5_cycle",
                "top5_distance_km",
                "true_phase",
                "pred_phase",
                "abs_phase_error",
                "refined_proxy_distance_km",
                "danger_prob",
            ]
        )
        for i in high_conf_wrong_indices:
            writer.writerow(
                [
                    i,
                    float(Y_test[i, 0]),
                    float(d_pred_km[i]),
                    float(true_t_s[i]),
                    float(pred_t_s[i]),
                    float(true_k[i]),
                    float(pred_k[i]),
                    float(abs_k_err[i]),
                    float(abs_t_err_min[i]),
                    int(true_cycle[i]),
                    int(raw_cycle_pred[i]),
                    int(cycle_pred[i]),
                    int(cycle_pred[i] - true_cycle[i]),
                    int(refined_applied[i]),
                    int(suspicious_recheck[i]),
                    int(local_refined_used[i]),
                    int(local_refined_rank[i]),
                    float(raw_cycle_prob[i]),
                    float(cycle_prob[i, cycle_pred[i]]),
                    float(true_cycle_prob[i]),
                    int(top2_cycle[i, 0]),
                    int(top2_cycle[i, 1]),
                    int(true_cycle_in_topk[i]),
                    int(true_cycle_topk_rank[i]),
                    float(true_cycle_topk_distance[i]),
                    int(top_cycles_np[i, 0]),
                    float(top_distances_np[i, 0]),
                    int(topk_candidate_value(top_cycles_np, i, 1, -1)),
                    float(topk_candidate_value(top_distances_np, i, 1, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 2, -1)),
                    float(topk_candidate_value(top_distances_np, i, 2, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 3, -1)),
                    float(topk_candidate_value(top_distances_np, i, 3, float("nan"))),
                    int(topk_candidate_value(top_cycles_np, i, 4, -1)),
                    float(topk_candidate_value(top_distances_np, i, 4, float("nan"))),
                    float(true_phase[i]),
                    float(phase_pred_np[i]),
                    float(abs_phase_err[i]),
                    float(refined_proxy_distance[i]),
                    float(danger_prob[i]),
                ]
            )

    text_path = os.path.join("logs", "time_error_analysis.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Time error analysis on true danger samples\n")
        f.write(f"Danger sample count: {len(danger_err)}\n")
        f.write(f"Mean absolute time error: {np.mean(danger_err):.6f} min\n")
        f.write(f"Mean absolute K error: {np.mean(abs_k_err[danger_mask]):.6f}\n")
        f.write(f"Median absolute K error: {np.median(abs_k_err[danger_mask]):.6f}\n")
        f.write(f"Median absolute time error: {np.median(danger_err):.6f} min\n")
        f.write(f"P75 absolute time error: {np.percentile(danger_err, 75):.6f} min\n")
        f.write(f"P90 absolute time error: {np.percentile(danger_err, 90):.6f} min\n")
        f.write(f"P95 absolute time error: {np.percentile(danger_err, 95):.6f} min\n")
        f.write(f"Max absolute time error: {np.max(danger_err):.6f} min\n")
        f.write(f"Cycle accuracy: {np.mean(danger_cycle_correct) * 100.0:.4f}%\n")
        f.write(f"Conservative refine threshold: {CYCLE_REFINE_CONFIDENCE_THRESHOLD:.2f}\n")
        f.write(f"Suspicious raw-distance threshold: {SUSPICIOUS_REFINE_RAW_DISTANCE_KM:.2f} km\n")
        f.write(f"Suspicious candidate-distance threshold: {SUSPICIOUS_REFINE_CANDIDATE_DISTANCE_KM:.2f} km\n")
        f.write(f"Suspicious improvement ratio: {SUSPICIOUS_REFINE_IMPROVEMENT_RATIO:.2f}\n")
        f.write(f"Local physical refinement enabled: {int(LOCAL_PHYSICAL_REFINE)}\n")
        f.write(f"Local physical refinement danger-prob threshold: {LOCAL_REFINE_DANGER_PROB_THRESHOLD:.2f}\n")
        f.write(f"Local physical refinement top-k: {TOP_K_CYCLE_DIAGNOSTIC}\n")
        f.write(f"Local physical refinement window fraction: {LOCAL_REFINE_WINDOW_FRACTION:.2f}\n")
        f.write(
            f"Refine applied on danger samples: {int(np.sum(danger_refined_applied))} "
            f"({np.mean(danger_refined_applied) * 100.0:.2f}%)\n"
        )
        f.write(
            f"Suspicious physical recheck on danger samples: {int(np.sum(danger_suspicious_recheck))} "
            f"({np.mean(danger_suspicious_recheck) * 100.0:.2f}%)\n"
        )
        f.write(
            f"Local physical refinement on danger samples: {int(np.sum(danger_local_refined))} "
            f"({np.mean(danger_local_refined) * 100.0:.2f}%)\n"
        )
        f.write(f"Hard cycle/error case count: {len(hard_indices)}\n")
        f.write(f"Hard cycle/error case CSV: {cycle_case_path}\n")
        f.write(f"High-confidence wrong-cycle threshold: {HIGH_CONF_WRONG_CYCLE_THRESHOLD:.2f}\n")
        f.write(f"High-confidence wrong-cycle count: {len(high_conf_wrong_indices)}\n")
        f.write(f"High-confidence wrong-cycle CSV: {high_conf_wrong_path}\n")
        if len(hard_indices) > 0:
            hard_true_in_topk = true_cycle_in_topk[hard_indices]
            f.write(
                f"Hard cycle/error true cycle in top-{TOP_K_CYCLE_DIAGNOSTIC}: "
                f"{int(np.sum(hard_true_in_topk))}/{len(hard_indices)} "
                f"({np.mean(hard_true_in_topk) * 100.0:.2f}%)\n"
            )
        if len(high_conf_wrong_indices) > 0:
            hc_err = abs_t_err_min[high_conf_wrong_indices]
            hc_dc = Y_test[high_conf_wrong_indices, 0]
            hc_phase_err = abs_phase_err[high_conf_wrong_indices]
            hc_true_tc_h = true_t_s[high_conf_wrong_indices] / 3600.0
            hc_true_in_topk = true_cycle_in_topk[high_conf_wrong_indices]
            f.write(f"High-confidence wrong mean time error: {np.mean(hc_err):.6f} min\n")
            f.write(f"High-confidence wrong median time error: {np.median(hc_err):.6f} min\n")
            f.write(f"High-confidence wrong mean Dc: {np.mean(hc_dc):.6f} km\n")
            f.write(f"High-confidence wrong median Dc: {np.median(hc_dc):.6f} km\n")
            f.write(f"High-confidence wrong mean phase error: {np.mean(hc_phase_err):.6f}\n")
            f.write(f"High-confidence wrong median true Tc: {np.median(hc_true_tc_h):.6f} h\n")
            f.write(
                f"High-confidence wrong true cycle in top-{TOP_K_CYCLE_DIAGNOSTIC}: "
                f"{int(np.sum(hc_true_in_topk))}/{len(high_conf_wrong_indices)} "
                f"({np.mean(hc_true_in_topk) * 100.0:.2f}%)\n"
            )
            f.write("High-confidence wrong true-cycle top counts:\n")
            for cyc in np.argsort(-true_cycle_counts)[:8]:
                if true_cycle_counts[cyc] > 0:
                    f.write(f"  K={int(cyc)}: {int(true_cycle_counts[cyc])}\n")
            f.write("High-confidence wrong raw-pred-cycle top counts:\n")
            for cyc in np.argsort(-raw_cycle_counts)[:8]:
                if raw_cycle_counts[cyc] > 0:
                    f.write(f"  K={int(cyc)}: {int(raw_cycle_counts[cyc])}\n")
            f.write("High-confidence wrong cycle-offset counts:\n")
            for off, cnt in sorted(zip(unique_offsets, offset_counts), key=lambda item: -item[1]):
                f.write(f"  offset={int(off)}: {int(cnt)}\n")
        if len(correct_err) > 0:
            f.write(f"Mean error when cycle is correct: {np.mean(correct_err):.6f} min\n")
            f.write(f"Median error when cycle is correct: {np.median(correct_err):.6f} min\n")
        if len(wrong_err) > 0:
            f.write(f"Mean error when cycle is wrong: {np.mean(wrong_err):.6f} min\n")
            f.write(f"Median error when cycle is wrong: {np.median(wrong_err):.6f} min\n")
        f.write("\nError bins in minutes:\n")
        for label, count in zip(labels, bin_counts):
            f.write(f"{label}: {count} ({count / max(len(danger_err), 1) * 100.0:.2f}%)\n")

    if HAS_MATPLOTLIB:
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].bar(labels, bin_counts, color="#4c78a8")
        axes[0].set_xlabel("Absolute TCA error (min)")
        axes[0].set_ylabel("Danger sample count")
        axes[0].set_title("Time Error Distribution")
        axes[0].grid(axis="y", alpha=0.25)

        box_data = []
        box_labels = []
        if len(correct_err) > 0:
            box_data.append(correct_err)
            box_labels.append("K correct")
        if len(wrong_err) > 0:
            box_data.append(wrong_err)
            box_labels.append("K wrong")
        axes[1].boxplot(box_data, tick_labels=box_labels, showfliers=False)
        axes[1].set_ylabel("Absolute TCA error (min)")
        axes[1].set_title("Error by Cycle Prediction")
        axes[1].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "time_error_distribution.png"))
        plt.close(fig)

    return text_path, csv_path, cycle_case_path


print("\n开始训练 Transformer 分类 + 回归模型...")
os.makedirs("logs", exist_ok=True)
metrics_path = os.path.join("logs", "train_metrics.csv")
demo_path = os.path.join("logs", "prediction_demo.txt")
summary_path = os.path.join("logs", "summary.txt")
test_summary_path = os.path.join("logs", "test_summary.txt")

with open(metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "epoch",
            "val_cls_acc",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_false_alarm",
            "val_miss_rate",
            "danger_d_mae_km",
            "danger_t_mae_min",
            "danger_k_mae",
            "danger_cycle_acc",
            "danger_cycle_mae",
            "danger_phase_mae",
            "val_cls_loss",
            "val_dist_loss",
            "val_cycle_loss",
            "val_phase_loss",
            "best_epoch",
        ]
    )

best_score = float("inf")
best_epoch = 0
best_params = params
stale_epochs = 0

for epoch in range(1, EPOCHS + 1):
    batch_perm = np.random.permutation(len(X_train))

    for b in range(num_batches):
        idx = batch_perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        params, opt_state, _, _ = train_step(
            params,
            opt_state,
            jnp.array(X_train[idx]),
            jnp.array(d_train_norm[idx]),
            jnp.array(cycle_train[idx]),
            jnp.array(phase_train[idx]),
            jnp.array(danger_train[idx]),
        )

    vm = val_metrics(params)
    score = float(
        (1.0 - vm["f1"]) * 1000.0
        + vm["false_alarm"] * 300.0
        + vm["miss_rate"] * 500.0
        + vm["d_mae"] * 10.0
        + vm["t_mae_min"]
    )

    if score < best_score:
        best_score = score
        best_epoch = epoch
        best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
        stale_epochs = 0
    else:
        stale_epochs += 1

    if epoch == 1 or epoch % 10 == 0 or stale_epochs == 0:
        print(
            f"轮次 {epoch:03d} | 分类准确率: {float(vm['cls_acc']) * 100:.2f}% "
            f"| 危险召回率: {float(vm['recall']) * 100:.2f}% | 误报率: {float(vm['false_alarm']) * 100:.2f}% "
            f"| D_MAE: {float(vm['d_mae']):.2f} km | T_MAE: {float(vm['t_mae_min']):.1f} min "
            f"| 周期准确率: {float(vm['cycle_acc']) * 100:.2f}% | Phase_MAE: {float(vm['phase_mae']):.4f} "
            f"| 最佳轮次: {best_epoch:03d}"
        )

    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                float(vm["cls_acc"]),
                float(vm["precision"]),
                float(vm["recall"]),
                float(vm["f1"]),
                float(vm["false_alarm"]),
                float(vm["miss_rate"]),
                float(vm["d_mae"]),
                float(vm["t_mae_min"]),
                float(vm["k_mae"]),
                float(vm["cycle_acc"]),
                float(vm["cycle_mae"]),
                float(vm["phase_mae"]),
                float(vm["loss_cls"]),
                float(vm["loss_d"]),
                float(vm["loss_cycle"]),
                float(vm["loss_phase"]),
                best_epoch,
            ]
        )

    if stale_epochs >= PATIENCE:
        print(f"早停触发：当前第 {epoch} 轮，最佳轮次为第 {best_epoch} 轮。")
        break

params = best_params
val_m = val_metrics(params)
test_m = test_metrics(params)


def write_metric_block(f, name, m):
    mf = to_float_dict(m)
    f.write(f"\n{name} results:\n")
    f.write(f"Accuracy: {mf['cls_acc'] * 100:.2f}%\n")
    f.write(f"Precision: {mf['precision'] * 100:.2f}%\n")
    f.write(f"Danger recall: {mf['recall'] * 100:.2f}%\n")
    f.write(f"F1: {mf['f1']:.4f}\n")
    f.write(f"False alarm rate: {mf['false_alarm'] * 100:.2f}%\n")
    f.write(f"Miss rate: {mf['miss_rate'] * 100:.2f}%\n")
    f.write(f"Danger D_MAE: {mf['d_mae']:.6f} km\n")
    f.write(f"Danger T_MAE_from_K: {mf['t_mae_min']:.6f} min\n")
    f.write(f"Danger K_MAE: {mf['k_mae']:.6f}\n")
    f.write(f"Cycle accuracy: {mf['cycle_acc'] * 100:.2f}%\n")
    f.write(f"Cycle_MAE: {mf['cycle_mae']:.6f}\n")
    f.write(f"Phase_MAE: {mf['phase_mae']:.6f}\n")


with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Transformer 分类 + 回归模型训练总结\n")
    f.write("输入文件：data/X_coe.npy；标签文件：data/Y_reg.npy，其中标签为 Dc/Tc。\n")
    f.write(f"危险判定阈值：Dc <= {DANGER_DISTANCE_KM:.1f} km\n")
    f.write("说明：Dc/Tc 回归误差只在真实危险样本上统计；预测为安全时，Dc/Tc 不作为有效回归结果解释。\n")
    f.write(f"训练集：{len(X_train)} 条，危险样本比例 {danger_train.mean() * 100:.2f}%\n")
    f.write(f"验证集：{len(X_val)} 条，危险样本比例 {danger_val.mean() * 100:.2f}%\n")
    f.write(f"测试集：{len(X_test)} 条，危险样本比例 {danger_test.mean() * 100:.2f}%\n")
    f.write(f"最佳验证轮次：第 {best_epoch} 轮\n")
    f.write(
        f"损失权重：分类={CLASS_LOSS_WEIGHT}，距离={DIST_LOSS_WEIGHT}，周期={CYCLE_LOSS_WEIGHT}，"
        f"相位={PHASE_LOSS_WEIGHT}，物理一致性={PHYSICS_CONSISTENCY_WEIGHT}\n"
    )
    write_metric_block(f, "验证集", val_m)
    write_metric_block(f, "测试集", test_m)
    f.write("\n输出文件：\n")
    f.write(f"训练指标 CSV：{metrics_path}\n")
    f.write(f"预测样例：{demo_path}\n")

tf = to_float_dict(test_m)
with open(test_summary_path, "w", encoding="utf-8") as f:
    f.write("测试集最终结果\n")
    write_metric_block(f, "测试集", test_m)

print(
    f"\n测试集最终结果 | 分类准确率: {tf['cls_acc'] * 100:.2f}% "
    f"| 危险召回率: {tf['recall'] * 100:.2f}% | 误报率: {tf['false_alarm'] * 100:.2f}% "
    f"| D_MAE: {tf['d_mae']:.2f} km | T_MAE: {tf['t_mae_min']:.1f} min "
    f"| 周期准确率: {tf['cycle_acc'] * 100:.2f}% | Phase_MAE: {tf['phase_mae']:.4f}"
)

print("\n============ 最佳模型在测试集上的预测样例 ============")
with open(demo_path, "w", encoding="utf-8") as f:
    f.write("============ 最佳模型在测试集上的预测样例 ============\n")
    f.write(f"最佳验证轮次：第 {best_epoch} 轮\n\n")

demo_count = min(8, len(X_test))
for i in range(demo_count):
    x_input = jnp.array(X_test[i])
    pred_d_norm, cycle_logits, phase_ratio, _, danger_logit = model.apply(params, x_input)
    pred_prob = float(jax.nn.sigmoid(danger_logit))
    pred_danger = pred_prob >= 0.5
    pred_d_km = float(jnp.clip(pred_d_norm, 0.0, 1.0)) * DANGER_DISTANCE_KM
    refined_cycle, refined_t_s, _, _, _, _ = refine_cycle_by_physical_distance(
        jnp.array(X_test[i : i + 1]),
        cycle_logits[None, :],
        jnp.array([phase_ratio]),
        TOP_K_CYCLE_REFINE,
    )
    pred_cycle = int(refined_cycle[0])
    pred_phase = float(phase_ratio)
    pred_t_s = float(refined_t_s[0])
    pred_k = pred_t_s / float(T_ref_test[i])

    true_danger = bool(danger_test[i] >= 0.5)
    true_d_km = Y_test[i, 0]
    true_t_s = Y_test[i, 1]
    true_k = true_t_s / float(T_ref_test[i])
    true_cycle = int(cycle_test[i])
    true_phase = float(phase_test[i])

    line_true = (
        f"  [真实] 危险: {true_danger} | Dc: {true_d_km:8.2f} km | "
        f"Tc周期: {true_cycle:2d} | 相位: {true_phase:.3f} | Tc: {true_t_s:8.0f} s"
    )
    if pred_danger:
        line_pred = (
            f"  [预测] 危险: True ({pred_prob * 100:5.1f}%) | Dc: {pred_d_km:8.2f} km | "
            f"Tc周期: {pred_cycle:2d} | 相位: {pred_phase:.3f} | Tc: {pred_t_s:8.0f} s"
        )
    else:
        line_pred = (
            f"  [预测] 危险: False ({pred_prob * 100:5.1f}%) | "
            "Dc/Tc 不适用（距离头只用于危险样本内回归）"
        )

    print(f"样本 {i + 1}:")
    print(line_true)
    print(line_pred + "\n")
    with open(demo_path, "a", encoding="utf-8") as f:
        f.write(f"样本 {i + 1}:\n")
        f.write(line_true + "\n")
        f.write(line_pred + "\n\n")

time_error_text_path, time_error_csv_path, cycle_case_path = analyze_test_time_errors()

fig_dir = generate_report_figures()
if fig_dir:
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"报告图保存目录：{fig_dir}\n")
        f.write(f"时间误差分析：{time_error_text_path}\n")
        f.write(f"时间误差逐样本CSV：{time_error_csv_path}\n")
    print(f"报告图已保存到：{fig_dir}")
    print(f"时间误差分析已保存到：{time_error_text_path}")

print(f"训练完成。输出文件：{metrics_path}，{summary_path}，{demo_path}")
