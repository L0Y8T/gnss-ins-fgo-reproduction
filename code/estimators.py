"""estimators.py: EKF 与 FGO 估计器

EKF 逐历元递归；FGO 按滑动窗口批处理。
两者共用完全相同的模型与噪声参数，保证对比公平。
"""

import sys
import math

from la import (zeros, identity, diag, col, flatten_col, matmul, transpose,
                add, sub, scale, gauss_solve, weighted_normal_equations, norm2)
from sim import horizontal_error, metrics

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NX = 4
IDX_E, IDX_N, IDX_VE, IDX_VN = 0, 1, 2, 3

def F_motion(dt):
    return [[1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

def F_imu(dt):
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

def Q_motion(cfg):
    """运动模型因子的协方差 Σ^MM（对应 Wen 2021 式(22)）。"""

    sp = cfg.mm_position_sigma
    sv = cfg.mm_velocity_sigma
    return diag([sp ** 2, sp ** 2, sv ** 2, sv ** 2])

def sigma_motion(cfg):
    return (cfg.mm_position_sigma, cfg.mm_velocity_sigma)

def Q_imu(cfg):
    """IMU 因子的协方差 Σ^INS（Wen 2021 式(27)）。"""
    s_v = cfg.imu_velocity_sigma

    # 1e6 会让权重掉到 1e-12，没必要这么极端，1e4 已经足够大。
    big = 1e4
    return diag([big, big, s_v ** 2, s_v ** 2])

def R_gnss(cfg):
    return diag([cfg.r_gnss, cfg.r_gnss])

def run_ekf(cfg, meas, x0=None):
    """跑一遍 EKF。"""
    dt = cfg.dt
    Qm = Q_motion(cfg)
    Qi = Q_imu(cfg)
    R = diag([cfg.r_gnss, cfg.r_gnss])

    if x0 is None:

        x = col([meas["gnss"][0][0], meas["gnss"][0][1], 0.0, 0.0])
    else:
        x = col(list(x0))

    P = diag([cfg.r_gnss, cfg.r_gnss, 100.0, 100.0])

    F = F_motion(dt)

    G = [[0.0, 0.0],
         [0.0, 0.0],
         [dt, 0.0],
         [0.0, dt]]

    states, errors = [], []

    for k in range(cfg.n_epochs):

        if k > 0:

            Fx = matmul(F, x)
            Gu = matmul(G, col(meas["imu"][k]))
            x_pred = add(Fx, Gu)

            FP = matmul(F, P)
            FPFT = matmul(FP, transpose(F))

            sig_v2 = cfg.imu_velocity_sigma ** 2
            Gu_cov = [[G[i][0] * sig_v2 * G[j][0] +
                       G[i][1] * sig_v2 * G[j][1]
                       for j in range(NX)] for i in range(NX)]
            P_pred = add(add(FPFT, Qm), Gu_cov)
        else:
            x_pred = x
            P_pred = P

        H = zeros(2, NX)
        H[0][0] = 1.0
        H[1][1] = 1.0

        z = col(meas["gnss"][k])
        y = sub(z, matmul(H, x_pred))
        HP = matmul(H, P_pred)
        S = add(matmul(HP, transpose(H)), R)
        K = matmul(matmul(P_pred, transpose(H)), _inverse_2x2(S))

        x = add(x_pred, matmul(K, y))

        KH = matmul(K, H)
        I_KH = sub(identity(NX), KH)
        P = matmul(I_KH, P_pred)

        states.append(flatten_col(x))
        errors.append(horizontal_error([x[0][0], x[1][0]], meas["true_p"][k]))

    return {"states": states, "errors": errors, "name": "EKF"}

def _inverse_2x2(S):
    """2×2 矩阵求逆的解析式（比通用高斯消元快，也更稳）。"""
    a, b = S[0][0], S[0][1]
    c, d = S[1][0], S[1][1]
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("2x2 矩阵奇异")
    return [[d / det, -b / det], [-c / det, a / det]]

def build_jacobian_and_residual(cfg, X_blocks, gnss_idx, gnss_z, imu_a,
                                v0_prior=True):
    """为一个滑动窗口构造雅可比矩阵 H 与残差向量 r。这是整个 FGO 的核心。"""
    K = len(X_blocks)
    n = NX * K
    dt = cfg.dt

    rows_H, rows_r, rows_w = [], [], []

    s_p, s_v = sigma_motion(cfg)
    s_imu_v = cfg.imu_velocity_sigma
    q_pos2 = s_p ** 2
    q_vel2 = s_v ** 2
    s_v2 = s_imu_v ** 2
    r_g2 = cfg.r_gnss

    # 这里不加 v_(g+1)=v_g 的软先验：那等于假设加速度恒为 0，会和 IMU 因子打架。
    for g in range(K - 1):
        base_g = NX * g
        base_n = NX * (g + 1)

        for d in (IDX_E, IDX_N):
            row = [0.0] * n
            row[base_n + d] += 1.0
            row[base_g + d] += -1.0
            row[base_g + IDX_VE + (d - IDX_E)] += -dt
            r_val = (X_blocks[g + 1][d]
                     - (X_blocks[g][d] + X_blocks[g][IDX_VE + (d - IDX_E)] * dt))
            rows_H.append(row)
            rows_r.append([r_val])
            rows_w.append(1.0 / q_pos2)

    for g in range(K - 1):
        base_g = NX * g
        base_n = NX * (g + 1)
        for d in (IDX_VE, IDX_VN):
            row = [0.0] * n
            row[base_n + d] += 1.0
            row[base_g + d] += -1.0
            r_val = X_blocks[g + 1][d] - (X_blocks[g][d] + imu_a[g][d - IDX_VE] * dt)
            rows_H.append(row)
            rows_r.append([r_val])
            rows_w.append(1.0 / s_v2)

    for g in range(K):
        eg = gnss_idx[g]
        if eg < 0:
            continue
        base = NX * g
        for d in (IDX_E, IDX_N):
            row = [0.0] * n
            row[base + d] += 1.0
            r_val = X_blocks[g][d] - gnss_z[eg][d - IDX_E]
            rows_H.append(row)
            rows_r.append([r_val])
            rows_w.append(1.0 / r_g2)

    if v0_prior and K == 1:
        var_v0 = 25.0
        for d in (IDX_VE, IDX_VN):
            row = [0.0] * n
            row[d] += 1.0
            rows_H.append(row)
            rows_r.append([X_blocks[0][d] - 0.0])
            rows_w.append(1.0 / var_v0)

    return rows_H, rows_r, rows_w

def fgo_solve_window(cfg, X, gnss_idx, gnss_z, imu_a, n_iter, verbose=False):
    """对一个窗口求解加权最小二乘（反复迭代 n_iter 次）。"""
    hist = []
    for it in range(n_iter):
        H, r, w = build_jacobian_and_residual(cfg, X, gnss_idx, gnss_z, imu_a)

        cost = sum(w[i] * r[i][0] ** 2 for i in range(len(r)))
        hist.append(cost)

        if verbose:
            print(f"      迭代 {it+1}: 代价 = {cost:.6e}")

        dX = flatten_col(weighted_normal_equations(H, w, r))

        for g in range(len(X)):
            for d in range(NX):
                X[g][d] += dX[NX * g + d]

        if norm2(dX) < 1e-10:
            if verbose:
                print(f"      -> 增量范数 {norm2(dX):.2e} < 1e-10，已收敛，提前退出")
            break

    H, r, w = build_jacobian_and_residual(cfg, X, gnss_idx, gnss_z, imu_a)
    hist.append(sum(w[i] * r[i][0] ** 2 for i in range(len(r))))
    return X, {"cost_history": hist, "n_iter_used": len(hist) - 1}

def run_fgo(cfg, meas, window=1, n_iter=1, verbose=False):
    """跑一遍滑动窗口 FGO。"""
    dt = cfg.dt
    K = max(1, window)
    n = cfg.n_epochs

    X_hist = [None] * n
    cur = [meas["gnss"][0][0], meas["gnss"][0][1], 0.0, 0.0]

    gnss_idx_all = list(range(n))
    cost_hist = []

    for k in range(n):

        g0 = max(0, k - K + 1)
        idxs = list(range(g0, k + 1))

        if g0 == k:

            if k == 0:
                X = [[cur[0], cur[1], cur[2], cur[3]]]
            else:
                prev = X_hist[k - 1]
                X = [[prev[IDX_E] + prev[IDX_VE] * dt,
                      prev[IDX_N] + prev[IDX_VN] * dt,
                      prev[IDX_VE] + meas["imu"][k][0] * dt,
                      prev[IDX_VN] + meas["imu"][k][1] * dt]]
        else:

            X = [list(X_hist[g]) for g in idxs[:-1]]

            prev = X[-1]
            new = [prev[IDX_E] + prev[IDX_VE] * dt,
                   prev[IDX_N] + prev[IDX_VN] * dt,
                   prev[IDX_VE] + meas["imu"][k][0] * dt,
                   prev[IDX_VN] + meas["imu"][k][1] * dt]
            X.append(new)

        gnss_idx = [g for g in range(g0, k + 1)]
        X, info = fgo_solve_window(cfg, X, gnss_idx, meas["gnss"], meas["imu"],
                                   n_iter, verbose=verbose)
        cost_hist.extend(info["cost_history"])

        for j, g in enumerate(idxs):
            X_hist[g] = list(X[j])
        cur = X_hist[k]

    states = X_hist
    errors = [horizontal_error([states[k][0], states[k][1]], meas["true_p"][k])
              for k in range(n)]

    return {"states": states, "errors": errors,
            "name": f"FGO(K={window}, iter={n_iter})",
            "window": window, "n_iter": n_iter,
            "cost_history": cost_hist}

def prepare_meas(cfg, states_true, meas):
    """把真值位置挂到 meas 上，供评估使用（估计器不该读它，只用于算误差）。"""
    meas = dict(meas)
    meas["true_p"] = [s["p"] for s in states_true]
    return meas

