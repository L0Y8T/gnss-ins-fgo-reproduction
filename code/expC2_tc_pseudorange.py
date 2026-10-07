"""expC2_tc_pseudorange.py —— 实验 C2：把观测模型换成真正的伪距（TC），迭代还无影响吗？

复现实现的是 LC（式 31）：GNSS 直接观测位置 z = p + v，对状态**线性**。
论文 4.3 节的消融实验用的是 TC（式 32）：原始伪距

        rho = || SV - X || + b + noise

对状态 X **非线性**。作者自己承认这条非线性「平凡（trivial）」。

本实验把这句话做成可测量的东西：
  C2a  同一套数据下，n_iter 从 1 扫到 30，TC 的误差曲线是否还像 LC 那样平？
  C2b  K=2 时 LC 是**恰定**的（robust kernel 数学上必定无效）。
       TC 下未知数 5K、方程数 3(K-1)+1+8K —— K=2 时是 10 对 25，**超定**。
       → robust kernel 会不会在 K=2 就复活？
  C2c  论文说「即使初始猜测差 100 m，视线方向依然算得准」。
       把窗口初值扰动 0/10/50/100 m，数它需要几次迭代收敛。

注意：本文件不算 EKF。要论证的是「非线性 → 迭代才有意义」，
      这是 FGO 自身的性质，不需要引入另一个估计器来搅局。
"""

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sim import Config, true_trajectory, horizontal_error, metrics
from robust import make_measurements_noise, kernel_weight
from expA_noise import RESULTS, write_csv

NX_TC = 5
IDX_E, IDX_N, IDX_VE, IDX_VN, IDX_B = 0, 1, 2, 3, 4

R_SV = 20_200_000.0          # GPS MEO 轨道半径 [m]，量级正确即可
ELEV = [15.0, 68.0, 35.0, 78.0, 22.0, 55.0, 42.0, 30.0]
AZIM = [15.0, 95.0, 170.0, 250.0, 320.0, 45.0, 210.0, 285.0]

NLOS_RATE = 0.12             # 与 LC 实验的离群比例一致
NLOS_BIAS = 25.0             # NLOS 是**正偏置**（多路径只会把伪距拉长）
CLOCK_TRUE = 1000.0          # 真实钟差 [m]，只做平移，不影响定位
CLOCK_DRIFT = 0.5            # 钟漂 [m/s]

DELTA = 2.0
CONV_TOL = 1e-6


# ── 星座几何 ──────────────────────────────────────────────────

def sat_enu():
    """返回卫星在本地 ENU 下的位置（8×3，单位 m）。"""
    out = []
    for el, az in zip(ELEV, AZIM):
        ce, se = math.cos(math.radians(el)), math.sin(math.radians(el))
        ca, sa = math.cos(math.radians(az)), math.sin(math.radians(az))
        out.append([R_SV * ce * sa, R_SV * ce * ca, R_SV * se])
    return np.array(out)


SATS = sat_enu()
UNIT = SATS / np.linalg.norm(SATS, axis=1, keepdims=True)


def gdop_horizontal():
    """水平 GDOP：由 [u_E, u_N, 1] 的设计矩阵算 (A^T A)^-1 的水平迹。"""
    A = np.hstack([UNIT[:, :2], np.ones((len(SATS), 1))])
    Q = np.linalg.inv(A.T @ A)
    return math.sqrt(Q[0, 0] + Q[1, 1])


# ── 伪距生成 ──────────────────────────────────────────────────

def make_pseudoranges(cfg, states, sigma_rho, nlos_rate=NLOS_RATE,
                      nlos_bias=NLOS_BIAS, seed=None):
    """由真值生成 8 颗星的伪距，含正偏置 NLOS 与高斯噪声。

    返回 (rho 列表, 每个历元的 NLOS 星号列表)。
    """
    rng = random.Random((cfg.seed if seed is None else seed) * 1000003 + 11)
    M = len(SATS)
    rho, bad_hist = [], []
    for k, s in enumerate(states):
        X = np.array([s["p"][0], s["p"][1], 0.0])
        b = CLOCK_TRUE + CLOCK_DRIFT * k
        r = np.linalg.norm(SATS - X, axis=1) + b

        bad = []
        if rng.random() < nlos_rate:
            n_bad = 1 if rng.random() < 0.6 else 2
            bad = rng.sample(range(M), n_bad)
            for j in bad:
                r[j] += nlos_bias
        bad_hist.append(bad)

        r = r + np.array([rng.gauss(0.0, sigma_rho) for _ in range(M)])
        rho.append(r)
    return rho, bad_hist


def make_case_tc(seed, n_epochs=150, sigma_rho=None):
    """一个种子下的完整算例：真值 + IMU（沿用 LC 的生成器）+ 伪距。"""
    cfg = Config()
    cfg.n_epochs = n_epochs
    cfg.seed = seed
    st = true_trajectory(cfg)

    # IMU 测量沿用原生成器，保证和 LC 实验是同一套运动学
    imu = make_measurements_noise(cfg, st, model="gaussian", outlier=False)["imu"]

    if sigma_rho is None:
        sigma_rho = cfg.gnss_sigma / gdop_horizontal()
    rho, bad = make_pseudoranges(cfg, st, sigma_rho, seed=seed)
    return cfg, st, imu, rho, bad, sigma_rho


# ── 因子图求解（numpy 实现）────────────────────────────────────

def _initial_window(meas_prev, imu, k, K, sigma_rho):
    """窗口内各历元的初值：位置/钟差用单历元线性化最小二乘，速度用 IMU 递推。"""
    return None  # 占位，实际在 solve 里做


def single_epoch_ls(rho_k, x0=None, n_iter=10):
    """单历元伪距定位（含钟差）。返回 [p_E, p_N, b]。"""
    M = len(SATS)
    if x0 is None:
        x0 = np.zeros(3)
    A = np.hstack([UNIT[:, :2], np.ones((M, 1))])
    x = np.array(x0, dtype=float)
    for _ in range(n_iter):
        X = np.array([x[0], x[1], 0.0])
        d = SATS - X
        D = np.linalg.norm(d, axis=1)
        r = rho_k - D - x[2]
        u = d / D[:, None]
        A = np.hstack([u[:, :2], -np.ones((M, 1))])
        dx = np.linalg.lstsq(A, -r, rcond=None)[0]
        x = x + dx
        if np.linalg.norm(dx) < 1e-9:
            break
    return x


def build_tc_window(cfg, X, rho_win, imu_win, sigma_rho, kernel, delta):
    """构造一个 TC 窗口的 H、r、w_base 与分组号。

    X 形状 (K, 5)。返回 (H, r, w_base, group)。
    分组口径：
      · 运动模型 / IMU 因子按 (2 行 = 一个 2D 残差组) 分组，与 expA 一致
      · 钟差因子 1 行一组
      · **每颗卫星的伪距各成一组**（论文式 32 就是这么写的：每历元每星一个因子）
    """
    K = X.shape[0]
    n = NX_TC * K
    dt = cfg.dt

    H, r, wb, grp = [], [], [], []
    gid = 0

    def push(row, res, base_w, g):
        H.append(row)
        r.append(res)
        wb.append(base_w)
        grp.append(g)

    w_pos = 1.0 / cfg.mm_position_sigma ** 2
    w_vel = 1.0 / cfg.mm_velocity_sigma ** 2
    w_imu = 1.0 / cfg.imu_velocity_sigma ** 2
    w_clk = 1.0 / 2.0 ** 2          # 钟差随机游走 sigma = 2 m
    w_rho = 1.0 / sigma_rho ** 2

    for g in range(K - 1):
        b0, b1 = NX_TC * g, NX_TC * (g + 1)
        for d, vd in ((IDX_E, IDX_VE), (IDX_N, IDX_VN)):
            row = np.zeros(n)
            row[b1 + d] = 1.0
            row[b0 + d] = -1.0
            row[b0 + vd] = -dt
            push(row, X[g + 1][d] - (X[g][d] + X[g][vd] * dt), w_pos, gid)
        gid += 1

    for g in range(K - 1):
        b0, b1 = NX_TC * g, NX_TC * (g + 1)
        for d in (IDX_VE, IDX_VN):
            row = np.zeros(n)
            row[b1 + d] = 1.0
            row[b0 + d] = -1.0
            push(row, X[g + 1][d] - (X[g][d] + imu_win[g][d - IDX_VE] * dt),
                 w_imu, gid)
        gid += 1

    for g in range(K - 1):
        b0, b1 = NX_TC * g, NX_TC * (g + 1)
        row = np.zeros(n)
        row[b1 + IDX_B] = 1.0
        row[b0 + IDX_B] = -1.0
        push(row, X[g + 1][IDX_B] - X[g][IDX_B], w_clk, gid)
        gid += 1

    for g in range(K):
        base = NX_TC * g
        P = np.array([X[g][IDX_E], X[g][IDX_N], 0.0])
        d = SATS - P
        D = np.linalg.norm(d, axis=1)
        for j in range(len(SATS)):
            row = np.zeros(n)
            row[base + IDX_E] = d[j, 0] / D[j]
            row[base + IDX_N] = d[j, 1] / D[j]
            row[base + IDX_B] = -1.0
            push(row, rho_win[g][j] - D[j] - X[g][IDX_B], w_rho, gid)
            gid += 1

    return np.array(H), np.array(r), np.array(wb), np.array(grp)


def group_starts(grp):
    """分组号是连续递增的（行按组顺序 push），所以组边界就是变化点。"""
    return np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])


def apply_kernel(r, wb, grp, kernel, delta, starts=None):
    """按组算 IRLS 权重，返回逐行权重（向量化）。"""
    if kernel == "l2":
        return wb
    if starts is None:
        starts = group_starts(grp)
    s = np.sqrt(np.add.reduceat(wb * r * r, starts))
    if kernel == "huber":
        f = np.where(s <= delta, 1.0, delta / np.maximum(s, 1e-300))
    else:
        f = 1.0 / (1.0 + (s / delta) ** 2)
    counts = np.diff(np.r_[starts, len(r)])
    return wb * np.repeat(f, counts)


_STARTS_CACHE = {}


def starts_for(K):
    """K 决定分组结构，故组边界可以按 K 缓存。"""
    if K not in _STARTS_CACHE:
        n = NX_TC * K
        grp = []
        for _ in range(K - 1):
            grp.extend([len(grp)] * 2)
        for _ in range(K - 1):
            grp.extend([len(grp)] * 2)
        for _ in range(K - 1):
            grp.extend([len(grp)] * 1)
        for _ in range(K * len(SATS)):
            grp.extend([len(grp)] * 1)
        _STARTS_CACHE[K] = group_starts(np.array(grp))
    return _STARTS_CACHE[K]


def gauss_newton_tc(cfg, X, rho_win, imu_win, sigma_rho, n_iter,
                    kernel="l2", delta=DELTA, tol=CONV_TOL):
    """反复迭代一个 TC 窗口。返回 (X, 迭代历史)。"""
    X = np.array(X, dtype=float)
    hist = []
    used = 0
    starts = starts_for(X.shape[0])
    dX = np.zeros(NX_TC * X.shape[0])
    for it in range(n_iter):
        H, r, wb, grp = build_tc_window(cfg, X, rho_win, imu_win,
                                        sigma_rho, kernel, delta)
        w = apply_kernel(r, wb, grp, kernel, delta, starts)
        cost = float(np.sum(w * r ** 2))
        hist.append(cost)

        Hw = H * w[:, None]
        A = H.T @ Hw
        g = Hw.T @ r
        try:
            dX = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, -g, rcond=None)[0]
        X = X + dX.reshape(-1, NX_TC)
        used = it + 1
        if np.linalg.norm(dX) < tol:
            break
    H, r, wb, grp = build_tc_window(cfg, X, rho_win, imu_win, sigma_rho,
                                    kernel, delta)
    w = apply_kernel(r, wb, grp, kernel, delta, starts)
    hist.append(float(np.sum(w * r ** 2)))
    return X, {"cost_history": hist, "n_iter_used": used,
               "final_step": float(np.linalg.norm(dX))}


def run_fgo_tc(cfg, st, imu, rho, sigma_rho, window=1, n_iter=1,
               kernel="l2", delta=DELTA, init_perturb=0.0):
    """滑动窗口 TC-FGO。"""
    K = max(1, window)
    n = cfg.n_epochs
    dt = cfg.dt

    X_hist = [None] * n
    used_hist = []
    p0 = single_epoch_ls(rho[0])

    for k in range(n):
        g0 = max(0, k - K + 1)
        idxs = list(range(g0, k + 1))
        X = np.zeros((len(idxs), NX_TC))

        for j, g in enumerate(idxs):
            if g < k and X_hist[g] is not None:
                X[j] = X_hist[g]
            else:
                if g == 0:
                    xk = single_epoch_ls(rho[0])
                else:
                    xk = single_epoch_ls(rho[g], x0=X[j - 1][[IDX_E, IDX_N, IDX_B]])

                X[j][IDX_E] = xk[0]
                X[j][IDX_N] = xk[1]
                X[j][IDX_B] = xk[2]
                X[j][IDX_VE] = X[j - 1][IDX_VE] + imu[g - 1][0] * dt if j > 0 else 0.0
                X[j][IDX_VN] = X[j - 1][IDX_VN] + imu[g - 1][1] * dt if j > 0 else 0.0

        if init_perturb > 0.0:
            rngp = random.Random(999 + k)
            for j in range(len(idxs)):
                X[j][IDX_E] += rngp.uniform(-1, 1) * init_perturb
                X[j][IDX_N] += rngp.uniform(-1, 1) * init_perturb

        rho_win = [rho[g] for g in idxs]
        imu_win = [imu[g] for g in range(g0, k)]

        X, info = gauss_newton_tc(cfg, X, rho_win, imu_win, sigma_rho,
                                  n_iter, kernel, delta)
        used_hist.append(info["n_iter_used"])

        for j, g in enumerate(idxs):
            X_hist[g] = X[j].copy()

    errors = [horizontal_error([X_hist[k][IDX_E], X_hist[k][IDX_N]], st[k]["p"])
              for k in range(n)]
    return {"states": X_hist, "errors": errors,
            "n_iter_used": used_hist}


def summarize(errors, burn_in=5, cat_thresh=15.0):
    m = metrics(errors, burn_in=burn_in)
    e = errors[burn_in:]
    m["cat_rate"] = sum(1 for x in e if x > cat_thresh) / float(len(e))
    return m


def mean_of(v):
    return sum(v) / len(v)


def collect(model_desc, seeds, K, kernel, n_iter, delta=DELTA):
    acc, ps = [], []
    for s in seeds:
        cfg, st, imu, rho, bad, sr = make_case_tc(s)
        r = run_fgo_tc(cfg, st, imu, rho, sr, window=K, n_iter=n_iter,
                       kernel=kernel, delta=delta)
        m = summarize(r["errors"])
        acc.append(m)
        ps.append(m["mean"])
    return ({k: mean_of([a[k] for a in acc]) for k in acc[0]}, ps)


# ── 主体 ──────────────────────────────────────────────────────

SEEDS = [20260919 + s * 7919 for s in range(6)]


def part_a():
    print("=" * 96)
    print("C2a：TC（原始伪距，非线性）下，迭代次数还有没有影响？")
    print("=" * 96)
    gd = gdop_horizontal()
    print(f"  星座 {len(SATS)} 颗｜水平 GDOP = {gd:.3f}")
    print(f"  标定：令 sigma_rho = 6.0/GDOP = {6.0/gd:.3f} m，"
          f"使单历元水平定位精度与 LC 的 6 m 对齐")
    print(f"  NLOS：{NLOS_RATE:.0%} 的历元，随机 1-2 颗星 +{NLOS_BIAS:.0f} m 正偏置")
    print()
    print("  对照（LC，来自实验 A3）：L2 与 robust 的全部 n_iter 结果在浮点上完全相同")
    print()

    rows = [["window", "kernel", "n_iter", "mean_m", "rmse_m", "p95_m", "max_m",
             "cat_rate_gt15m"]]
    grid = {}
    for K in (2, 3, 10, 20):
        for kernel in ("l2", "huber", "cauchy"):
            vals = []
            for ni in (1, 2, 3, 5, 10, 30):
                avg, _ = collect(None, SEEDS, K, kernel, ni)
                grid[(K, kernel, ni)] = avg
                vals.append((ni, avg["mean"]))
                rows.append([K, kernel, ni] + [f"{avg[f]:.4f}" for f in
                                               ("mean", "rmse", "p95", "max",
                                                "cat_rate")])
            lo, hi = min(v for _, v in vals), max(v for _, v in vals)
            best = min(vals, key=lambda t: t[1])
            print(f"  K={K:<3} {kernel:<7} " +
                  "  ".join(f"i={ni}:{v:6.3f}" for ni, v in vals))
            print(f"          极差 {hi-lo:6.3f} m（{100*(hi-lo)/lo:5.1f}%）"
                  f"   最优 n_iter={best[0]}   "
                  f"{'<- 迭代有影响' if 100*(hi-lo)/lo > 2.0 else '<- 迭代无影响'}")
        print()
    write_csv(os.path.join(RESULTS, "expC2a_tc_iteration.csv"), rows)
    return grid


def part_b(grid):
    print("=" * 96)
    print("C2b：K=2 时 robust kernel 在 TC 下会不会复活？")
    print("=" * 96)
    print("  LC 下 K=2 是恰定的：未知数 8、约束 8 -> 残差恒为 0 -> robust 权重恒等于基准，")
    print("  实验 A0 实测三者的状态差 = 1.7e-13 m，即**数学上必定无效**。")
    print()
    K = 2
    n_unk = NX_TC * K
    n_row = 2 * (K - 1) + 2 * (K - 1) + (K - 1) + len(SATS) * K
    print(f"  TC 下 K=2：未知数 {n_unk}，方程数 {n_row} -> 冗余 {n_row - n_unk}")
    print("  即 K=2 就已经超定，robust kernel 有东西可以降权。")
    print()
    print(f"  {'kernel':<9} {'均值 (m)':>10} {'P95 (m)':>9} {'超15m':>8} "
          f"{'相对 L2 改善':>13}")
    print("  " + "-" * 60)
    base = grid[(K, "l2", 1)]["mean"]
    rows = [["window", "kernel", "n_iter", "mean_m", "p95_m", "cat_rate_gt15m",
             "gain_vs_L2_pct"]]
    for kernel in ("l2", "huber", "cauchy"):
        a = grid[(K, kernel, 1)]
        gain = 100.0 * (base - a["mean"]) / base
        print(f"  {kernel:<9} {a['mean']:>10.4f} {a['p95']:>9.3f} "
              f"{a['cat_rate']:>7.1%} {gain:>+12.1f}%")
        rows.append([K, kernel, 1, f"{a['mean']:.4f}", f"{a['p95']:.4f}",
                     f"{a['cat_rate']:.4f}", f"{gain:.2f}"])
    print()
    write_csv(os.path.join(RESULTS, "expC2b_tc_k2_robust.csv"), rows)


def part_c():
    print("=" * 96)
    print("C2c：论文说「初值差 100 m，视线方向依然算得准」—— 实测需要几次迭代")
    print("=" * 96)
    print("  做法：把每个窗口内所有历元的位置初值随机扰动 d，看收敛所需迭代次数。")
    print("  收敛判据：单次增量范数 ||dX|| < 1e-6")
    print()
    print(f"  {'初值扰动 (m)':>12} {'平均迭代次数':>13} {'<2 次的窗口占比':>16} "
          f"{'均值误差 (m)':>13}")
    print("  " + "-" * 62)
    rows = [["init_perturb_m", "mean_iters_used", "frac_converged_in_1_iter",
             "mean_error_m"]]
    for d in (0.0, 10.0, 50.0, 100.0, 500.0):
        used_all, errs = [], []
        for s in SEEDS:
            cfg, st, imu, rho, bad, sr = make_case_tc(s)
            r = run_fgo_tc(cfg, st, imu, rho, sr, window=10, n_iter=30,
                           kernel="l2", init_perturb=d)
            used_all.extend(r["n_iter_used"])
            errs.append(summarize(r["errors"])["mean"])
        frac = sum(1 for u in used_all if u <= 1) / len(used_all)
        mi = mean_of(used_all)
        print(f"  {d:>12.0f} {mi:>13.2f} {frac:>15.1%} {mean_of(errs):>13.4f}")
        rows.append([d, f"{mi:.3f}", f"{frac:.4f}", f"{mean_of(errs):.4f}"])
    print()
    print("  若「100 m 扰动」与「0 m 扰动」的迭代次数差别很小，"
          "则论文「非线性平凡」的说法在这套配置下成立。")
    print()
    write_csv(os.path.join(RESULTS, "expC2c_trivial_nonlinearity.csv"), rows)


def main():
    grid = part_a()
    part_b(grid)
    part_c()


if __name__ == "__main__":
    main()
