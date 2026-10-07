"""expA_noise.py —— 实验 A：非高斯噪声下，FGO 的优势还成立吗？

文伟松 2026-09-28 指出：这份仿真的噪声不是高斯。
本实验把这句话做成受控实验：

  固定方差、只改形状（同方差异形状），把「非高斯性」从「噪声变大了」里分离出来。
  然后看 4 个估计器：EKF / FGO(L2) / FGO(Huber) / FGO(Cauchy)。

关键对照是 L2 与 robust kernel 的差别：
  最小二乘对大残差按平方惩罚，一个离群点就能把解拖走。
  Huber / Cauchy 把大残差降权，理论上应当恢复精度——
  但前提是系统有冗余（K>=3），因为 K=2 恰定时残差恒为 0，没有东西可以降权。
"""

import os
import sys
import math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sim import Config, true_trajectory, horizontal_error, metrics, percentile
from estimators import (NX, IDX_E, IDX_N, IDX_VE, IDX_VN, sigma_motion,
                        run_ekf)
from robust import (MODEL_LABEL, KERNEL_LABEL, kernel_weight,
                    make_measurements_noise, effective_sigma)
from la import zeros, flatten_col, norm2, weighted_normal_equations

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

N_EPOCHS = 150
N_SEEDS = 6
WINDOWS = [2, 3, 10, 20]
N_ITER = 15
DELTA = 2.0

# ── 带 robust kernel 的因子图求解 ─────────────────────────────

def build_H_r_w(cfg, X, gnss_idx, gnss_z, imu_a, kernel="l2", delta=DELTA):
    """按 *因子* 分组构造 H、r、w，权重可由 kernel 依残差重算。

    kernel="l2" 时与 estimators.build_jacobian_and_residual 完全等价
    （行顺序与权重逐位相同，见自检）。
    """
    K = len(X)
    n = NX * K
    dt = cfg.dt

    s_p, s_v = sigma_motion(cfg)
    w_pos = 1.0 / s_p ** 2
    w_vel = 1.0 / s_v ** 2
    w_imu = 1.0 / cfg.imu_velocity_sigma ** 2
    w_gnss = 1.0 / cfg.r_gnss

    groups = []

    for g in range(K - 1):
        base_g, base_n = NX * g, NX * (g + 1)
        specs = []
        for d in (IDX_E, IDX_N):
            row = [0.0] * n
            row[base_n + d] += 1.0
            row[base_g + d] += -1.0
            row[base_g + IDX_VE + (d - IDX_E)] += -dt
            r_val = (X[g + 1][d]
                     - (X[g][d] + X[g][IDX_VE + (d - IDX_E)] * dt))
            specs.append((row, r_val))
        groups.append((w_pos, specs))

    for g in range(K - 1):
        base_g, base_n = NX * g, NX * (g + 1)
        specs = []
        for d in (IDX_VE, IDX_VN):
            row = [0.0] * n
            row[base_n + d] += 1.0
            row[base_g + d] += -1.0
            r_val = X[g + 1][d] - (X[g][d] + imu_a[g][d - IDX_VE] * dt)
            specs.append((row, r_val))
        groups.append((w_imu, specs))

    for g in range(K):
        eg = gnss_idx[g]
        if eg < 0:
            continue
        base = NX * g
        specs = []
        for d in (IDX_E, IDX_N):
            row = [0.0] * n
            row[base + d] += 1.0
            r_val = X[g][d] - gnss_z[eg][d - IDX_E]
            specs.append((row, r_val))
        groups.append((w_gnss, specs))

    if K == 1:
        specs = []
        for d in (IDX_VE, IDX_VN):
            row = [0.0] * n
            row[d] += 1.0
            specs.append((row, X[0][d] - 0.0))
        groups.append((1.0 / 25.0, specs))

    H, r, w = [], [], []
    for w_base, specs in groups:
        ww = kernel_weight([rv for _, rv in specs], w_base, kernel, delta)
        for row, rv in specs:
            H.append(row)
            r.append([rv])
            w.append(ww)
    return H, r, w

def solve_window(cfg, X, gnss_idx, gnss_z, imu_a, n_iter,
                 kernel="l2", delta=DELTA):
    """反复迭代求解一个窗口，返回代价历史与每步增量范数。"""
    hist, deltas = [], []
    for _ in range(n_iter):
        H, r, w = build_H_r_w(cfg, X, gnss_idx, gnss_z, imu_a, kernel, delta)
        hist.append(sum(w[i] * r[i][0] ** 2 for i in range(len(r))))

        dX = flatten_col(weighted_normal_equations(H, w, r))
        nd = norm2(dX)
        deltas.append(nd)

        for g in range(len(X)):
            for d in range(NX):
                X[g][d] += dX[NX * g + d]

        if nd < 1e-10:
            break

    H, r, w = build_H_r_w(cfg, X, gnss_idx, gnss_z, imu_a, kernel, delta)
    hist.append(sum(w[i] * r[i][0] ** 2 for i in range(len(r))))
    return X, {"cost_history": hist, "delta_norms": deltas,
               "n_iter_used": len(deltas)}

def run_fgo_kernel(cfg, meas, window=1, n_iter=1, kernel="l2", delta=DELTA):
    """带 robust kernel 的滑动窗口 FGO。结构照抄 estimators.run_fgo。"""
    dt = cfg.dt
    K = max(1, window)
    n = cfg.n_epochs

    X_hist = [None] * n
    cur = [meas["gnss"][0][0], meas["gnss"][0][1], 0.0, 0.0]
    cost_hist, step_hist, iters_used = [], [], []

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
            X.append([prev[IDX_E] + prev[IDX_VE] * dt,
                      prev[IDX_N] + prev[IDX_VN] * dt,
                      prev[IDX_VE] + meas["imu"][k][0] * dt,
                      prev[IDX_VN] + meas["imu"][k][1] * dt])

        gnss_idx = [g for g in range(g0, k + 1)]
        X, info = solve_window(cfg, X, gnss_idx, meas["gnss"], meas["imu"],
                               n_iter, kernel, delta)
        cost_hist.extend(info["cost_history"])
        step_hist.extend(info["delta_norms"])
        iters_used.append(info["n_iter_used"])

        for j, g in enumerate(idxs):
            X_hist[g] = list(X[j])
        cur = X_hist[k]

    errors = [horizontal_error([X_hist[k][0], X_hist[k][1]], meas["true_p"][k])
              for k in range(n)]

    return {"states": X_hist, "errors": errors, "cost_history": cost_hist,
            "delta_norms": step_hist, "iters_used": iters_used,
            "name": f"FGO(K={window}, {kernel})"}

# ── 评估口径 ──────────────────────────────────────────────────

def summarize(errors, burn_in=5, cat_thresh=15.0):
    m = metrics(errors, burn_in=burn_in)
    e = errors[burn_in:]
    m["cat_rate"] = sum(1 for x in e if x > cat_thresh) / float(len(e))
    return m

def fmt(m):
    return (f"均值 {m['mean']:7.2f}  RMSE {m['rmse']:7.2f}  "
            f"P95 {m['p95']:8.2f}  最大 {m['max']:9.2f}  超 {15:.0f}m 占比 {m['cat_rate']:6.1%}")

def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"    -> 已写出 {os.path.relpath(path, os.path.dirname(RESULTS))}")

def make_case(model, outlier, seed):
    cfg = Config()
    cfg.n_epochs = N_EPOCHS
    cfg.seed = seed
    st = true_trajectory(cfg)
    meas = make_measurements_noise(cfg, st, model=model, outlier=outlier)
    meas["true_p"] = [s["p"] for s in st]
    return cfg, st, meas

def avg_over_seeds(fn, seeds):
    """跑多个种子，返回各指标均值。"""
    acc = {}
    for s in seeds:
        m = fn(s)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}

def mean_of(values):
    return sum(values) / len(values)

# ── 自检：kernel="l2" 必须与原有实现等价 ──────────────────────

def selftest_equivalence():
    from estimators import build_jacobian_and_residual
    cfg = Config()
    cfg.n_epochs = 150
    st = true_trajectory(cfg)
    meas = make_measurements_noise(cfg, st, model="gaussian")
    meas["true_p"] = [s["p"] for s in st]

    print("=" * 78)
    print("自检：kernel='l2' 的新实现应与 estimators.build_jacobian_and_residual 等价")
    print("=" * 78)
    ok = True
    for K in (2, 3, 10):
        X = [[meas["gnss"][g][0], meas["gnss"][g][1], 0.0, 0.0] for g in range(K)]
        H1, r1, w1 = build_H_r_w(cfg, [x[:] for x in X], list(range(K)),
                                 meas["gnss"], meas["imu"], kernel="l2")
        H2, r2, w2 = build_jacobian_and_residual(cfg, [x[:] for x in X],
                                                 list(range(K)), meas["gnss"],
                                                 meas["imu"])
        dh = max(abs(a - b) for ra, rb in zip(H1, H2) for a, b in zip(ra, rb))
        dr = max(abs(a[0] - b[0]) for a, b in zip(r1, r2))
        dw = max(abs(a - b) for a, b in zip(w1, w2))
        same_shape = (len(H1) == len(H2) and len(H1[0]) == len(H2[0]))
        print(f"  K={K:>3}: 行数 {len(H1)} vs {len(H2)}，列数 {len(H1[0])} vs "
              f"{len(H2[0])}，max|dH|={dh:.2e} max|dr|={dr:.2e} max|dw|={dw:.2e}")
        ok = ok and same_shape and dh == 0.0 and dr == 0.0 and dw == 0.0
    print("  " + ("通过：完全等价。" if ok else "不通过，需排查。"))
    print()
    return ok

# ── 实验主体 ──────────────────────────────────────────────────

CONDITIONS = [
    ("gaussian", False, "纯高斯（同方差基准）"),
    ("laplace", False, "拉普拉斯（重尾）"),
    ("student_t3", False, "Student-t (ν=3)（极重尾）"),
    ("contaminated", False, "污染高斯 (ε=0.1, κ=3)"),
    ("gaussian", True, "原版仿真（高斯 + 12% 离群偏置）"),
]

def expA0_redundancy_check():
    """K=2 恰定时，robust kernel 应当完全无效 —— 因为残差恒为 0。"""
    print("=" * 78)
    print("实验 A0：冗余度检验 —— 恰定（K=2）时 robust kernel 还有用吗？")
    print("=" * 78)
    print("  推理：K=2 时未知数 8、约束 8，系统恰定 -> 解满足全部约束 -> 残差 = 0")
    print("        -> robust 权重依残差计算，残差为零则权重恒等于基准权重")
    print("        -> L2 / Huber / Cauchy 应当给出 *完全相同* 的解。")
    print()
    rows = [["window", "kernel", "mean_m", "max_abs_diff_vs_l2_m"]]
    for K in (2, 3):
        base = None
        for kernel in ("l2", "huber", "cauchy"):
            cfg, st, meas = make_case("gaussian", True, 20260919)
            r = run_fgo_kernel(cfg, meas, window=K, n_iter=N_ITER, kernel=kernel)
            m = summarize(r["errors"])
            if base is None:
                base = r["states"]
                diff = 0.0
            else:
                diff = max(abs(r["states"][k][i] - base[k][i])
                           for k in range(len(base)) for i in range(NX))
            print(f"  K={K}  {kernel:<8} 均值 {m['mean']:7.4f} m   "
                  f"与 L2 的最大状态差 {diff:.3e} m")
            rows.append([K, kernel, f"{m['mean']:.4f}", f"{diff:.6e}"])
    print()
    print("  预期：K=2 差值在 1e-15 量级（舍入误差）= 三者完全相同；")
    print("        K=3 起系统超定、残差非零，robust kernel 才开始起作用。")
    print()
    return rows

def expA1_noise_matrix():
    seeds = [20260919 + s * 7919 for s in range(N_SEEDS)]
    print("=" * 78)
    print("实验 A1：非高斯噪声下 EKF vs FGO，以及 robust kernel 的作用")
    print("=" * 78)
    print(f"  历元数 {N_EPOCHS}｜种子数 {N_SEEDS}｜窗口 {WINDOWS}｜迭代上限 {N_ITER}｜delta = {DELTA}")
    print("  所有「同方差」条件都归一化到 sigma = 6.0 m，只改噪声形状。")
    print()

    rows = [["noise_model", "outlier", "true_sigma_m", "estimator", "window",
             "mean_m", "rmse_m", "p95_m", "max_m", "cat_rate_gt15m",
             "n_iter_used"]]

    summary = {}
    for model, outlier, label in CONDITIONS:
        tsig = effective_sigma(Config(), model=model, outlier=outlier)
        print("-" * 78)
        print(f"【{label}】  真实 sigma = {tsig:.2f} m")
        print("-" * 78)

        def ekf_metrics(s):
            cfg, st, meas = make_case(model, outlier, s)
            m = summarize(run_ekf(cfg, meas)["errors"])
            return {"mean": m["mean"], "rmse": m["rmse"], "p95": m["p95"],
                    "max": m["max"], "cat_rate": m["cat_rate"], "it": 1.0}

        em = avg_over_seeds(ekf_metrics, seeds)
        print(f"  {'EKF':<22} {fmt(em)}")
        rows.append([model, outlier, f"{tsig:.3f}", "EKF", 0] +
                    [f"{em[k]:.4f}" for k in ("mean", "rmse", "p95", "max", "cat_rate")] +
                    [f"{em['it']:.1f}"])
        summary[(model, outlier, "EKF")] = em

        for kernel in ("l2", "huber", "cauchy"):
            for K in WINDOWS:
                def fgo_metrics(s, K=K, kernel=kernel):
                    cfg, st, meas = make_case(model, outlier, s)
                    r = run_fgo_kernel(cfg, meas, window=K, n_iter=N_ITER,
                                       kernel=kernel)
                    m = summarize(r["errors"])
                    return {"mean": m["mean"], "rmse": m["rmse"], "p95": m["p95"],
                            "max": m["max"], "cat_rate": m["cat_rate"],
                            "it": mean_of(r["iters_used"])}
                fm = avg_over_seeds(fgo_metrics, seeds)
                nm = f"FGO K={K} {KERNEL_LABEL[kernel]}"
                print(f"  {nm:<22} {fmt(fm)}  迭代均 {fm['it']:.1f}")
                rows.append([model, outlier, f"{tsig:.3f}", KERNEL_LABEL[kernel], K] +
                            [f"{fm[k]:.4f}" for k in ("mean", "rmse", "p95", "max", "cat_rate")] +
                            [f"{fm['it']:.2f}"])
                summary[(model, outlier, kernel, K)] = fm

        print()

    write_csv(os.path.join(RESULTS, "expA1_noise.csv"), rows)
    return summary, rows

def expA2_iteration_with_kernel():
    """L2 与 robust kernel 在迭代次数上的行为差异。"""
    print("=" * 78)
    print("实验 A2：打开 robust kernel 之后，迭代次数才开始有影响")
    print("=" * 78)
    print("  固定窗口 K=20，扫迭代次数。")
    print("  预期：L2 第 2 步后增量范数已到 1e-11 量级 —— 后续迭代全是空转；")
    print("        Huber / Cauchy 因为权重依残差重算，需要约 20 步才收敛。")
    print()

    iters = [1, 2, 3, 5, 8, 12, 20, 30]
    rows = [["noise_model", "kernel", "n_iter", "mean_m", "rmse_m",
             "max_m", "final_cost"]]
    K = 20
    for model, outlier, label in [("student_t3", False, "Student-t (ν=3)"),
                                  ("gaussian", True, "原版（含离群点）")]:
        cfg, st, meas = make_case(model, outlier, 20260919)
        print(f"  【{label}】")
        for kernel in ("l2", "huber", "cauchy"):
            vals = []
            for it in iters:
                r = run_fgo_kernel(cfg, meas, window=K, n_iter=it, kernel=kernel)
                m = summarize(r["errors"])
                vals.append(m["mean"])
                rows.append([model, kernel, it, f"{m['mean']:.4f}",
                             f"{m['rmse']:.4f}", f"{m['max']:.4f}",
                             f"{r['cost_history'][-1]:.6e}"])
            spread = max(vals) - min(vals)
            first = vals[0]
            last = vals[-1]
            print(f"    {kernel:<8} 均值 {min(vals):7.3f} ~ {max(vals):7.3f} m"
                  f"（极差 {spread:.3e}）｜第 1 次 {first:7.3f} -> 第 30 次 {last:7.3f}"
                  f"（改善 {100*(first-last)/first:5.1f}%）")
        print()

    write_csv(os.path.join(RESULTS, "expA2_iteration_robust.csv"), rows)
    return rows

def expA_main():
    selftest_equivalence()
    expA0_redundancy_check()
    s, _ = expA1_noise_matrix()
    expA2_iteration_with_kernel()
    print("=" * 78)
    print("实验 A 完成。")
    print("=" * 78)
    return s

if __name__ == "__main__":
    expA_main()
