"""expC3_iteration_multiseed.py —— 实验 C3：多种子钉死「多迭代反而更差」

背景（来自 A3 / A4）：
  A3 观察到 Cauchy 在 n_iter=1 时最好、迭代到收敛反而变差（4.4498 m -> 4.8097 m）。
  但 A3 只有 6 个种子、只扫了 K=2/3/10/20 中的一部分，且没有解释「为什么更差」。

本实验做两件事：
  ① 把结论钉死：30 个种子，n_iter 从 1 扫到 30，每个 (K, kernel) 做配对比较，
     给出胜出种子数与配对均值差 —— 排除抽样偶然。
  ② 给出机制：在**最终解**上重建窗口，读出 GNSS 因子的 robust 权重与鲁棒代价。
     若「鲁棒代价下降」而「真实误差上升」同时发生，
     则说明 IRLS 收敛到了鲁棒代价的另一个（更差的）极小点 ——
     即 Cauchy 的 rho 非凸，迭代把它推进了错误的盆地。
     Cauchy 的 rho 确实是**非凸**的，这是本实验要检验的机制假设。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from expA_noise import (make_case, run_fgo_kernel, summarize, write_csv,
                        RESULTS, N_SEEDS, mean_of)
from robust import kernel_weight

N_ITER_LIST = [1, 2, 3, 5, 10, 15, 20, 30]
WINDOWS = [2, 3, 10, 20]
KERNELS = ["l2", "huber", "cauchy"]
DELTA = 2.0
BIG_SEEDS = 30

CONDITIONS = [
    ("gaussian", True, "原版仿真（高斯 + 12% 离群偏置）", ["l2", "huber", "cauchy"]),
    ("gaussian", False, "纯高斯（无离群点，对照）", ["l2", "cauchy"]),
]


def diagnose(cfg, meas, states, K, kernel, delta):
    """在最终解上重建每个窗口，读出 GNSS 因子的平均 robust 权重与鲁棒代价。

    返回 (平均权重比, 平均每历元鲁棒代价)。权重比 = w_robust / w_base。
      = 1.0 表示 robust 没起作用；
      < 1.0 表示被降权了（离群因子被压住，但正常因子也可能被误压）。
    """
    base = 1.0 / cfg.r_gnss
    ratios, costs = [], []
    for k in range(K - 1, cfg.n_epochs):
        g0 = k - K + 1
        for g in range(g0, k + 1):
            Xg = states[g]
            zg = meas["gnss"][g]
            rv = [Xg[0] - zg[0], Xg[1] - zg[1]]
            ww = kernel_weight(rv, base, kernel, delta)
            ratios.append(ww / base)
            costs.append(ww * (rv[0] ** 2 + rv[1] ** 2))
    return mean_of(ratios), mean_of(costs)


def run_one(model, outlier, seed, K, kernel, n_iter, delta=DELTA):
    cfg, st, meas = make_case(model, outlier, seed)
    r = run_fgo_kernel(cfg, meas, window=K, n_iter=n_iter,
                       kernel=kernel, delta=delta)
    m = summarize(r["errors"])
    ratio, cost = diagnose(cfg, meas, r["states"], K, kernel, delta)
    return m["mean"], m["p95"], m["cat_rate"], ratio, cost, mean_of(r["iters_used"])


def sign_test_p(wins, n):
    """双侧符号检验的精确 p 值（二项分布，p=0.5）。"""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main():
    print("=" * 88)
    print("实验 C3：多种子钉死「多迭代反而更差」，并给出机制")
    print("=" * 88)
    print(f"  种子 {BIG_SEEDS}｜K {WINDOWS}｜n_iter {N_ITER_LIST}｜delta = {DELTA}")
    print("  配对比较：同一个种子下 n_iter=k 与 n_iter=1 的 mean 误差之差。")
    print()

    rows = [["noise_model", "outlier", "window", "kernel", "n_iter",
             "mean_m", "p95_m", "cat_rate_gt15m", "iters_used",
             "gnss_weight_ratio", "gnss_robust_cost",
             "paired_diff_vs_iter1_m", "paired_win_seeds", "sign_test_p"]]

    table = {}
    per_seed = {}

    for model, outlier, label, kernels in CONDITIONS:
        seeds = [20260919 + s * 7919 for s in range(BIG_SEEDS)]
        print("-" * 88)
        print(f"【{label}】")
        print("-" * 88)

        for K in WINDOWS:
            for kernel in kernels:
                res = {}
                for ni in N_ITER_LIST:
                    acc = {k: [] for k in
                           ("mean", "p95", "cat", "ratio", "cost", "used")}
                    for s in seeds:
                        mean, p95, cat, ratio, cost, used = run_one(
                            model, outlier, s, K, kernel, ni)
                        acc["mean"].append(mean)
                        acc["p95"].append(p95)
                        acc["cat"].append(cat)
                        acc["ratio"].append(ratio)
                        acc["cost"].append(cost)
                        acc["used"].append(used)
                    res[ni] = {k: mean_of(v) for k, v in acc.items()}
                    per_seed[(model, outlier, K, kernel, ni)] = acc["mean"]

                print(f"  K={K:<3} {kernel}")
                for ni in N_ITER_LIST:
                    r = res[ni]
                    print(f"      n_iter={ni:<3} 均值 {r['mean']:7.4f} m  "
                          f"P95 {r['p95']:7.3f}  超15m {r['cat']:5.1%}  "
                          f"实际迭代 {r['used']:5.1f}  "
                          f"GNSS权重 {r['ratio']:.4f}  鲁棒代价 {r['cost']:8.2f}")
                    rows.append([model, outlier, K, kernel, ni,
                                 f"{r['mean']:.4f}", f"{r['p95']:.4f}",
                                 f"{r['cat']:.4f}", f"{r['used']:.2f}",
                                 f"{r['ratio']:.5f}", f"{r['cost']:.4f}",
                                 "", "", ""])
                    table[(model, outlier, K, kernel, ni)] = r
                print()

    write_csv(os.path.join(RESULTS, "expC3_iteration_multiseed.csv"), rows)

    # ── 配对对比：n_iter=1 vs n_iter=30（复用上面已经跑出来的逐种子数据）──
    print("=" * 88)
    print("【C3 ①：配对比较 —— n_iter=1 是否为最优？】")
    print("=" * 88)
    print(f"  {'噪声条件':<22} {'K':>3} {'kernel':>8} | "
          f"{'iter=1':>9} {'iter=30':>9} {'差':>9} | {'1 更好的种子数':>14} {'p':>9}")
    print("  " + "-" * 88)

    paired_rows = [["noise_model", "outlier", "window", "kernel",
                    "mean_iter1_m", "mean_iter30_m", "diff_m",
                    "seeds_iter1_better", "n_seeds", "sign_test_p"]]
    n_seeds = BIG_SEEDS
    for model, outlier, label, kernels in CONDITIONS:
        tag = "原版(含离群)" if outlier else model
        for K in WINDOWS:
            for kernel in kernels:
                e1 = per_seed[(model, outlier, K, kernel, 1)]
                e30 = per_seed[(model, outlier, K, kernel, 30)]
                diffs = [b - a for a, b in zip(e1, e30)]   # >0 表示 iter=1 更好
                wins = sum(1 for d in diffs if d > 0)
                p = sign_test_p(wins, n_seeds)
                print(f"  {tag:<22} {K:>3} {kernel:>8} | "
                      f"{mean_of(e1):>9.4f} {mean_of(e30):>9.4f} "
                      f"{mean_of(diffs):>+9.4f} | {wins:>10}/{n_seeds:<3} "
                      f"{p:>9.2e}")
                paired_rows.append([model, outlier, K, kernel,
                                    f"{mean_of(e1):.4f}", f"{mean_of(e30):.4f}",
                                    f"{mean_of(diffs):.4f}", wins, n_seeds,
                                    f"{p:.3e}"])
        print()

    write_csv(os.path.join(RESULTS, "expC3_paired_iter1_vs_iter30.csv"),
              paired_rows)

    print("=" * 88)
    print("【C3 ②：机制 —— 鲁棒代价 vs 真实误差，随迭代次数怎么走】")
    print("=" * 88)
    print("  若「鲁棒代价下降」与「真实误差上升」同时出现，")
    print("  说明 IRLS 收敛到了鲁棒代价的更差极小点（Cauchy 的 rho 非凸）。")
    print()
    print(f"  {'K':>3} {'kernel':>8} | {'iter=1 代价':>12} {'iter=30 代价':>13} "
          f"{'代价变化':>10} | {'iter=1 误差':>12} {'iter=30 误差':>13} "
          f"{'误差变化':>10} | {'结论':>10}")
    print("  " + "-" * 108)
    for K in WINDOWS:
        for kernel in ("huber", "cauchy"):
            a = table[("gaussian", True, K, kernel, 1)]
            b = table[("gaussian", True, K, kernel, 30)]
            dc = 100.0 * (b["cost"] - a["cost"]) / a["cost"]
            de = 100.0 * (b["mean"] - a["mean"]) / a["mean"]
            if dc < 0 and de > 0:
                verdict = "机制成立"
            elif dc < 0 and de < 0:
                verdict = "代价降误差也降"
            else:
                verdict = "不符"
            print(f"  {K:>3} {kernel:>8} | {a['cost']:>12.2f} {b['cost']:>13.2f} "
                  f"{dc:>+9.1f}% | {a['mean']:>12.4f} {b['mean']:>13.4f} "
                  f"{de:>+9.1f}% | {verdict:>10}")
    print()


if __name__ == "__main__":
    main()
