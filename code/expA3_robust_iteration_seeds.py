"""expA3_robust_iteration_seeds.py —— 复核「robust kernel 下迭代越多是否越差」

实验 A2 的单种子结果显示：L2 的迭代次数完全无影响（极差 1e-15），
而 Huber / Cauchy 的极差达 0.1~0.3 m —— 且是 *负影响*（多迭代反而变差）。

这个结论如果成立，值得单独写出来（说明 robust kernel 不能一味迭代到收敛，
否则会过拟合当前的残差实现）。单种子不足以支撑，这里用多种子复核。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from robust import KERNEL_LABEL
from expA_noise import (make_case, run_fgo_kernel, summarize, write_csv,
                        RESULTS, mean_of)

SEEDS = 4
K = 20
ITERS = [1, 2, 3, 5, 8, 12, 20, 30]
CONDS = [("student_t3", False, "Student-t (ν=3)"),
         ("gaussian", True, "原版（高斯 + 12% 离群偏置）")]

def main():
    print("=" * 78)
    print("实验 A3：robust kernel 下，迭代次数到底是正影响还是负影响？（多种子复核）")
    print("=" * 78)
    print(f"  窗口 K={K}｜种子数 {SEEDS}｜迭代次数 {ITERS}")
    print()

    rows = [["noise_model", "kernel", "n_iter", "mean_m", "std_across_seeds",
             "rmse_m", "p95_m", "max_m", "cat_rate_gt15m"]]

    seeds = [20260919 + s * 7919 for s in range(SEEDS)]

    for model, outlier, label in CONDS:
        print("-" * 78)
        print(f"【{label}】")
        print("-" * 78)
        print(f"  {'kernel':<10} {'迭代':>5} | {'均值(m)':>9} {'种子间标准差':>13} "
              f"{'P95':>8} {'最大':>8} {'超15m':>7}")
        print("  " + "-" * 72)

        curve = {}
        for kernel in ("l2", "huber", "cauchy"):
            for it in ITERS:
                means, rmse, p95, mx, cat = [], [], [], [], []
                for s in seeds:
                    cfg, st, meas = make_case(model, outlier, s)
                    r = run_fgo_kernel(cfg, meas, window=K, n_iter=it,
                                       kernel=kernel)
                    m = summarize(r["errors"])
                    means.append(m["mean"]); rmse.append(m["rmse"])
                    p95.append(m["p95"]); mx.append(m["max"])
                    cat.append(m["cat_rate"])
                avg = mean_of(means)
                sd = (sum((x - avg) ** 2 for x in means) / (len(means) - 1)) ** 0.5
                curve[(kernel, it)] = avg
                rows.append([model, KERNEL_LABEL[kernel], it, f"{avg:.4f}",
                             f"{sd:.4f}", f"{mean_of(rmse):.4f}",
                             f"{mean_of(p95):.4f}", f"{mean_of(mx):.4f}",
                             f"{mean_of(cat):.4f}"])
                print(f"  {KERNEL_LABEL[kernel]:<10} {it:>5} | {avg:>9.4f} "
                      f"{sd:>13.4f} {mean_of(p95):>8.3f} {mean_of(mx):>8.3f} "
                      f"{mean_of(cat):>7.1%}")
            best_it = min((it for it in ITERS), key=lambda i: curve[(kernel, i)])
            worst_it = max((it for it in ITERS), key=lambda i: curve[(kernel, i)])
            span = max(curve[(kernel, i)] for i in ITERS) - \
                   min(curve[(kernel, i)] for i in ITERS)
            print(f"    -> {KERNEL_LABEL[kernel]}: 极差 {span:.4f} m，"
                  f"最优迭代 {best_it}（{curve[(kernel, best_it)]:.4f} m），"
                  f"最差迭代 {worst_it}（{curve[(kernel, worst_it)]:.4f} m）")
            print()

    write_csv(os.path.join(RESULTS, "expA3_robust_iteration_seeds.csv"), rows)

    print("=" * 78)
    print("【判读】")
    print("=" * 78)
    print("  · L2 那一行的极差若在 1e-14 量级 -> 迭代次数对 L2 毫无影响（与原复现一致）。")
    print("  · Huber / Cauchy 的极差若显著更大 -> 迭代次数确实开始起作用。")
    print("  · 若最优迭代次数是最小的 1~3 而非最大的 20~30 -> 说明多迭代会 *过拟合残差*，")
    print("    单纯迭代到收敛不是最优策略（这与 GNC / 限制迭代次数的工程做法一致）。")
    print()

if __name__ == "__main__":
    main()
