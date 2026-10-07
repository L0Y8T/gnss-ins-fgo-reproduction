"""expC1_delta_sweep.py —— 实验 C1：robust kernel 的 delta 敏感性扫描

背景（来自 A3 / A4）：
  robust kernel 在本套仿真里的收益在 n_iter=1 时最大，且只在 K>=3（超定）时存在。
  但 A3/A4 全程用的是 delta = 2.0 —— 一个**由我选定**的调参常数。

本实验要回答的质疑：
  「robust kernel 的收益是不是靠调 delta 调出来的？」
  → 把 delta 扫一遍。若收益在整个 delta 区间内稳定，则不是调参巧合；
    若只在 delta=2.0 附近出现，则 A4 的结论必须降级为「需调参」。

设计：
  delta ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 8.0}
  kernel ∈ {huber, cauchy}
  对照 = L2（无 delta）
  K ∈ {3, 10, 20}（跳过 K=2：A0 已证明恰定时 robust 权重恒等于基准）
  n_iter 固定为 1（A3 测出的最优），另在 K=20 上补一组 n_iter=15 看交互
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from expA_noise import (make_case, run_fgo_kernel, summarize, write_csv,
                        RESULTS, CONDITIONS, N_SEEDS, mean_of)

DELTAS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
KERNELS = ["huber", "cauchy"]
WINDOWS = [3, 10, 20]
FIELDS = ("mean", "rmse", "p95", "max", "cat_rate")


def collect(model, outlier, seeds, K, kernel, n_iter, delta):
    """返回各指标在多颗种子上的平均值，以及每个种子的 mean 误差（供配对比较）。"""
    acc = {f: [] for f in FIELDS}
    per_seed_mean = []
    for s in seeds:
        cfg, st, meas = make_case(model, outlier, s)
        r = run_fgo_kernel(cfg, meas, window=K, n_iter=n_iter,
                           kernel=kernel, delta=delta)
        m = summarize(r["errors"])
        for f in FIELDS:
            acc[f].append(m[f])
        per_seed_mean.append(m["mean"])
    return ({f: mean_of(acc[f]) for f in FIELDS}, per_seed_mean)


def main():
    seeds = [20260919 + s * 7919 for s in range(N_SEEDS)]
    print("=" * 84)
    print("实验 C1：robust kernel 的 delta 敏感性扫描")
    print("=" * 84)
    print(f"  种子 {N_SEEDS}｜K {WINDOWS}｜n_iter = 1（另在 K=20 补 n_iter=15）")
    print(f"  delta 扫描 {DELTAS}")
    print("  问题：A4 的收益是稳定的，还是 delta=2.0 的调参巧合？")
    print()

    rows = [["noise_model", "outlier", "window", "n_iter", "kernel", "delta",
             "mean_m", "rmse_m", "p95_m", "max_m", "cat_rate_gt15m",
             "gain_vs_L2_pct", "win_seeds_over_L2"]]

    summary = {}

    for model, outlier, label in CONDITIONS:
        print("-" * 84)
        print(f"【{label}】")
        print("-" * 84)

        for n_iter in (1, 15):
            windows = WINDOWS if n_iter == 1 else [20]
            for K in windows:
                l2, l2_ps = collect(model, outlier, seeds, K, "l2", n_iter, 2.0)
                print(f"  K={K:<3} n_iter={n_iter:<3} {'L2':<8} "
                      f"均值 {l2['mean']:7.3f} m   P95 {l2['p95']:6.2f}   "
                      f"超15m {l2['cat_rate']:5.1%}")
                rows.append([model, outlier, K, n_iter, "l2", "-"] +
                            [f"{l2[f]:.4f}" for f in FIELDS] + ["-", "-"])

                for kernel in KERNELS:
                    for delta in DELTAS:
                        r, r_ps = collect(model, outlier, seeds, K, kernel,
                                          n_iter, delta)
                        diffs = [a - b for a, b in zip(l2_ps, r_ps)]
                        win = sum(1 for d in diffs if d > 0)
                        gain = 100.0 * mean_of(diffs) / l2["mean"]
                        mark = " <<<" if delta == 2.0 else ""
                        print(f"  K={K:<3} n_iter={n_iter:<3} "
                              f"{kernel+' d='+str(delta):<12} "
                              f"均值 {r['mean']:7.3f} m   P95 {r['p95']:6.2f}   "
                              f"超15m {r['cat_rate']:5.1%}   "
                              f"配对 {mean_of(diffs):+6.3f} m ({gain:+5.1f}%)  "
                              f"胜 {win}/{len(seeds)}{mark}")
                        rows.append([model, outlier, K, n_iter, kernel, delta] +
                                    [f"{r[f]:.4f}" for f in FIELDS] +
                                    [f"{gain:.2f}", f"{win}/{len(seeds)}"])
                        summary[(model, outlier, K, n_iter, kernel, delta)] = gain
                print()

    write_csv(os.path.join(RESULTS, "expC1_delta_sweep.csv"), rows)

    print("=" * 84)
    print("【C1 汇总结论：同一个 (K, kernel) 下，gain 随 delta 的变化范围】")
    print("=" * 84)
    print(f"  {'噪声条件':<24} {'K':>3} {'kernel':>8} | "
          f"{'min gain':>9} {'max gain':>9} {'跨度':>8} | {'是否全程为正':>12}")
    print("  " + "-" * 84)
    for model, outlier, label in CONDITIONS:
        tag = "原版(含离群)" if outlier else model
        for K in WINDOWS:
            for kernel in KERNELS:
                gs = [summary[(model, outlier, K, 1, kernel, d)] for d in DELTAS]
                span = max(gs) - min(gs)
                allpos = "是" if min(gs) > 0 else "**否**"
                print(f"  {tag:<24} {K:>3} {kernel:>8} | "
                      f"{min(gs):>+9.1f} {max(gs):>+9.1f} {span:>8.1f} | "
                      f"{allpos:>12}")
        print()

    print("  gain = robust 相对 L2 的平均误差改善（%），正 = robust 更好。")
    print("  判据：若「跨度」很小且「全程为正」，则收益不依赖 delta 的取值。")
    print()


if __name__ == "__main__":
    main()
