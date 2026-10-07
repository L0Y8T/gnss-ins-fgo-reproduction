"""expA4_robust_best.py —— robust kernel 在「最优迭代次数」下的完整网格

实验 A3 发现：robust kernel 的收益在迭代 1 次时最大，迭代到收敛反而变差。
A1 主表用的是 n_iter=15（接近收敛），因此 *低估* 了 robust kernel 的实际收益。
本脚本把窗口网格在 n_iter=1 下重跑一遍，并做配对显著性检验。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from expA_noise import (make_case, run_fgo_kernel, summarize, write_csv,
                        RESULTS, CONDITIONS, WINDOWS, N_SEEDS, mean_of)
from estimators import run_ekf
from robust import effective_sigma
from sim import Config

FIELDS = ("mean", "rmse", "p95", "max", "cat_rate")


def collect(model, outlier, seeds, K, kernel, n_iter):
    """返回各指标在多个种子上的平均值。"""
    acc = {f: [] for f in FIELDS}
    per_seed_mean = []
    for s in seeds:
        cfg, st, meas = make_case(model, outlier, s)
        r = run_fgo_kernel(cfg, meas, window=K, n_iter=n_iter, kernel=kernel)
        m = summarize(r["errors"])
        for f in FIELDS:
            acc[f].append(m[f])
        per_seed_mean.append(m["mean"])
    avg = {f: mean_of(acc[f]) for f in FIELDS}
    return avg, per_seed_mean


def main():
    seeds = [20260919 + s * 7919 for s in range(N_SEEDS)]
    print("=" * 78)
    print("实验 A4：robust kernel 在 n_iter=1（其最优设置）下的完整网格")
    print("=" * 78)
    print(f"  种子数 {N_SEEDS}｜窗口 {WINDOWS}｜robust kernel 用 n_iter=1，L2 用 n_iter=15")
    print()

    rows = [["noise_model", "outlier", "true_sigma_m", "estimator", "window",
             "mean_m", "rmse_m", "p95_m", "max_m", "cat_rate_gt15m",
             "paired_gain_vs_L2_m", "paired_win_seeds"]]
    paired = {}

    for model, outlier, label in CONDITIONS:
        tsig = effective_sigma(Config(), model=model, outlier=outlier)
        print("-" * 78)
        print(f"【{label}】  真实 sigma = {tsig:.2f} m")
        print("-" * 78)

        e_mean = []
        for s in seeds:
            cfg, st, meas = make_case(model, outlier, s)
            e_mean.append(summarize(run_ekf(cfg, meas)["errors"])["mean"])
        print(f"  {'EKF':<30} 均值 {mean_of(e_mean):7.3f} m")
        rows.append([model, outlier, f"{tsig:.3f}", "EKF", 0,
                     f"{mean_of(e_mean):.4f}", "", "", "", "", "", ""])

        for K in WINDOWS:
            l2, l2_ps = collect(model, outlier, seeds, K, "l2", 15)
            print(f"  {'FGO K=%-3d L2' % K:<30} 均值 {l2['mean']:7.3f} m  "
                  f"P95 {l2['p95']:6.2f}  最大 {l2['max']:6.2f}  "
                  f"超15m {l2['cat_rate']:5.1%}")
            rows.append([model, outlier, f"{tsig:.3f}", "L2", K] +
                        [f"{l2[f]:.4f}" for f in FIELDS] + ["", ""])

            for kernel in ("huber", "cauchy"):
                rk, rk_ps = collect(model, outlier, seeds, K, kernel, 1)
                diffs = [a - b for a, b in zip(l2_ps, rk_ps)]
                win = sum(1 for d in diffs if d > 0)
                gain = 100.0 * mean_of(diffs) / l2["mean"]
                print(f"  {'FGO K=%-3d %s(iter=1)' % (K, kernel):<30} "
                      f"均值 {rk['mean']:7.3f} m  P95 {rk['p95']:6.2f}  "
                      f"最大 {rk['max']:6.2f}  超15m {rk['cat_rate']:5.1%}   "
                      f"配对 {mean_of(diffs):+.3f} m（{gain:+5.1f}%）"
                      f"  胜 {win}/{len(seeds)}")
                paired[(model, outlier, kernel, K)] = (mean_of(diffs), win,
                                                       len(seeds), gain)
                rows.append([model, outlier, f"{tsig:.3f}", kernel, K] +
                            [f"{rk[f]:.4f}" for f in FIELDS] +
                            [f"{mean_of(diffs):.4f}", f"{win}/{len(seeds)}"])
        print()

    write_csv(os.path.join(RESULTS, "expA4_robust_best.csv"), rows)

    print("=" * 78)
    print("【配对显著性汇总】robust(n_iter=1) 相对 L2 的改善")
    print("=" * 78)
    print(f"  {'噪声模型':<22} {'K':>3} | {'Huber 配对差':>13} {'胜出':>7} | "
          f"{'Cauchy 配对差':>13} {'胜出':>7}")
    print("  " + "-" * 78)
    for model, outlier, label in CONDITIONS:
        tag = "原版(含离群)" if outlier else model
        for K in WINDOWS:
            h = paired[(model, outlier, "huber", K)]
            c = paired[(model, outlier, "cauchy", K)]
            print(f"  {tag:<22} {K:>3} | {h[0]:>+13.3f} {h[1]:>4}/{h[2]:<2} | "
                  f"{c[0]:>+13.3f} {c[1]:>4}/{c[2]:<2}")
    print()
    print("  配对差为正 = robust 更好；「胜出」是种子中 robust 胜出的个数。")
    print("  若某一行全部种子都胜出，说明不是抽样偶然。")
    print()


if __name__ == "__main__":
    main()
