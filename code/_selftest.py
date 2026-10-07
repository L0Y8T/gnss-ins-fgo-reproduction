"""_selftest.py —— 快速自检：确认 EKF 与 FGO 都能跑，并检验关键理论预期

运行：python _selftest.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sim import Config, true_trajectory, make_measurements, metrics
from estimators import run_ekf, run_fgo, prepare_meas

def main():
    cfg = Config()
    cfg.n_epochs = 120

    true_st = true_trajectory(cfg)
    meas = prepare_meas(cfg, true_st, make_measurements(cfg, true_st))

    print("=" * 70)
    print("自检 1：EKF")
    print("=" * 70)
    r_ekf = run_ekf(cfg, meas)
    m = metrics(r_ekf["errors"], burn_in=5)
    print(f"  {r_ekf['name']:24s} 均值 {m['mean']:6.2f} m   "
          f"标准差 {m['std']:6.2f} m   P95 {m['p95']:6.2f} m")

    print()
    print("=" * 70)
    print("自检 2：FGO 单次迭代 vs 多次迭代（窗口 = 5）")
    print("=" * 70)
    r1 = run_fgo(cfg, meas, window=5, n_iter=1)
    m1 = metrics(r1["errors"], burn_in=5)
    print(f"  {r1['name']:24s} 均值 {m1['mean']:6.2f} m   标准差 {m1['std']:6.2f} m")
    r5 = run_fgo(cfg, meas, window=5, n_iter=5)
    m5 = metrics(r5["errors"], burn_in=5)
    print(f"  {r5['name']:24s} 均值 {m5['mean']:6.2f} m   标准差 {m5['std']:6.2f} m")
    print(f"  -> 迭代 1 次与 5 次的差别：{abs(m1['mean']-m5['mean']):.3e} m")

    print()
    print("=" * 70)
    print("自检 3：观察一轮窗口求解的代价函数变化（验证「一步收敛」）")
    print("=" * 70)
    print("  （下面用窗口 = 10、迭代 3 次，打印每轮迭代的代价函数）")
    rv = run_fgo(cfg, meas, window=10, n_iter=3, verbose=False)

    from estimators import fgo_solve_window
    K = 10
    X = [[0.0, 0.0, 0.0, 0.0] for _ in range(K)]

    for j in range(K):
        X[j] = [meas["true_p"][j][0], meas["true_p"][j][1], 0.0, 0.0]
    gidx = list(range(K))
    X2, info = fgo_solve_window(cfg, [list(x) for x in X], gidx,
                                meas["gnss"], meas["imu"], 3, verbose=True)
    print(f"  代价函数历史: {[f'{c:.4e}' for c in info['cost_history']]}")

    print()
    print("=" * 70)
    print("自检 4：FGO 不同窗口大小的趋势")
    print("=" * 70)
    for K in [2, 5, 10, 30, cfg.n_epochs]:
        r = run_fgo(cfg, meas, window=K, n_iter=1)
        mm = metrics(r["errors"], burn_in=5)
        print(f"  窗口 {K:4d} 历元: 均值 {mm['mean']:6.2f} m   "
              f"标准差 {mm['std']:6.2f} m   P95 {mm['p95']:6.2f} m")

    print()
    print("自检完成。")

if __name__ == "__main__":
    main()

