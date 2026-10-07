"""_smoke_c2.py —— C2 的小规模冒烟测试，确认接口和数值都没问题。"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import expC2_tc_pseudorange as C

print("GDOP_h =", round(C.gdop_horizontal(), 4))
print("sigma_rho =", round(6.0 / C.gdop_horizontal(), 4))

print("\n--- 自检：starts_for(K) 必须与 build_tc_window 的分组一致 ---")
import numpy as np
for K in (1, 2, 3, 10, 20):
    cfg, st, imu, rho, bad, sr = C.make_case_tc(20260919, n_epochs=40)
    X = np.zeros((K, C.NX_TC))
    X[:, 0] = 1.0
    X[:, 4] = 1000.0
    H, r, wb, grp = C.build_tc_window(cfg, X, [rho[0]] * K,
                                      [imu[0]] * max(0, K - 1), sr, "l2", 2.0)
    a = C.starts_for(K)
    b = C.group_starts(grp)
    ok = a.shape == b.shape and bool(np.all(a == b))
    print(f"  K={K:<3} rows={len(grp):<5} groups={len(b):<5} "
          f"starts 一致={ok}  wb 唯一值数={len(set(np.round(wb,12)))}")
    assert ok, f"K={K} 分组不一致"
print("  通过。")

seeds = [20260919, 20260919 + 7919]
print("\n--- 单历元 LS 精度（真实钟差 1000 m，看能不能估回来）---")
for s in seeds:
    cfg, st, imu, rho, bad, sr = C.make_case_tc(s)
    x = C.single_epoch_ls(rho[0])
    err = ((x[0] - st[0]["p"][0]) ** 2 + (x[1] - st[0]["p"][1]) ** 2) ** 0.5
    print(f"  seed={s}  估出位置误差 {err:6.3f} m   钟差 {x[2]:9.2f} m "
          f"(真值 {C.CLOCK_TRUE})  首历元 NLOS 星 {bad[0]}")

print("\n--- FGO-TC 冒烟：K=2/10，n_iter=1/10，三种 kernel ---")
for K in (2, 10):
    for kern in ("l2", "huber", "cauchy"):
        for ni in (1, 10):
            t0 = time.perf_counter()
            cfg, st, imu, rho, bad, sr = C.make_case_tc(20260919)
            r = C.run_fgo_tc(cfg, st, imu, rho, sr, window=K, n_iter=ni,
                             kernel=kern)
            dt = time.perf_counter() - t0
            m = C.summarize(r["errors"])
            print(f"  K={K:<3} {kern:<7} n_iter={ni:<3} -> {dt:6.2f}s  "
                  f"均值 {m['mean']:7.3f} m  P95 {m['p95']:7.2f}  "
                  f"实际迭代 {sum(r['n_iter_used'])/len(r['n_iter_used']):.1f}")
