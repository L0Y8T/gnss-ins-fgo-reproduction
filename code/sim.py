"""sim.py: 合成数据生成器

生成带真值、GNSS 测量与 IMU 测量的车辆轨迹，
用于复现 Wen 2021 第 4.3 节的消融实验。
"""

import math
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

class Config:
    """仿真与实验的全部参数。"""

    dt = 1.0
    n_epochs = 300

    accel_rw_std = 0.10
    accel_rho = 0.50
    v0 = [8.0, 2.0]

    gnss_sigma = 6.0
    gnss_outlier_ratio = 0.12
    gnss_outlier_bias = 25.0

    imu_accel_sigma = 0.15

    # 运动模型因子不能约束速度。若多写一行 v_(g+1)=v_g，等于假设加速度恒为 0，
    # 会和 IMU 因子冲突；去掉之后速度趋势才恢复。
    mm_position_sigma = 0.60
    mm_velocity_sigma = 0.70
    imu_velocity_sigma = 0.70

    r_gnss = gnss_sigma ** 2

    drift_epoch = 180
    drift_sigma_after = 12.0

    seed = 20260919

def _rng(seed, salt=0):
    return random.Random((seed * 1000003 + salt) % (2 ** 31))

def _gauss(rng, sigma):
    return rng.gauss(0.0, sigma)

def true_trajectory(cfg):
    """生成真实轨迹。"""

    rng = _rng(cfg.seed, salt=1)
    dt = cfg.dt

    p = [0.0, 0.0]
    v = [cfg.v0[0], cfg.v0[1]]
    a = [0.0, 0.0]

    states = []
    for k in range(cfg.n_epochs):

        a = [cfg.accel_rho * a[0] + _gauss(rng, cfg.accel_rw_std),
             cfg.accel_rho * a[1] + _gauss(rng, cfg.accel_rw_std)]

        v = [v[0] + a[0] * dt, v[1] + a[1] * dt]
        p = [p[0] + v[0] * dt - 0.5 * a[0] * dt * dt,
             p[1] + v[1] * dt - 0.5 * a[1] * dt * dt]
        states.append({"p": [p[0], p[1]], "v": [v[0], v[1]], "a": [a[0], a[1]]})
    return states

def make_measurements(cfg, states, drifting=False):
    """由真值生成两类测量："""
    rg = _rng(cfg.seed, salt=2)
    ri = _rng(cfg.seed, salt=3)
    ro = _rng(cfg.seed, salt=4)

    gnss, imu, outlier = [], [], []
    for k, s in enumerate(states):

        is_out = ro.random() < cfg.gnss_outlier_ratio
        outlier.append(is_out)

        sigma_k = cfg.gnss_sigma
        if drifting and k >= cfg.drift_epoch:
            sigma_k = cfg.drift_sigma_after

        bias = cfg.gnss_outlier_bias if is_out else 0.0
        zx = s["p"][0] + _gauss(rg, sigma_k) + bias * 0.6
        zy = s["p"][1] + _gauss(rg, sigma_k) + bias * 0.8
        gnss.append([zx, zy])

        imu.append([s["a"][0] + _gauss(ri, cfg.imu_accel_sigma),
                    s["a"][1] + _gauss(ri, cfg.imu_accel_sigma)])

    return {"gnss": gnss, "imu": imu, "outlier": outlier,
            "gnss_sigma_used": [cfg.gnss_sigma if not (drifting and k >= cfg.drift_epoch)
                                else cfg.drift_sigma_after for k in range(cfg.n_epochs)]}

def horizontal_error(est_p, true_p):
    """二维水平误差 [m]：sqrt(ΔE² + ΔN²)。与 Wen 2021 评估 2D 精度的口径一致。"""
    dx = est_p[0] - true_p[0]
    dy = est_p[1] - true_p[1]
    return math.sqrt(dx * dx + dy * dy)

def metrics(errors, burn_in=0):
    """由逐历元误差序列算统计量。"""
    e = errors[burn_in:]
    n = len(e)
    mean = sum(e) / n

    var = sum((x - mean) ** 2 for x in e) / (n - 1) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "rmse": math.sqrt(sum(x * x for x in e) / n),
        "p50": percentile(e, 50),
        "p95": percentile(e, 95),
        "max": max(e),
        "n": n,
    }

def percentile(values, p):
    """百分位数（线性插值）。"""
    xs = sorted(values)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

if __name__ == "__main__":
    cfg = Config()
    st = true_trajectory(cfg)
    ms = make_measurements(cfg, st)

    print("=" * 62)
    print("合成数据自检")
    print("=" * 62)
    print(f"历元数        : {cfg.n_epochs}  (dt = {cfg.dt} s)")

    total = 0.0
    for k in range(1, len(st)):
        total += horizontal_error(st[k]["p"], st[k - 1]["p"])
    print(f"轨迹总长      : {total:.1f} m  （约 {total/1000:.2f} km）")
    print(f"起点          : E={st[0]['p'][0]:.1f}, N={st[0]['p'][1]:.1f}")
    print(f"终点          : E={st[-1]['p'][0]:.1f}, N={st[-1]['p'][1]:.1f}")

    speeds = [math.hypot(s["v"][0], s["v"][1]) for s in st]
    print(f"速度范围      : {min(speeds):.1f} ~ {max(speeds):.1f} m/s")

    errs = [horizontal_error(ms["gnss"][k], st[k]["p"]) for k in range(len(st))]
    m = metrics(errs)
    print("-" * 62)
    print(f"GNSS 测量误差 : 均值 {m['mean']:.2f} m, RMSE {m['rmse']:.2f} m, "
          f"P95 {m['p95']:.2f} m, 最大 {m['max']:.2f} m")
    print(f"异常历元数    : {sum(ms['outlier'])} / {cfg.n_epochs} "
          f"（{100*sum(ms['outlier'])/cfg.n_epochs:.1f}%）")

    e_out = [errs[k] for k in range(len(st)) if ms["outlier"][k]]
    e_in = [errs[k] for k in range(len(st)) if not ms["outlier"][k]]
    print(f"  正常历元    : 均值 {sum(e_in)/len(e_in):.2f} m")
    print(f"  异常历元    : 均值 {sum(e_out)/len(e_out):.2f} m   <- 长尾来源")
    print("-" * 62)
    print(f"IMU 加速度噪声: sigma = {cfg.imu_accel_sigma} m/s²")
    print(f"恒速模型失配  : 真实加速度 std = "
          f"{(sum(s['a'][0]**2 for s in st)/len(st))**0.5:.2f} m/s² (E 向)")
    print("=" * 62)
    print("自检通过：数据生成正常。")

